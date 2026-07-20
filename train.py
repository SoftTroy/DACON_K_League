# -*- coding: utf-8 -*-
import os
import sys
import warnings
import random
import pickle
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

# 경고 무시
warnings.filterwarnings('ignore')

# ==========================================
# 1. 환경 설정 및 재현성 확보
# ==========================================

# 경로 설정
DATA_DIR = './data'
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 모델 로드 설정
BASE_MODEL_NAME = 'tabpfn-v2.5-regressor-v2.5_default.ckpt'
LOCAL_BASE_MODEL_PATH = os.path.join(MODEL_DIR, BASE_MODEL_NAME)

if os.path.exists(LOCAL_BASE_MODEL_PATH):
    print(f"✅ Local Base Model Found: {LOCAL_BASE_MODEL_PATH}")
    os.environ['TABPFN_MODEL_PATH'] = LOCAL_BASE_MODEL_PATH
    os.environ['HF_HUB_OFFLINE'] = '1'
else:
    print(f"⚠️ Local Base Model NOT found at {LOCAL_BASE_MODEL_PATH}")
    print("   System will attempt to download the model from HuggingFace.")

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')
TEST_FOLDER_PATH = os.path.join(DATA_DIR, 'test')
MATCH_INFO_PATH = os.path.join(DATA_DIR, 'match_info.csv')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==========================================
# 2. TabPFN 패치 및 유틸리티
# ==========================================
try:
    import tabpfn_common_utils.telemetry.interactive.flows as flows
    flows._trigger_prompts = lambda *args, **kwargs: False
except Exception:
    pass

try:
    from tabpfn.preprocessors.kdi_transformer import KDITransformerWithNaN
    def explicit_init(self, alpha=1.0, output_distribution='normal'):
        super(KDITransformerWithNaN, self).__init__(alpha=alpha, output_distribution=output_distribution)
        self.alpha = alpha
        self.output_distribution = output_distribution
    KDITransformerWithNaN.__init__ = explicit_init
except ImportError:
    pass

# ==========================================
# 3. 데이터 로드 및 전처리
# ==========================================

def read_csv_file(args):
    """
    path: test.csv의 'path' 컬럼 값 (예: ./test/153363/153363_1.csv)
    base_folder: 실제 데이터 폴더 경로 (예: ./data/test)
    """
    file_path, base_folder = args
    try:
        # 경로 정제
        if file_path.startswith('./test/'):
            rel_path = file_path[7:]
        elif file_path.startswith('./test'):
            rel_path = file_path[6:].lstrip(os.path.sep)
        else:
            rel_path = file_path

        full_path = os.path.join(base_folder, rel_path)

        # 중복 폴더 구조 대응
        if not os.path.exists(full_path):
            nested_path = os.path.join(base_folder, 'test', rel_path)
            if os.path.exists(nested_path):
                full_path = nested_path

        # 파일 존재 여부 최종 확인
        if not os.path.exists(full_path):
            # 디버깅을 위해 첫 번째 실패 사례만 출력하고 싶을 수 있음
            return None

        return pd.read_csv(full_path, engine='pyarrow', dtype_backend='pyarrow')
    except Exception as e:
        return None

def load_test_data_fast(test_csv_path, test_folder_path, max_workers=16):
    test_meta_df = pd.read_csv(test_csv_path)
    tasks = [(row['path'], test_folder_path) for _, row in test_meta_df.iterrows()]

    print(f"   > Loading {len(tasks)} files from {test_folder_path}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(read_csv_file, tasks), total=len(tasks), desc="Loading Test Data"))

    valid_dfs = [df for df in results if df is not None]

    # [수정] 데이터 로드 실패 시 즉시 에러 발생
    if not valid_dfs:
        print(f"❌ Error: No files loaded! Check if data is unzipped correctly in {test_folder_path}")
        raise ValueError("No files loaded.")

    print(f"   ✅ Successfully loaded {len(valid_dfs)} / {len(tasks)} files.")
    return pd.concat(valid_dfs, ignore_index=True), test_meta_df

def assign_tactical_zone(x, y):
    if x < 35: x_zone = 0
    elif x < 70: x_zone = 1
    else: x_zone = 2
    if y < 22.66: y_zone = 0
    elif y < 45.33: y_zone = 1
    else: y_zone = 2
    return x_zone * 3 + y_zone

def is_in_penalty_box(x, y):
    return 1 if (x >= 88.5) and (13.84 <= y <= 54.16) else 0

def create_tabular_features(df, is_train=True):
    mode = 'Train' if is_train else 'Test'
    print(f"Generating features for {mode}...")

    df = df.sort_values(['game_episode', 'action_id']).reset_index(drop=True)
    grouped = df.groupby('game_episode')
    features = []

    for name, group in tqdm(grouped, desc=f"Processing {mode}"):
        last_event = group.iloc[-1]
        prev_event = group.iloc[-2] if len(group) > 1 else last_event

        curr_id = last_event['action_id']
        prev_id = prev_event['action_id']
        action_id_gap = max(0, (curr_id - prev_id) - 1)

        time_diff = max(0, last_event['time_seconds'] - prev_event['time_seconds'])
        time_per_missing_action = time_diff / (action_id_gap + 1)

        curr_x, curr_y = last_event['start_x'], last_event['start_y']
        prev_end_x, prev_end_y = prev_event['end_x'], prev_event['end_y']
        prev_dx = prev_event['end_x'] - prev_event['start_x']
        prev_dy = prev_event['end_y'] - prev_event['start_y']

        dist_from_prev = np.sqrt((curr_x - prev_end_x)**2 + (curr_y - prev_end_y)**2) if len(group) > 1 else 0
        prev_dist = np.sqrt(prev_dx**2 + prev_dy**2)

        prev_angle = np.arctan2(prev_dy, prev_dx)
        goal_angle = np.arctan2(34 - curr_y, 105 - curr_x)
        move_angle = np.arctan2(curr_y - prev_end_y, curr_x - prev_end_x)

        direction_consistency = np.cos(prev_angle - goal_angle)
        angle_diff = np.abs(goal_angle - move_angle)
        dist_to_goal = np.sqrt((105 - curr_x)**2 + (34 - curr_y)**2)

        curr_type = str(last_event['type_name'])
        curr_result = str(last_event.get('result_name', 'None'))
        event_type = f"{curr_type}_{curr_result}"
        prev_type_str = str(prev_event['type_name'])
        prev_result_str = str(prev_event.get('result_name', 'None'))
        prev_event_type = f"{prev_type_str}_{prev_result_str}"

        is_same_team = 1 if last_event['team_id'] == prev_event['team_id'] else 0
        zone_id = assign_tactical_zone(curr_x, curr_y)
        in_box = is_in_penalty_box(curr_x, curr_y)

        row = {
            'game_episode': name,
            'game_id': last_event['game_id'],
            'start_x': curr_x, 'start_y': curr_y, 'time_seconds': last_event['time_seconds'],
            'prev_x': prev_event['start_x'], 'prev_y': prev_event['start_y'],
            'event_type': event_type, 'prev_event_type': prev_event_type,
            'period_id': last_event['period_id'], 'team_id': last_event['team_id'], 'player_id': last_event['player_id'],
            'is_same_team': is_same_team, 'zone_id': zone_id, 'in_box': in_box,
            'time_since_prev': time_diff, 'dist_from_prev': dist_from_prev, 'prev_dist': prev_dist,
            'action_id_gap': action_id_gap, 'time_per_missing_action': time_per_missing_action,
            'direction_consistency': direction_consistency, 'move_angle': move_angle, 'angle_diff': angle_diff,
            'dist_to_goal': dist_to_goal, 'angle_to_goal': goal_angle,
        }

        if is_train:
            row['delta_x'] = last_event['end_x'] - last_event['start_x']
            row['delta_y'] = last_event['end_y'] - last_event['start_y']

        features.append(row)

    return pd.DataFrame(features)

def manual_finetune(regressor, X_train, y_train, epochs=20, lr=1e-4, batch_size=1024):
    device = regressor.device
    model = regressor.model_
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    print(f"    > [Fine-tuning] Start Loop (Epochs: {epochs}, LR: {lr})...")
    n_samples = len(X_train)
    indices = np.arange(n_samples)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        batch_count = 0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_X_np = X_train.iloc[batch_indices].values
            batch_y_np = y_train.iloc[batch_indices].values
            batch_X_tensor = torch.tensor(batch_X_np, dtype=torch.float32).to(device)
            batch_y_tensor = torch.tensor(batch_y_np, dtype=torch.float32).to(device)

            model_input = {"main": batch_X_tensor.unsqueeze(0), "y": batch_y_tensor.unsqueeze(0)}
            optimizer.zero_grad()
            try:
                outputs = model(model_input, y=batch_y_tensor.unsqueeze(0))
                preds = outputs[0] if isinstance(outputs, tuple) else outputs
                preds = preds.squeeze()
                targets = batch_y_tensor.squeeze()
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            except Exception:
                continue
    model.eval()
    return regressor

def merge_match_info(feat_df, info_df):
    merged = pd.merge(feat_df, info_df, on='game_id', how='left')
    merged['is_home'] = (merged['team_id'] == merged['home_team_id']).astype(int)
    merged['opponent_id'] = np.where(merged['is_home'] == 1, merged['away_team_id'], merged['home_team_id'])
    return merged

# ==========================================
# 4. 메인 학습 파이프라인
# ==========================================

def main():
    print("\n>>> [1/4] Loading Data...")

    # 1. Train Data 로드
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: Train data not found at {TRAIN_PATH}")
        return
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"   ✅ Train Data Loaded: {len(train_df)} rows")

    # 2. Test Data 로드 (경로 문제 확인용)
    test_df, _ = load_test_data_fast(TEST_CSV_PATH, TEST_FOLDER_PATH)

    match_info = pd.read_csv(MATCH_INFO_PATH)
    match_cols = ['game_id', 'home_team_id', 'away_team_id', 'venue']
    match_info = match_info[match_cols]

    print("\n>>> [2/4] Feature Engineering...")
    train_feat = create_tabular_features(train_df, is_train=True)
    test_feat = create_tabular_features(test_df, is_train=False)

    train_feat = merge_match_info(train_feat, match_info)
    test_feat = merge_match_info(test_feat, match_info)

    # 변수 변환
    target_cols_log = ['prev_dist', 'dist_from_prev', 'time_since_prev', 'time_per_missing_action']
    for col in target_cols_log:
        train_feat[col+'_log1p'] = np.log1p(train_feat[col])
        test_feat[col+'_log1p'] = np.log1p(test_feat[col])

    target_cols_cat = [
        'event_type', 'prev_event_type', 'in_box', 'zone_id',
        'is_same_team', 'period_id', 'team_id', 'player_id',
        'venue', 'opponent_id', 'is_home'
    ]

    for col in target_cols_cat:
        new_col = col + '_cat'
        le = LabelEncoder()
        # Train과 Test 전체 데이터로 fitting
        all_vals = pd.concat([train_feat[col], test_feat[col]]).fillna(-1).astype(str)
        le.fit(all_vals)
        train_feat[new_col] = le.transform(train_feat[col].fillna(-1).astype(str))
        test_feat[new_col] = le.transform(test_feat[col].fillna(-1).astype(str))

    base_features = [
        'start_x', 'start_y', 'time_seconds', 'prev_x', 'prev_y',
        'dist_to_goal', 'angle_to_goal', 'direction_consistency', 'move_angle', 'angle_diff',
        'action_id_gap', 'time_per_missing_action'
    ]
    log_features = [col + '_log1p' for col in target_cols_log]
    cat_features = [col + '_cat' for col in target_cols_cat]
    final_features = base_features + log_features + cat_features

    X = train_feat[final_features]
    y_dx = train_feat['delta_x']
    y_dy = train_feat['delta_y']

    print(f"  > Final Feature Count: {len(final_features)}")
    print(f"  > Total Samples: {len(X)}")

    # ==========================================
    # 5. 모델 학습 및 저장
    # ==========================================
    print("\n>>> [3/4] Starting K-Fold Training & Saving Models...")

    FT_EPOCHS = 20
    FT_LEARNING_RATE = 1e-4
    FT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_dx)):
        print(f"\n[Fold {fold+1}/5] Training...")
        X_train = X.iloc[train_idx]
        y_dx_train = y_dx.iloc[train_idx]
        y_dy_train = y_dy.iloc[train_idx]

        # --- Delta X 학습 ---
        print(f"  > Training Delta X...")
        model_arg = LOCAL_BASE_MODEL_PATH if os.path.exists(LOCAL_BASE_MODEL_PATH) else None

        reg_dx = TabPFNRegressor(
            device=FT_DEVICE,
            model_path=model_arg,
            ignore_pretraining_limits=True,
            fit_mode='low_memory'
        )
        reg_dx.fit(X_train, y_dx_train)
        reg_dx = manual_finetune(reg_dx, X_train, y_dx_train, epochs=FT_EPOCHS, lr=FT_LEARNING_RATE)

        # 모델 가중치 저장
        dx_save_path = os.path.join(MODEL_DIR, f'tabpfn_dx_fold{fold}.pt')
        torch.save(reg_dx.model_.state_dict(), dx_save_path)
        print(f"    >> Saved: {dx_save_path}")

        # --- Delta Y 학습 ---
        print(f"  > Training Delta Y...")
        model_arg = LOCAL_BASE_MODEL_PATH if os.path.exists(LOCAL_BASE_MODEL_PATH) else None

        reg_dy = TabPFNRegressor(
            device=FT_DEVICE,
            model_path=model_arg,
            ignore_pretraining_limits=True,
            fit_mode='low_memory'
        )
        reg_dy.fit(X_train, y_dy_train)
        reg_dy = manual_finetune(reg_dy, X_train, y_dy_train, epochs=FT_EPOCHS, lr=FT_LEARNING_RATE)

        dy_save_path = os.path.join(MODEL_DIR, f'tabpfn_dy_fold{fold}.pt')
        torch.save(reg_dy.model_.state_dict(), dy_save_path)
        print(f"    >> Saved: {dy_save_path}")

    print("\n>>> [4/4] All Training Finished & Models Saved in './models/'")

if __name__ == '__main__':
    main()
