K-리그 선수 트래킹 데이터 기반 움직임 예측 (TabPFN 활용)
Staticronaldo

<개요>
본 프로젝트는 K-리그 경기 내 선수의 트래킹 데이터를 분석하여 시계열적인 움직임 좌표를 예측합니다. 정형 데이터에 강력한 성능을 보이는 TabPFN(Tabular Prior-Data Fitted Network) 모델을 도입하고, x축과 y축의 변화량(dx, dy)을 각각 학습하여 예측 정밀도를 극대화했습니다.

<분석 방법론>
A. 모델링 전략
- Model: TabPFN
- Target: 선수의 다음 위치 좌표 자체가 아닌, 변화량(dx, dy)을 타겟으로 설정하여 학습 안정성 확보.
- Ensemble: 5-Fold Cross Validation을 적용하여 과적합을 방지하고 일반화 성능 확보.
    - dx 예측 모델 5개 (fold0 ~ fold4)
    - dy 예측 모델 5개 (fold0 ~ fold4)

B. 데이터 처리
- 별도의 전처리 스크립트 없이 학습 및 추론 단계에서 데이터 로딩과 동시에 피처 엔지니어링 수행 (train.py / inference.py 내부 통합).

3. 파일 구조 (File Structure)
📦 Project_Root
 ┣ 📂 test           # 원본 데이터셋 (data_description, match_info, train/test csv 등)
 ┣ 📂 models         # 학습된 TabPFN 모델 가중치 저장소
 ┃ ┣ 📜 tabpfn_dx_fold[0-4].pt  # x축 변화량 예측 모델 (5 Folds)
 ┃ ┗ 📜 tabpfn_dy_fold[0-4].pt  # y축 변화량 예측 모델 (5 Folds)
 ┣ 📜 train.py       # 데이터 로드, 전처리 및 TabPFN 모델 학습 (Fold별 저장)
 ┣ 📜 inference.py   # 테스트 데이터 로드, 앙상블 추론 및 submission.csv 생성
 ┣ 📜 requirements.txt # 필요 라이브러리 목록
 ┗ 📜 submission.csv   # 최종 제출 파일

4. 실행 방법 (Usage)

1. 환경 설정:
   pip install -r requirements.txt

2. 모델 학습 (Training):
   python train.py
   * `test/train.csv`를 로드하여 학습을 진행하며, `models/` 폴더에 fold별 .pt 파일이 저장됩니다.

3. 추론 및 결과 생성 (Inference):
   python inference.py
   * 학습된 모델들을 로드하여 `test/test.csv`에 대한 예측을 수행하고 `submission.csv`를 생성합니다.
