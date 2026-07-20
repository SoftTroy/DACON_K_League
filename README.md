# K-리그 선수 트래킹 데이터 기반 움직임 예측 (TabPFN 활용)

**작성자:** Staticronaldo

---

## License Notice

본 프로젝트는 코드 및 구현 방법을 공유하기 위한 저장소입니다.

> **`test/`(데이터셋) 및 `models/`(학습된 모델 가중치)는 데이터 라이선스 및 배포 정책에 따라 저장소에 포함되어 있지 않습니다.**

프로젝트를 실행하려면 데이터 제공 기관(또는 대회 운영 측)의 라이선스 정책에 따라 적법하게 제공받은 데이터셋과, 해당 데이터로 직접 학습한 모델을 사용해야 합니다.

라이선스 정책을 준수하기 위함이므로 양해 부탁드립니다.

---

## 개요

본 프로젝트는 K-리그 경기 내 선수의 트래킹 데이터를 분석하여 시계열적인 움직임 좌표를 예측합니다.

정형 데이터에 강력한 성능을 보이는 **TabPFN (Tabular Prior-Data Fitted Network)** 모델을 도입하고, x축과 y축의 변화량(`dx`, `dy`)을 각각 학습하여 예측 정밀도를 극대화했습니다.

---

## 분석 방법론

### A. 모델링 전략

- **Model:** TabPFN
- **Target:** 선수의 다음 위치 좌표 자체가 아닌 변화량(`dx`, `dy`)을 예측 대상으로 설정하여 학습 안정성을 확보
- **Ensemble:** 5-Fold Cross Validation을 적용하여 과적합을 방지하고 일반화 성능을 확보
  - `dx` 예측 모델 5개 (`fold0` ~ `fold4`)
  - `dy` 예측 모델 5개 (`fold0` ~ `fold4`)

### B. 데이터 처리

- 별도의 전처리 스크립트 없이 학습 및 추론 단계에서 데이터 로딩과 동시에 피처 엔지니어링을 수행합니다.
- 모든 데이터 처리 과정은 `train.py` 및 `inference.py` 내부에 통합되어 있습니다.

---

## 파일 구조 (File Structure)

```text
📦 Project_Root
┣ 📂 test
┃ ┗ 원본 데이터셋
┃    ├ data_description
┃    ├ match_info
┃    ├ train.csv
┃    └ test.csv
┃
┣ 📂 models
┃ ├ 📜 tabpfn_dx_fold0.pt
┃ ├ 📜 tabpfn_dx_fold1.pt
┃ ├ 📜 tabpfn_dx_fold2.pt
┃ ├ 📜 tabpfn_dx_fold3.pt
┃ ├ 📜 tabpfn_dx_fold4.pt
┃ ├ 📜 tabpfn_dy_fold0.pt
┃ ├ 📜 tabpfn_dy_fold1.pt
┃ ├ 📜 tabpfn_dy_fold2.pt
┃ ├ 📜 tabpfn_dy_fold3.pt
┃ └ 📜 tabpfn_dy_fold4.pt
┃
┣ 📜 train.py
┃ └ 데이터 로드, 전처리 및 TabPFN 모델 학습 (Fold별 저장)
┃
┣ 📜 inference.py
┃ └ 테스트 데이터 로드, 앙상블 추론 및 submission.csv 생성
┃
┣ 📜 requirements.txt
┃ └ 필요 라이브러리 목록
┃
┗ 📜 submission.csv
   └ 최종 제출 파일
```

---

## 실행 방법 (Usage)

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 모델 학습 (Training)

```bash
python train.py
```

- `test/train.csv`를 로드하여 학습을 진행합니다.
- 학습이 완료되면 `models/` 폴더에 Fold별 `.pt` 모델 파일이 저장됩니다.

### 3. 추론 및 결과 생성 (Inference)

```bash
python inference.py
```

- 학습된 모델들을 로드하여 `test/test.csv`에 대한 예측을 수행합니다.
- 최종 결과는 `submission.csv`로 저장됩니다.
