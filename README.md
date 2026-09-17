# 💳 신용카드 이상거래 탐지 : 비용 기반 FDS 의사결정 전략
금융 손실을 최소화하는 이상거래 탐지 시스템 설계
**탐지 성능이 높은 모델이 실제 운영에서도 가장 좋은 선택일까?**

> Elliptic Envelope와 비용 기반 탐지 기준을 적용해, 대조 모델 대비 예상 손실을 **64.3%** 줄일 수 있는 운영안을 도출

## 🎯 Analysis Objective
사기 탐지율을 높이는 것에 그치지 않고,
**"사기를 놓치는 비용(FN)과 정상 거래를 막는 비용(FP) 사이에서 실제 손실을 최소화할 수 있는 모델과 운영 기준을 어떻게 선택할 것인가?"**

## 📌 Project Overview
극단적인 클래스 불균형과 FN/FP의 비대칭적 손실 구조 때문에, 탐지 성능(F1-score)만으로는 실제 운영에 적합한 모델과 기준을 판단할 수 없다는 문제의식에서 출발했습니다.


#### Project Information
- 기간 : 2026. 
- 역할 : 개인 프로젝트
- Tools : Python(scikit-learn, SHAP, PyTorch/TensorFlow)
- Models : Elliptic Envelope, Isolation Forest, Autoencoder
- 핵심 기법 :  Cost-based Threshold Optimization, XAI (SHAP), Unsupervised Anomaly Detection


#### 📊 Data
데이콘 신용카드 사기거래 탐지 경진대회

| Dataset | 설명 | 규모 |
| --- | --- | --- |
| Train | 라벨 없음 (대부분 정상 거래로 추정) | 113,842건 |
| Validation | 정상/사기 라벨 포함 | 28,462건 (정상 28,432 / 사기 30) |
| Test | 라벨 없음 | 142,503건 |

- 변수 : V1~V30 (비식별화된 PCA 변환 변수)


## 🚨 Problem
이상거래 탐지는 다음과 같은 문제 특성으로 인해 분류 성능만으로는 탐지 기준을 결정하기 어려움.

1. **극단적 클래스 불균형** : 정상 거래 99.89% vs 사기 거래 0.11% (Validation 기준 28,432건 vs 30건) → Accuracy만으로는 탐지 성능을 평가하기 어려움

2. **비대칭적 손실 구조** : 금융 서비스에서는 FN과 FP가 발생시키는 비용이 동일하지 않음
    - FN(미탐) : 실제 사기 거래를 정상으로 승인 → 직접적인 금융 손실 → 고비용
    - FP(오탐) : 정상 거래를 사기로 판단 → 고객 불편 및 운영 비용 → 저비용
    - F1-score 최적화는 FN과 FP의 균형을 고려할 뿐, 두 오류의 비용 차이를 반영하지 못함.


3. **미라벨 데이터 환경** : 실제 라벨 지정이 어려운 운영 환경 확장을 고려하면, 비지도 이상 탐지 기법 적용 필요

(Train/Test: Train(113,842건)과 Test(142,503건)는 정상/사기 여부를 알 수 없고, 라벨이 있는 Validation(28,462건)만으로 모델을 검증해야 함)


## 🔑 Key Findings
1. **Elliptic Envelope + 비용 최적화 임계값으로 예상 손실 64.3%(935만원) 절감**
알고리즘 선택 + 임계값 최적화만으로 확보한 수치
|       | Isolation Forest | Elliptic Envelope |
| ----- | ---------------: | ----------------: |
| FN    |              13건 |            **5건** |
| FP    |              31건 |            **4건** |
| 예상 손실 |          1,455만원 |         **520만원** |


2. **F1 최적 지점과 Total Cost 최적 지점은 다르며, 비즈니스 목표에 맞춰 후자를 채택함**
성능이 가장 좋은 모델과 비즈니스 목표에 가장 부합하는 모델은 다를 수 있음 (상세 비교는 아래 Cost-based Threshold Optimization 참고)


3. **SHAP 분석 결과, 모델이 EDA에서 확인된 이상 패턴(V17, V14, V12)을 근거로 판단하고 있음**
성능이 우연히 나온 게 아니라, 데이터의 실제 구조를 학습했다는 신뢰성을 검증



## 🔍 Analysis
### 01. EDA — 사기 거래는 어떤 구조를 가지고 있는가?
- 단일 변수의 상관계수는 |0.3| 이하로 선형 분리가 어려움.
- 사기 거래는 정상 거래와 완전히 분리된 독립 군집이 아님
- 정상 분포의 외곽 /꼬리 영역으로 이탈하는 구조
- 일부 Feature에서 정상 분포의 경계를 벗어나는 패턴

→ 사기 거래가 독립된 군집보다 정상 분포의 경계에서 이탈하는 형태를 보였기 때문에, 정상 분포를 모델링한 뒤 이탈 거래를 탐지하는 비지도 접근이 현 데이터 구조에 적합

### 02. Model Comparison - 어떤 이상 탐지 모델이 적합한가?
| Model | Precision | Recall |  F1 |
| --------------------- | --------: | -------: | -------: |
| Isolation Forest | 0.38 | 0.57 | 0.45 |
| Autoencoder | 0.59 | 0.57 |  0.58 |
| **Elliptic Envelope** |  **0.83** | **0.83** | **0.83** |

*참고(모델 설명)
- `Isolation Forest` : 축 직교 분할 방식이 정상과 사기가 겹치는 경계 구간을 충분히 분리하지 못해 오탐이 많음

- `Autoencoder` : 사기 표본이 극소수여서 정상 거래와 사기 거래의 복원 오차 분포가 충분히 분리되지 않아 사기를 "정상처럼" 복원함

- `Elliptic Envelope`: 정상 거래의 공분산 구조를 기반으로 타원형 경계를 형성해, 현 데이터의 이상치 분포를 가장 안정적으로 구분

> 💡 스케일러 비교 : MinMax / Standard / Robust / None 조합을 비교한 결과, 스케일러에 따른 성능 차이가 거의 없었음. PCA 변환 변수의 특성을 고려하여 불필요한 변환을 최소화하고 No-Scaling을 채택

## 03. ★ Cost-based Threshold Optimization — 어디에서 탐지를 멈출 것인가?
> F1-score가 가장 높은 지점이 실제 손실도 가장 적을까?

**$$\text{Total Cost} = (\text{FN} \times 20) + (\text{FP} \times 1)$$**

- **F1 최적점 ≠ 비용 최적점**
    - FP 5 / FN 5(F1=0.83)가 F1 기준으로는 최적점이지만, FN:FP=20:1의 비용 구조를 반영한 Total Cost 기준으로는 FP 4 / FN 5(F1=0.85)가 최적점.
    - 두 지점의 성능 차이는 미미하지만, 실제 비즈니스 목표(손실 최소화)에 맞춰 후자를 운영 임계값으로 채택함
- FN:FP = 20:1의 상대적 비용을 가정 (사기 미탐 비용을 오탐 비용의 20배로 가정, 사기 1건당 피해액 100만 원 가정)

> Elliptic Envelope의 이상치 점수를 기준으로 탐지 기준을 이동시키며 F1-score와 Total Cost를 비교했습니다. 그 결과 두 기준의 최적 지점은 일치하지 않았고, 이번 분석에서는 실제 손실을 최소화하는 기준을 운영 기준으로 선택

### 04. XAI(SHAP) - 모델은 왜 이 거래를 이상하다고 판단했는가?
SHAP Beeswarm/Waterfall Plot으로 모델이 실제로 어떤 변수를 근거로 사기를 판정하는지 검증 (EDA 결과와 일치하는지 교차 확인)

- Beeswarm Plot 기준 사기 판정에 가장 큰 영향을 미치는 변수는 V17, V14, V12 순 — EDA 단계에서 이미 정상 분포 이탈이 확인됐던 변수와 일치

- 값이 낮을수록(파란색) 사기 방향으로 기여하는 패턴도 EDA의 상관관계 방향성과 일치

- 여러 변수의 상호작용으로 사기 여부를 판단하고 있음을 개별 거래 Waterfall Plot으로 확인


## 🚦 운영 전략 : 위험도 기반 3단계 대응 체계
Elliptic Envelope의 이상치 스코어를 기준으로 거래를 3단계로 자동 분류하여 관제 효율을 높입니다.

| 위험도 | 대응 | 실제 분포 |
| :--- | :--- | :--- |
| **High Risk** | 즉시 차단 + 관제팀 알림 | 사기 거래의 83.33%가 이 구간에 집중 |
| **Medium Risk** | 추가 인증(OTP 등) 후 승인 여부 결정 | 정상/사기 모두 0%대로 거의 없음 |
| **Low Risk** | 정상 승인 | 정상 거래의 99.98%가 이 구간에 안전하게 분류 |

→ 정상과 사기가 서로 다른 위험도 구간에 명확히 분리되어, 무분별한 차단 없이 탐지 성능과 고객 경험을 동시에 확보


## 📁 Structure
```
credit-card-fraud-detection/
│
├── README.md
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   ├── 04_cost_optimization.ipynb
│   ├── 05_xai_shap.ipynb
│   └── 06_operational_strategy.ipynb
│
├── src/
│   └── cost_function.py
│
└── outputs/
    ├── model_comparison.png
    ├── cost_comparison.png
    ├── confusion_matrix.png
    └── shap_summary.png
```

## Analysis Flow

01. [EDA](notebooks/01_eda.ipynb)
    → 이상거래 데이터의 구조와 분포 확인

02. [Model Comparison](notebooks/02_model_comparison.ipynb)
    → Isolation Forest / AutoEncoder / Elliptic Envelope 비교

03. [Hyperparameter Tuning](notebooks/03_hyperparameter_tuning.ipynb)
    → contamination 및 주요 설정값 비교

04. [Cost Optimization](notebooks/04_cost_optimization.ipynb)
    → FN/FP 비용을 반영한 모델 및 탐지 기준 비교

05. [XAI](notebooks/05_xai_shap.ipynb)
    → SHAP을 통한 모델의 주요 판단 요인 분석

06. [Operational Strategy](notebooks/06_operational_strategy.ipynb)
    → 위험도에 따른 FDS 운영 전략 설계
    

## 📁 Repository
📎 References
[Dacon — 신용카드 사기 거래 탐지 AI 경진대회]

## Note
- 본 프로젝트는 데이콘 신용카드 이상거래 탐지 경진대회 데이터를 기반으로 작성되었습니다.
- 당시 대회에서는 상위 4%의 성적을 얻었습니다
- 이번 프로젝트에서는 대회 성능 최적화에서 나아가, **비용 함수 기반 비즈니스 의사결정 프레임워크**로 재설계한 프로젝트 

기존 대회에서는 탐지 성능을 높여 평가 지표에서 좋은 결과를 얻는 것이 목표였고, 그 결과 상위 4%의 성적을 얻었습니다. 이번 분석에서는 여기서 한 단계 더 나아가 "탐지 성능이 높은 모델이 실제 비즈니스를 가정했을 때도 가장 좋은 선택일까?"라는 질문을 던졌습니다. 실제 FDS에서는 사기를 놓치는 FN과 정상 거래를 잘못 차단하는 FP의 비용이 다르기 때문에, 단순히 F1-score가 높은 모델을 선택하는 것만으로는 충분하지 않다고 판단했습니다.(자소서)



## 🧠 What I Learned
모델 성능을 비교하면서 높은 F1-score를 얻는 것과 실제 운영에 적합한 모델을 선택하는 것은 다른 문제라는 점을 확인했습니다.

특히 이상거래 탐지에서는 FN과 FP의 비용이 다르기 때문에, F1-score만으로 모델과 탐지 기준을 결정하기 어렵다고 판단했습니다.

이에 FN과 FP의 상대적 비용을 반영한 Total Cost를 추가적인 평가 기준으로 설정하고, 모델의 탐지 성능과 예상 손실을 함께 비교

또한 contamination과 Anomaly Score cutoff를 설정하는 과정에서 탐지 기준 자체에도 근거가 필요하다는 점을 확인

## ⚠️ (Limitations)
- contamination 및 일부 Score 기준 설정에 Validation 정보를 참고한 부분이 있어 완전한 비지도 학습으로 보기는 어렵습니다.
- 비용 가중치와 피해액은 실제 금융 데이터가 아닌 가정값(운영 도입 시 도메인 금융 데이터를 기반으로 재산정 필요)
- 과거 데이터 기반 분석이므로 실제 운영 환경에서의 Data Drift 및 실시간 운영 환경에 대한 추가 검증이 필요





▼ 참고용

2) contamination 파라미터를 다시 본 이유

Elliptic Envelope의 contamination은 얼핏 "그냥 하이퍼파라미터"처럼 보이지만, 사실상 모델이 내부적으로 갖는 이상치 비율 기준(threshold) 그 자체입니다. 즉 이 값을 튜닝하는 것은 일반적인 하이퍼파라미터 튜닝이 아니라 탐지 기준선을 직접 조정하는 행위에 가깝다는 것을 뒤늦게 인지했고, 이후 이 파라미터를 어떤 기준으로 정당화할지(성능만으로 정당화하지 않기)를 별도로 고민하게 됐습니다.



4) Anomaly Score 컷오프(상위 0.1%) 선택 기준

처음에는 "이 컷오프에서 성능이 제일 잘 나오니까"로 선택하려 했지만, 특정 cutoff에서 성능이 높다는 것만으로 탐지 기준을 정당화하기에는 근거가 약하다고 판단했습니다. 
그래서 Validation에서 확인된 사기 거래 비율(0.11%)을 참고하여 이상치 비율의 범위를 설정하고, 해당 범위 안에서 모델 성능을 비교해 최종 기준을 선택
→ 성능이 가장 좋은 값을 선택하는 것과, 그 값을 왜 사용해야 하는지 설명하는 것은 별개의 문제임을 확인


1) 스케일러 비교: "굳이 스케일링이 필요할까?"

Isolation Forest, AutoEncoder, Elliptic Envelope 각각에 대해 MinMax/Standard/Robust/None 스케일러를 조합해 총 220회 그리드서치를 돌렸는데, 스케일러 종류에 따른 성능 차이가 거의 없었습니다. 처음엔 튜닝이 잘못됐나 싶었지만, 변수 자체가 이미 PCA로 변환된 값(V1~V30)이라 스케일이 어느 정도 유사한 분포를 갖고 있었기 때문이라고 판단했습니다. 최종적으로는 불필요한 데이터 변형을 최소화한다는 원칙에 따라 No-scaling(None)을 채택했습니다. (Top 3 조합 모두 F1 0.83으로 동일 — scaler 선택이 성능을 좌우하지 않는다는 근거)


3) 실은 완전한 비지도 학습이 아니었다는 한계

contamination 값과 이상치 스코어(Anomaly Score)를 몇 %까지 컷오프할지 모두 Validation set의 정답 라벨을 보고 설정했습니다. 이는 엄밀히 말하면 완전한 비지도 학습이 아니라 준지도적(semi-supervised) 성격이 개입된 절충안입니다. 실무에서 라벨이 전혀 없는 상황이라면 이 방식은 그대로 적용할 수 없고, 별도의 held-out 테스트셋으로 재검증이 필요하다는 점을 한계로 남겨두었습니다.
