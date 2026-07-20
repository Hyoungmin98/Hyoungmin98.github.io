---
layout: post
title:  "Review: 'Beyond GMV: the relevance of covariance matrix estimation for risk-based portfolio construction'"
date:   2026-07-18 22:38:28 +0900
math: true
categories: Review
tags: Quant ML 
---

**해당 게시글은 Dom et al,(2025)의 "Beyond GMV: the relevance of covariance matrix estimation for risk-based portfolio construction" 논문을 리뷰한 글입니다. 본 게시물에서 인용한 논문 및 자료에 대한 상세 정보는 아래의 링크를 통해 확인하실 수 있습니다.  
[Dom et al., (2020)](https://doi.org/10.1080/14697688.2025.2468268)**   


이 논문은 GMV 포트폴리오의 다양한 제약 하에서 서로 다른 공분산 추정 모델이 어떤 영향을 미치는지, 공분산 추정 모델 선택이 얼마나 중요한지를 보여준다.   
GMV 포트폴리오에 대해 연구한다면 읽어보면 좋을 만한 논문이며, 실제로 내가 진행한 몇몇 실험들에서도 이 논문이 주장했던 것과 유사한 결과가 많이 나왔던 만큼 리뷰할 가치가 있는 논문이라고 생각한다.  


# 1. Introduction  
주식 포트폴리오를 구성하는 데에 있어서 수익률의 variance-covariance(VCV) 행렬은 핵심 요소이다. MVO 및 GMV 포트폴리오에서 약간의 추정 오차만 발생해도 공분산의 역행렬로 변환되는 과정에서 이러한 오차가 증폭되기 때문.    
지금까지 수많은 VCV 추정치가 제안되었으나, long-only 제약을 제외하고는 대부분 GMV 포트폴리오에 추가적인 제약을 두지 않는다.  
→ 이로 인해 실제로 구현했을 때 높은 레버리지, turnover 등의 비현실적인 결과가 도출될 수 있음  

따라서 본 논문은 GMV와 Risk-Parity 포트폴리오를 활용한 다양한 리스크 기반 포트폴리오에서 다양한 VCV 추정 모델들을 비교한다.  
특히 GMV 포트폴리오를 제약 없는 환경에서 복잡한 VCV 추정기를 활용하는 것은 실무적으로 현실성이 떨어질 수 있으니, 현실을 반영한 다양한 제약을 추가하여 성과를 비교한다.  

1990년 1월 ~ 2021년 12월 까지의 미국 시가총액 상위 500개 주식을 사용하여 다양한 위험 기반 포트폴리오 구조에서 VCV 추정기를 평가한다.  
성과 지표로는 포트폴리오의 사후 변동성을 주요 지표로 활용하며, 이외에도 다음의 지표들을 활용한다.  
- risk-adjusted returns  
- asset weight concentration  
- portfolio turnover  
- transaction cost  
- factor exposure  

**이때 VCV 추정기의 일관성은 평가 지표로 고려하지 않는데, 본 논문에서는 이 지표를 활용하기 위해서는 시뮬레이션을 진행해야 하며, 실제 성과에 큰 영향을 미치지 않을 수 있기 때문이라고 언급한다.**  

추가적으로 VCV 추정기는 전통적 추정기와 최신 추정기를 모두 활용하며,   
- shrinkage  
- time-dynamics  
- factor structure  
의 큰 틀을 두고 선택하였다.  
구체적으로 Ledoit and Wolf(2004b, 2024) 의 linear(LS) & nonlinear(NLS) shrinkage, Engle(2002) 의 dynamic conditional correlation(DCC) 모델, 이들을 결합한 DCC-NLS, RiskMetrics(RM), 두 가지의 factor structure를 적용한다.  

제약조건의 경우 unconstrained, long-only, maximum-weight constraint를 적용하며, transaction cost penalty를 추가로 고려한다.  
포트폴리오는 GMV 포트폴리오와 리스크 패리티를 사용하며, 리스크 패리티는 equal risk contribution(ERC) 와 Hierarchical risk parity(HRP) 포트폴리오를 사용한다.  

본 논문의 주요 발견은 다음과 같다.  
- 비제약 GMV 포트폴리오의 경우, NLS 및 DCC-NLS 추정기가 더 나은 성과를 보임.  
- 추정기 간 총 노출(gross exposure) 및 turnover의 차이가 크게 나타남.  
- long-only 제약을 추가하면 비제약 GMV 보다 높은 샤프비율을 기록하지만, 여전히 큰 거래비용이 발생하고 자산 수가 매우 집중되어있음.  
- 현실적인 제약을 추가해줌에 따라, 추정기 간 성과의 차이가 6.37%(비제약 GMV)에서 0.56%(거래비용 패널티를 포함한 가중치 제약 GMV)로 크게 줄어들었음.  
- 시간에 따른 변동성을 고려하는 것은 여전히 통계적으로 유의미하며, 동적 추정치가 지속적으로 가장 좋은 성과를 기록한다.  
→ 즉 현실적인 테스트 포트폴리오에서는 VCV 추정기의 선택이 크게 중요하지 않다는 것을 보여준다.  

따라서 본 논문에서는 실무에서 VCV 추정기를 평가할 때 GMV 포트폴리오에 가중치 제약 및 거래비용 패널티를 추가한 상태에서 비교해야 한다는 것을 주장한다.  


# 2. Designing test portfolios for evaluating variance-covariance estimators  
## 2.1. Global minimum-variance (GMV) portfolio  
$\min_{w_t} w_t^{\prime}\Sigma_t w_t \quad \text{s.t.} \quad \iota^{\prime}w_t=1,$   
→ closed form: $w_t^{*}=\frac{\Sigma_t^{-1}\iota}{\iota^{\prime}\Sigma_t^{-1}\iota}.$   

**GMV 포트폴리오가 왜 중요한가?**  
→ 입력 파라미터가 VCV만 존재하기 때문에 단순하다는 장점 뿐만 아니라 실제 테스트 포트폴리오에서도 종종 MVO보다 더 높은 샤프비율 및 낮은 변동성을 기록함.  

본 논문에서 사용하는 포트폴리오 환경 및 VCV 추정 방법은 다음과 같다.  
![image1.png](/assets/images/Beyond_GMV/image1.png)   


### 2.1.1. Long-only constraints   
Lon-only 제약은 두 가지 장점이 존재한다.  
1. 레버리지가 큰 포트폴리오가 되는 것을 막아준다.  
2. VCV 행렬 추정기에 암묵적인 축소 효과(implicit shrinkage)를 부여한다(Jagannathan and Ma 2003). 따라서 공분산 행렬의 추정오차가 포트폴리오에 미치는 부정적인 영향을 완화하는 데 도움이 된다.  

Zhao et al.(2023)은 VCV 행렬을 직접 축소하는 방법과 포트폴리오에 총익스포저 제약을 부과하는 방법을 비교한다.  
연구 결과, 일정 수준의 숏 포지션이 허용되는 한, 중간 정도의 총 익스포저 제약을 부과하더라도 VCV 행렬에 nonlinear shrinkage를 적용하는 것이 여전히 유용한 것(즉 총 익스포져 제약이 공분산 추정 오차를 크게 완화시키지 못함)으로 나타났다.  
→ 총 익스포져 제약의 자유도는 하나인 반면, NLS는 $N_t$ 개의 자유도를 가지기 때문.  


### 2.1.2. Transaction cost penalty andmaximum-weight constraints  
long-only 제약만으로는 자산 집중도 및 거래비용이 완전히 통제되지 못하기 때문에 더 실용적인 포트폴리오를 만들기 위해서 거래비용 패널티와 1% 최대 비중 제약을 함께 적용한 long-only GMV 포트폴리오를 고려한다.  

![image1.png](/assets/images/Beyond_GMV/image2.png)   
$c_{t,i}$: 거래 비용  
$τ_t^{fix}$: 자산이 유니버스에서 빠질 때 발생하는 고정 거래비용  

자산별 거래비용은 Briere et al., (2020) 의 모델을 사용하였으며, 본 논문에서는 위 포트폴리오를 GMVCON이라고 부른다.  

## 2.2. Risk parity portfolios  
GMV 포트폴리오는 미래 포트폴리오 분산을 최소화하도록 하여 포트폴리오를 다각화하려고 하지만 리스크 패리티 포트폴리오는 리스크 다각화 자체가 목적인 포트폴리오이다.  
리스크 패리티 포트폴리오는 다음의 두 가지 포트폴리오를 사용한다.  

### 2.2.1. Equal risk contribution(ERC) portfolio  
ERC 포트폴리오는 각 자산이 포트폴리오에 기여하는 위험이 모두 동일하도록 설정한다.  
이를 다음과 같은 최적화 식으로 나타낼 수 있다.  
![image1.png](/assets/images/Beyond_GMV/image3.png)   
위 식을 다음과 같은 GMV 문제로도 표현할 수 있다.  
![image1.png](/assets/images/Beyond_GMV/image4.png)   

Maillard et al., (2010) 에서는 ERC 포트폴리오, GMV 포트폴리오, EW 포트폴리오의 in-sample에서의 변동성이 다음과 같다는 것을 보여준다.  
$σ_{GMV,t}≤σ_{ERC,t}≤σ_{EW,t}$  
물론 사후 포트폴리오에서는 위 부등식이 항상 성립하지 않으며, Out-of-Sample 에서 비교를 직접 해봐야 한다.  

### 2.2.2. Hierarchical risk parity (HRP) portfolio  
López de Prado(2016) 는 금융 자산의 계층적 구조를 가정하여 각 자산마다 link를 줄임으로써 N-1개의 엣지를 가진 minimum spanning tree를 만드는 방식을 제안한다.  
HRP는 크게 3단계로 요약되는데,  
1. tree clustering  
2. quasi-diagonalization  
3. recursive bisection  

## 2.3. Benchmark portfolios  
이외에도 본 논문에서는 3가지의 간단한 벤치마크 포트폴리오를 사용한다.  
- 동일 가중 포트폴리오(EW)  
- 가치가중 포트폴리오(VW)  
- inverse-variance(IV)  

**inverse-variance?**  
$w_{i,t}^{\mathrm{IV}}=\frac{1/\sigma_{i,t}}{\sum_{i=1}^{N_t}1/\sigma_{i,t}}$  
→ 개별 자산의 과거 변동성의 역수를 활용하여 포트폴리오 가중치 계산. 분산이 클수록 포트폴리오 가중치가 낮아지는 구조.  


# 3. Estimating large VCV matrices  
본 논문에서는 다음의 VCV 추정기를 비교한다.  
1. Sample estimator  
2. Shrinkage estimators  
    - Linear shrinkage  
    - Nonliear shrinkage  
3. Dynamic estimators  
    - Dynamic conditional correlation model  
    - RiskMetrics  
4. Factor models(CAPM)    
    - exact factor model  
    - approximate factor model

DCC의 경우 Engle et al., (2019) 의 DCC-NLS 와 De Nard et al., (2021) 의 AFM-DCC-NLS 를 각각 사용한다.  

※ **DCC-NLS? AFM-DCC-NLS?**       
자산 공분산 행렬을 시간에 따라 동적으로 변화시키고 nonlinear shrinkage와 결합하는 것.  

- NLS  
먼저 본 논문에서는 NLS의 경우 Ledoit and Wolf(2022a) 의 Quadratic Inverse Shrinkage(QIS)를 사용한다.  
Sample Covariance를 고유값 분해하면 $S = UΛU^⊤$ 인데, 이때 NLS는 eigenvector U는 유지하고 eigenvalue Λ를 더 안정적인 값으로 조정하는 방식이다.  
$\hat Σ^{NLS}=U \hat Λ^{NLS} U^⊤$  

- DCC  
전체 공분산은 표준편차 대각행렬 D와 상관계수 행렬 R을 활용하여 다음과 같이 분해된다.  
$Σ_t = D_tR_tD_t$  

먼저 각 자산 i에 대해 GRACH 모델을 사용하여 분산을 추정한다.  
$\sigma_{i,t}^2 = \omega_i + \alpha_i r_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2$  
이를 통해 $D_t = diag(\sigma_{1,t},...,\sigma_{N,t})$ 를 만든다.  

다음으로 각 자산수익률을 $z_t = D_t^{-1}r_t$ 형태로 표준화한 후 상관관계를 다음과 같이 업데이트한다.  
$Q_t = (1 - a - b)\bar{Q} + a z_{t-1} z_{t-1}^\top + b Q_{t-1}$  
$Q_t$ = pseudo-correlation matrix  
$\bar{Q}$ = 장기 평균 correlation target  

→ 이후 정규화 과정을 거쳐 $R_t$ 생성 : $R_t = \text{diag}(Q_t)^{-1/2} Q_t \, \text{diag}(Q_t)^{-1/2}$  

최종적으로 $\Sigma_t^{\text{DCC-NLS}} = D_t R_t D_t$  → DCC-NLS  

DCC-NLS에서 NLS는 DCC 재귀식의 장기 상관관계 타깃 $\bar{Q}$ 를 추정하는 데 적용한다.  
- 기존 DCC: $z_t = D_t^{-1}r_t$ 에서 $S = \frac{1}{T-1}\sum^{T}_{t=1} (z_t - \bar{z})(z_t - \bar{z})^T$  
→ $\bar{Q} = \mathrm{diag}(S)^{-1/2} S \mathrm{diag}(S)^{-1/2}$  

기존 DCC는 $\bar{Q} = Corr(z_t)$ 로 쓸 때 자산 수 N이 커질 경우 불안정해지기 때문에 DCC-NLS에서는 $S = \frac{1}{T-1}\sum_{t=1}^{T} (z_t - \bar{z})(z_t - \bar{z})^\top$ 를 고유값 분해한다.  
$S_s= UΛU^⊤$  
본 논문에서 사용한 QIS는 inverse shrinkage를 강조하는데, GMV에서는 공분산의 역행렬이 들어가면서 오차가 크게 증폭되기 때문.  

concentration ratio: $c = \frac{N}{T-1}$ 으로 정의한 후, $\Lambda = diag(\lambda_1,...,\lambda_N)$ 에서 각 고유값의 역수를 $x_i = \lambda^{-1}_i$ 로 정의한다.  
(이후 NLS 방식으로 $\lambda$ 를 조정하는 과정은 추후에 리뷰로 자세히 다룰 예정)  


팩터모델의 경우 De Nard et al., (2021) 은 AFM-DCC-NLS 방법을 제안했는데, factor model-based covariance 구조를 팩터 부분과 잔차 부분으로 분류할 때 잔차 공분산에 동적 변화를 주는 것이다.  
$Σ_{u,t}=D_{u,t}R_{u,t}D_{u,t}$  
- $D_{u,t}$ : 잔차별 변동성  
- $R_{u,t}$ : 잔차별 상관관계  

따라서 적용 방식은 DCC-NLS와 같으며, 본 논문에서는 De Nard et al., (2021) 의 방식을 사용했기 때문에 잔차 공분산에 DCC-NLS를 적용했다(다른 논문에서는 팩터 공분산에 DCC-NLS를 적용하기도 함).  

- 팩터모델에서 exact factor model을 가정할 경우 DCC는 잔차에 적용이 불가능하다. DCC를 사용하는 순간 비대각 원소가 살아나기 때문.  
- 따라서 exact factor model(DFM)은 팩터 공분산에 DCC를 적용하는 경우만 존재한다. 잔차를 dynamic하게 하고 싶을 경우 GARCH로 잔차 분산을 동적으로 변화시켜줄 수 있음.  
- approximate factor model(AFM)은 팩터 공분산에도, 잔차에도 DCC를 적용할 수 있으며, 일반적으로는 단순 DCC를 적용하는 경우 or DCC-NLS를 적용하는 경우로 나뉜다.  







# 4. Empirical design  
## 4.1. Data   
데이터는 1990년 1월 1일 ~ 2021년 12월 31일까지의 NYSE, AMEX, NASDAQ 증권거래소에 상장된 주식 코드 10 or 11을 가진 미국 주식으로 구성한다.  

VCV 행렬은 5년 추정 기간을 가진 롤링윈도우 방식을 사용하며, 포트폴리오는 매달 말에 리밸런싱된다.  
in-sample 기간동안 데이터가 존재하는 종목을 대상으로 하며, 각각 500, 1000개 종목을 유니버스로 사용한다.  

포트폴리오의 실용성을 평가하기 위해 거래비용을 제외한 성과를 기록하였으며, Briere et al., (2020)의 모델을 사용하여 자산별 거래비용을 추정했다.  

아래는 추정된 거래비용의 분포를 나타낸다.  
![image1.png](/assets/images/Beyond_GMV/image5.png)   

## 4.2. Performance metrics  
포트폴리오의 성과 평가는 포트폴리오 변동성, 즉 OOS 기간의 포트폴리오 수익률 표준편차를 사용한다.  
통계적 유의성을 판별하기 위해 Ledoit and Wolf (2011) 의 pairwise variance test를 사용한다.  

- LS VS Sample  
- NLS VS LS  
- RM-NLS VS NLS  
- DCC-NLS VS RM-NLS  
- EFM VS Sample   
- AFM-DCC-NLS VS DCC-NLS  

사후 변동성 이외에도 수익률, 포트폴리오 집중도, 턴오버 등을 계산하며, 포트폴리오 집중도의 경우 3가지 지표를 사용한다.  
- 절대 비중이 0.1% 이상인 평균 포지션 수(POS)  
- 월별로 가장 큰 절댓값 상위 10개 포지션 합게 평균(MAXW)  
- 월별 유효 포트폴리오 비중(WEFF)  

![image1.png](/assets/images/Beyond_GMV/image6.png)   


# 5. Horse racing VCV estimators in risk-based portfolios  
## 5.1. Evaluating VCV matrix estimators by ex post portfolio volatility  
![image1.png](/assets/images/Beyond_GMV/image7.png)   
표 3은 벤치마크 포트폴리오, 리스크 기반 포트폴리오, 그리고 표 1에 자세히 나와 있는 선택된 VCV 행렬 추정치들의 사후 변동성을 나타낸다.  

### 5.1.1. How does the choice of shrinkage method affect performance?  
- 제약 없는 GMV에서 표본 추정치 기준으로 10.91%의 변동성을 기록하며 기본 벤치마크 포트폴리오를 능가한다.  
→ 숏포지션이 가능하기 때문에 체계적 요인에 대한 노출을 줄일 수 있음  
- 롱온리 제약에서는 전체 포트폴리오의 변동성이 증가하지만, 여전히 샘플 추정치 기반 GMV는 벤치마크 포트폴리오보다 낮은 변동성을 보인다.  
- Sample → shrinkage로 가면 제약 없는 GMV의 사후 변동성을 줄일 수 있으며, 이때 LS보다 NLS의 성능이 더 좋다.  
- 롱온리 제약을 적용할 경우 Sample과 LS 및 NLS의 차이는 거의 사라지며, 추가 패널티 및 제약을 적용한 GMV CON 환경에서도 성과 차이를 보이지 못한다.  


### 5.1.2. Do dynamic estimators outperform static estimators?  
- 단순 RM 추정기는 제약 없는 GMV 포트폴리오에서 가장 성능이 좋지 않다.  
- 반면 RM-NLS는 성과가 좋으며, DCC-NLS의 경우에도 제약 없는 GMV에서 가장 우수한 성과를 기록한다.  
→ 제약 없는 GMV 포트폴리오에서는 shrinkage와 dynamic estimator가 모두 필요함  
- 롱온리 환경에서도 동적 추정기가 전반적으로 가장 우수한 성과를 기록하며, RM과 RM-NLS의 성과가 비슷해지는 것을 통해 shrinkage는 롱온리 환경에서 이점을 주지 못한다는 것을 알 수 있음.  
- DCC-NLS는 GMV CON 을 제외한 모든 포트폴리오에서 비교 VCV 대상 중 가장 우수한 성능을 보임.  


### 5.1.3. Do minimum-variance problems benefit from a factor structure?  
결과에 따르면 거의 대부분에서 EFM이 가장 저조한 성과를 기록함.  
→ 추가적인 편향이 추정 오류 감소의 이점을 능가한다는 것을 시사  

AFM-DCC-NLS의 경우 좋은 성과를 기록하긴 하지만 이 또한 DCC-NLS를 능가하지 못함.  

※ 선행연구에서 FF5 factor model과 CAPM을 대상으로 AFM-DCC-NLS를 비교한 결과 CAPM이 오히려 더 좋은 성과를 기록했다고 주장함. 본 연구에서는 observable factor model를 사용하였으나 일반적으로 latent factor model을 사용하는 경우가 많기 때문에 latent factor model로도 비교해봤으면 어땠을까 하는 생각...  


### 5.1.4. Subperiod analysis  
표 3의 패널 C와 D는 고변동성 및 저변동성 구간에서 각각의 포트폴리오 성과가 어떻게 다른지를 보인다.  

VIX 지수가 5년 이동평균을 상회하면 해당 기간을 고변동성 구간으로 설정하며, 하회하면 저변동성 구간으로 설정한다.  
→ 46%의 저변동성 국면과 54%의 고변동성 국면으로 분리  

결론적으로 전체 기간을 두 국면으로 분리해도 RM, RM-NLS, DCC-NLS와 같이 dynamic estimators가 좋은 성과를 기록한다.  


## 5.2. Risk-based portfolios in practice  
이 절에서는 포트폴리오의 실용성을 평가하기 위해 turnover가 성과에 미치는 영향을 확인하고 포트폴리오의 가중치 분포를 분석하여 포트폴리오 집중도 및 분산 정도를 평가한다.  
![image1.png](/assets/images/Beyond_GMV/image8.png)   

### 5.2.1. Unconstrained global minimum-variance portfolios  
표 4를 통해 제약 없는 GMV 포트폴리오가 일반적으로 가장 낮은 변동성을 기록하지만 그만큼 매우 높은 turnover 및 거래비용, 높은 레버리지를 보인다.  

팩터모델을 사용하면 이러한 턴오버 및 거래비용이 현저히 줄어들긴 하지만 여전히 롱온리 제약을 적용한 GMV보다는 높은 값을 기록한다.  

특히 DCC-NLS는 제약 없는 GMV 포트폴리오에서 가장 높은 샤프비율(거래비용을 반영한 후도)을 기록하며, 이는 전체 평가 대상 중 가장 높은 샤프비율이다.  

그럼에도 불구하고 제약 없는 GMV 포트폴리오는 상위 10개 종목의 비중이 크기 때문에 현실성이 떨어진다.  

### 5.2.2. Long-only GMV portfolios  
롱온리 제약을 적용할 경우 턴오버가 많이 줄어들었으며, dynamic estimator가 전반적으로 포트폴리오 턴오버를 키운다는 것을 확인할 수 있다.  

비제약 GMV와 비교했을 때, 포트폴리오의 턴오버는 많이 줄어들었지만 그만큼 변동성이 증가하는 trade-off가 생긴다.  

특히 롱온리 GMV도 실용성 측면에서 문제가 있는데, 유효 포트폴리오 비중(WEFF)이 30% 미만이라는 점이다. 따라서 본 논문에서는 이를 보완한 GMV CON에서의 비교를 제안한다.  

GMV CON에서의 결과를 보면, 제약으로 상한 비중(1% 이하)을 걸어놓은 만큼 GMV 포트폴리오의 유효 비중(WEFF)이 100개를 조금 넘는 정도에 불과하다.  

또한 거래비용 및 턴오버가 줄어들게 되는데, 여전히 dynamic estimator들은 높은 턴오버를 기록한다.  



### 5.2.3. Beyond GMV portfolios  
(생략)  


### 5.2.4. Risk-based portfolio selection and factor investing  
표 4의 결과에서는 어떤 VCV 추정기를 선택하든, 롱온리 GMV는 단순 VW 포트폴리오보다 높은 샤프비율을 달성했다.   
이러한 결과가 나타난 이유를 설명하기 위해, 본 논문에서는 각 포트폴리오 수익률에 대해 팩터 스타일분석을 실시하여 주요 체계적 요인 노출을 살펴보았다.  

![image1.png](/assets/images/Beyond_GMV/image9.png)   
여기서는 5년 추정기간을 사용한 DCC-NLS 추정기를 바탕으로 구성한 포트폴리오의 수익률에 초점을 맞춘다.  

- 모든 GMV 포트폴리오에서 시장 베타가 매우 낮으며, 그 값이 1보다 통계적으로 유의하게 작다.  
→ GMV 포트폴리오가 시장 노출도가 낮은 자산을 주로 선택한다는 Clarke et al.(2011)과 Scherer(2011)의 연구 결과와 일치  
- 또한 모든 GMV 포트폴리오는 저변동성 요인에 대해 약 0.10～0.24의 양(+)의 베타를 가지며, 통계적으로도 매우 유의하다.  
