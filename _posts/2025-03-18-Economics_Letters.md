---
layout: post
title:  "Review: Factor-Based Portfolio Optimization"
date:   2025-03-18 22:38:28 +0900
math: true
categories: Review
tags: Quant ML 
---

**해당 게시글은 Auh and Cho(2023)의 "Factor-Based Portfolio Optimization"을 리뷰한 글입니다. 논문에 대한 개인적인 의견도 포함되어 있습니다.**    
출처: https://www.sciencedirect.com/science/article/pii/S0165176523001623?casa_token=odPIWEKcSVYAAAAA:Yp_JNQ3-tK4lM5cLctz1Nnn7uATCxamemgHayTyFThr-TdGMbF1expbjvz-Fi3eV_LDPJPZd1ls


## Introduction    
이 논문은 CAPM(Capitl asset pricing model)을 통해 미래의 자산가격을 예측함으로써 MVO(Mean-Variance Optimization) 전략이 가지는 문제점을 완화한 연구이다. 주요 아이디어는 CAPM을 통한 미래 자산가격 예측과 머신러닝(SVR)을 활용함으로써 시장가격 예측 후 CAPM으로 자산가격을 예측하는 두가지 전략이다.  

포트폴리오 최적화는 평균-분산 최적화, 최소분산 최적화, 최대다각화 등과 같이 다양한 전략을 가지고 있지만 공통적으로 매개변수 추정 오류로 인한 성능저하의 문제점을 가지고 있다. 이에 따라 포트폴리오 최적화에 사용되는 기대수익률, 공분산 등을 과거 수익률을 기반으로 만들지만 Michaud(1989)는 이러한 접근법이 추정 잡음으로 인해 심각한 문제를 일으킨다고 주장한다.  

이에 본 논문에서는 두 단계에 걸쳐 포트폴리오 최적화에 활용되는 기대수익률을 추정한다.  
Step 1: 노이즈를 제거해주기 위해 CAPM 모델을 활용한다. 독립변수를 시장 베타, 종속변수를 다음시점에서의 자산 수익률로 설정한 후 회귀분석을 진행.  

Step 2: 머신러닝 모델 중 SVR(Support Vector Regression)을 활용한다. SVR을 통해 미래의 시장 수익률을 예측한다.  

포트폴리오 구성 종목은 다우존스 지수(DJI)를 사용하며, 공매도가 불가능하다는 조건에서 샤프비율을 최대화호도록 최적화를 진행한다.  

$$max\, SR = \frac{\mu'w}{(w'\sum w)^{\frac{1}{2}}}$$   

$$s.t. \, e'w = 1$$  
$$w \geq 0$$  

## Step 1  
이때 목적함수에 들어가는 $\mu$ 값을 추정하는 것이 핵심이다. CAPM을 통한 미래수익률 추정 과정은 다음과 같다.  
$$\mu^{factor}_{i,t+1} = \bar{r}^f_t + \hat{\beta}_i(\bar{r}^m_t - \bar{r}^f_t)$$  

베타값 추정은 영업일 기준 252일의 look-back window를 가진다.     

공분산 추정은 과거 공분산을 그대로 활용한다. 본 논문에서는 주요 초점을 기대수익률에 맞추었기 때문에 기존 방법대로 공분산을 추정한다.  

벤치마크 포트폴리오로는 GMVP(Global Minimum Variance Portfolio)와 MDRP(Maximum Diversification Ratio Portfolio)를 활용한다.  
GMVP는 최소 분산 포트폴리오로, 기대수익률 추정이 필요하지 않고 오로지 포트폴리오의 위험을 최소화시키는 데에 초점을 둔다.  
MDRP는 최대 다각화 포트폴리오로 자산 간 가중치를 최대한 다각화하는 데에 초점을 둔다. 이또한 기대수익률 추정이 필요하지 않는다.  

포트폴리오 성능 비교를 위해  in-sample에서의 샤프비율, 변동성, DR을 비교한다. 또한 HHI를 추가적으로 성능 비교에 활용한다. 이후 OOS(Out-of-sample)에서의 성과도 비교한다.  
포트폴리오의 매월 초의 일일 수익률을 평가하여 성능을 비교한다.  

## Step 2  
이번에는 머신러닝 방법론을 통해 시장 수익률을 먼저 예측한 후 CAPM을 통해 자산 수익률을 추정한다.  
이때 활용하는 머신러닝 방법론은 SVR(Support Vector Regression) 모델을 사용한다.  

SVR을 사용한 이유는 다음과 같다.  
1. 지도학습 알고리즘으로 입력-출력 쌍에 대해 작동하기 때문에 최상의 신호 조합을 찾는데 적절함.  
2. SVR은 비선형적인 관계를 포착할 수 있음.  

**※ SVR을 사용한 이유에 대한 개인적인 의견**    
**- Gu et al.(2020)에서는 Fama-French 3 factors와 이에 추가적으로 다양한 머신러닝 방법론을 통해 자산 수익률을 예측할 수 있는 적절한 모델을 비교했다. 이때 가장 유의미했던 모델이 Neural Network였는데(물론 여기서는 SVR을 사용하지 않았음.) 이 모델을 해당 연구에 적용하면 더 유의미한 예측력을 보일 수 있지 않을까하는 생각이 들었다. 다만 이상치에 대해 덜 민감하게 반응하는 SVR의 장점이 효과적으로 발휘했을 수도 있다고 생각한다.**    

이때 시장 모델을 사용하기 때문에 옵션 시장, 시장 심리 및 거시 경제 역학과 같은 광범위한 출처를 기반으로 총 주식 시장에 대한 22개의 예측 변수를 선택하여 SVR을 구성한다.  
→ **※ 이 부분이 본 논문의 핵심 포인트 중 하나라고 생각. 후에 결과에서도 나오지만 시장수익률을 예측한 후 CAPM까지 수행한 포트폴리오가 가장 좋은 성능을 기록함. 단순히 Step1만 진행하여 CAPM으로만 자산수익률을 예측한 모델은 단순 MVO와 큰 차이를 보이지 않았기 때문에 좋은 성능을 기록한 원인은 Step2에서 기반한다고 생각. 이때 Step2에서 비교적 복잡성이 덜한 SVR을 활용했음에도 성과가 잘나오는 이유가 적절한 독립변수 선택으로부터 파생되었다고 생각한다.**   

독립변수 목록: 1. BSEYD, 2. Ln(BSEYD), 3. D/P, 4. E/P, 5. CPI, 6. INDPRO, 7. UNRATE, 8. Term Spread, 9. $\pi^{MS}$, 10. Left-jump Intensity and Left-jump Volatility, 11. Option Slope, 12. CEFD, 13. NIPO, 14. RIPO, 15. PDND, 16. EQIS, 17. Investor Sentiment, 18. Investor Sentiment(Ort.), 19. CCI(Ort.), 20. CREDIT, 21. Momentum, 22. Moving Average of Momemtum   

본 논문에서는 포트폴리오를 총 5개를 구성한다.  
1. 과거 수익률 기반 최대샤프비율 포트폴리오(SRHist)  
2. Step1 과정만 진행한 최대샤프비율 포트폴리오(SRF1)  
3. Step1,2 모두 진행한 최대샤프비율 포트폴리오(SRML)   
4. 최소분산 포트폴리오(GMVP)  
5. 최대다각화 포트폴리오(MDRP)  

포트폴리오 자산군은 DJI 구성종목, CRSP Value-weighted 시장수익률을 CAPM의 팩터 수익률로 사용하여 진행. 표본 기간은 2010~2021년으로 설정. 포트폴리오는 매월 리밸런싱하며 한달동안 보유하고 다음달 리밸런싱되는 구조.  


## In-Sample Performance
![image1.png](/assets/images/Economics_letters/image1.png)  
In-Sample에서의 포트폴리오 성능 평가 결과 샤프비율은 과거 수익률을 매개변수로 넣은 SRHist 전략이 가장 높은 값을 보였다. 다만 변동성 또한 SRHist가 가장 높았으며 가장 낮은 Diversification ratio, 가장 높은 HHI(Herfindahl–HirschmanIndex)를 기록한 것을 보아 다른 포트폴리오 전략보다 우월하다고 생각하진 않는다(샤프비율은 높았지만 리스크도 그만큼 높아졌기 때문).   
HHI에서는 특히 좀 의외의 결과를 볼 수 있었다. 최대 다각화를 목적으로 하는 MDRP보다 SRF1, SRML 포트폴리오가 더 낮은 HHI 값을 보였는데 이는 두 전략의 자산들이 더 낮은 자산 집중도를 가진다는 것을 알 수 있다.



## OOS Performance
![image1.png](/assets/images/Economics_letters/image2.png)  

포트폴리오는 한달동안 보유한 상태로 매달 리밸런싱을 진행함으로써 일별 수익률을 기록한다.  
이를 1년의 롤링 윈도우 내에서 샤프비율과 MDD(Maximum drawdown)을 측정한다. 2010~2021년 중에서 처음 1년동안은 계산하지 않고 나머지 11년동안 측정한 것으로 예상된다.(대략 252*11 = 2772로 비슷한 값을 가짐.)  

- OOS 결과에 따르면 Step 1,2를 모두 거친 SRML 전략이 가장 우수한 성능을 기록하는 것을 볼 수 있다. 가장 높은 샤프비율 뿐만 아니라 가장 낮은 MMD 값을 보였다.  
- 이는 최적화 결과에서 발생하는 추정 잡음을 CAPM을 통해 효과적으로 완화할 수 있음을 나타낸다.  
- 실제로 in-sample에서의 가장 높은 샤프비율 값을 보인 SRHist 포트폴리오가 OOS 결과에서는 가장 낮은 샤프비율 값을 보이는 것을 통해 과거 수익률을 기반으로 하는 MVO 전략이 좋지 못하다는 것을 보여준다.  


![image1.png](/assets/images/Economics_letters/image3.png)  
위 그림을 통해 SRML 포트폴리오가 다우존스 지수 수익률보다 더 낮은 변동성과 높은 기대수익률을 보인다는 것을 알 수 있다.  

![image1.png](/assets/images/Economics_letters/image4.png)  
기간별 누적수익률 또한 SRML 포트폴리오가 가장 우수한 것을 보인다. SRF1 포트폴리오의 경우 SRHist 포트폴리오와 큰 차이를 보이지 않는 것을 보아 SVR으로 시장수익률을 예측하는 것이 주요하게 작용했다고 생각한다. 
또한 논문에서는 SRML 포트폴리오로 12년의 기간동안 약 7배의 투자 수익률을 보였다.  

이를 통해 본 논문에서는 평균-분산 최적화 전략을 적극적으로 활용하는 포트폴리오 매니저에게 새로운 접근 방식을 제공할 수 있음을 보여준다. 

※ 추가적으로 기대수익률을 추정하는 것 뿐만 아니라 공분산 행렬만 추정(OnlyCov), 기대수익률과 공분산 행렬 모두 추정(Both)하여 포트폴리오를 구성해보았으나 각각 0.874, 1.124의 샤프비율 값을 보였다.  


**이 논문을 읽으면서 개인적으로 놀라웠던 점은 이렇게 쉽고 간단한 모델으로 유의미한 성과를 보였다는 것이었다. 머신러닝&딥러닝 모델이 많이 발전하고 학부생들도 예측성과를 어떻게든 높이기 위해 다양한 딥러닝 모델을 사용하지만 정작 해석이 불가능하고 설득력이 떨어진다는 점이 공통적으로 발생하는 한계점이다. 그러나 이 논문은 학부생도 비교적 이해하기 쉽고 경영&경제학적으로 설명력이 강한 CAPM 모델과 SVR만을 활용하여 포트폴리오의 성과를 높이고 MVO의 문제점을 완화했다는 것이 이 논문의 특별함이라고 생각한다. 또한 SVR을 구성하는 과정에서 사용한 22개의 예측변수가 포트폴리오의 성과를 높이는 데에 큰 역할을 했을 것이라고 생각한다.** 
