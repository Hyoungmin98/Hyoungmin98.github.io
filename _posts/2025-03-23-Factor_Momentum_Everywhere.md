---
layout: post
title:  "Review: Factor Momentum Everywhere"
date:   2025-05-03 22:38:28 +0900
math: true
categories: Review
tags: Quant ML 
---

해당 게시글은 Gupta and Kelly(2018)의 "Factor Momentum Everywhere" 논문을 리뷰한 글입니다.   

# Introduction  
가격 모멘텀은 최근 다른 자산에 비해 높은 수익률을 누렸던 자산이 미래에도 높은 수익률을 보일 가능성이 더 높은 현상을 말한다. 일반적으로 개별 주식 간 횡단면 거래 전략으로 구현된다.  
본 논문에서는 기존 가격 모멘텀의 성과를 뛰어넘는 "팩터 모멘텀(Factor Mementum)"을 소개한다.  
팩터 모멘텀 전략은 주식 모멘텀, 산업 모멘텀 뿐만 아니라 가치 팩터 등의 다른 팩터를 능가하는 성과를 보여준다. 하지만 주식 모멘텀을 대체하지는 않으며 실제로 팩터 모멘텀은 주식 모멘텀을 통제한 후에도 통계적으로 유의한 알파 값을 가지기 때문에 주식 모멘텀과 팩터 모멘텀, 가치 팩터를 적절하게 결합하는 것이 포트폴리오 구성에 있어서 중요하다.  

- 본 논문에서는 65개의 팩터를 수집하여 팩터 모멘텀을 분석한 결과 월 평균 AR(1)계수는 0.11이었으며 65개 팩터 중 59개가 양의 값을 가졌고, 그중 49개 팩터가 유의한 양의 값을 가졌다.  
- 한달 전의 개별 팩터 수익률을 기반으로 팩터 모멘텀을 적용한 포트폴리오는 연간 샤프비율 0.84, 정보비율(Information Ratio) 0.33을 기록했다. 이러한 포트폴리오를 TSFM(Time Series Factor Momentum)이라고 부른다. raw factor를 통제한 후 계산한 알파 또한 65개중 61개 팩터에서 양의 값을 보이며 그중 47개 팩터에서 유의미한 양의 값을 가진다.    
- 1개월 팩터 모멘텀 전략의 성과가 가장 강력했지만 12개월 모멘텀을 적용해도 긍정적인 성과를 보인다.  


※ 팩터 수익률 구성 방법   
1. 매 기간마다 팩터값 상하위 1%에 해당하는 극단값은 제거.  
2. NYSE 시가총액 중앙값(국제 샘플은 80백분위)을 기준으로 Small/Large 그룹으로 분류.  
3. 팩터 값 기준으로 30/40/30 으로 나누어 Low, Medium, High 로 분류. 이때 각 팩터마다 수익률이 높을 것으로 추정되는 방향이 다르기에 원 논문에서 수익률이 더 높게 나온 방향을 High로 포지션 설정.  
4. 이렇게 나온 2*3 = 6개의 포트폴리오에 대해 0.5 × (Large High + Small High) – 0.5 × (Large Low + Small Low) 로 해당 기간의 롱-숏 팩터 수익률을 구한다.  



## Factors at a Glance  
![image1.png](/assets/images/factor_momentum/image1.png)    
Exhibit 1 은 총 65개의 팩터의 기대수익률, 샤프비율, FF5 alpha 값을 나타낸 지표이다. 65개 팩터중 25개의 팩터가 0과 다름이 유의하지 않게 나타났고 3개는 음의 수익률을 보였다. 가장 유의미하게 나타난 팩터는 beta, momentum 등의 팩터였다.  

또한 이 65개의 팩터들은 대부분 상관계수가 0.25이하로 서로 독립적이라는 것을 알 수 있다. 실제로 PCA를 진행한 결과 전체 팩터 수익률 공분산구조의 90%를 설명하려면 19개의 주성분이 필요했다.  



#  Factor Momentum  
## Factor Persistence  
65개의 팩터에 대해 AR(1)의 값을 분석한 결과 59개가 양의 월별 AR(1) 값을, 그중 49개가 통계적으로 유의함을 보였다(평균 AR(1) 계수값 0.11).  
→ 비교를 위해 Moskowitz,Ooi,and Pedersen(2012)의 논문에서 주장한 바에 따르면 초과시장 수익률의 월간 AR(1) 계수가 0.07인데 이 값은 시계열 모멘텀 전략을 적용하기에 충분하다고 제시한 수준이다.따라서 이는 과거의 수익률을 기반으로 팩터를 추정하는 것이 가능하다는 것을 나타냄. 즉 팩터 모멘텀 전략을 사용할 수 있는 근거가 충분하다는 것을 시사한다.  


## Time Series Factor Momentum  
![image1.png](/assets/images/factor_momentum/image2.png)   

$f^{TSFM}_{i,j,t+1} = s_{i,j,t} * f_{i,t+1}$  
$s_{i,j,t} = min(max(\frac{1}{\sigma_{i,j,t}}\sum^j_{\gamma =1}f_{i,t-\gamma+1}, -2),2)$   
스케일링 계수 s를 이용하여 팩터 i에 대해 TSFM 전략을 구성한다. 과거 수익률&변동성이 좋으면 매수, 나쁘면 매도하는 전략. 이때 너무 극단적인 포지션을 방지하기 위해 스케일링 계수 s의 범위는 [-2,2]로 설정한다.  

TSFM은 raw factor를 독립변수로 하고 선형회귀를 통해 알파를 평가할 수 있다(Exhibit 3). 연간 알파의 경우 65개 팩터 중 61개의 팩터가 양의 값을 보였으며 이중 47개 팩터가 통계적으로 유의했다. 각 팩터 모멘텀 전략의 샤프비율은 56개의 팩터가 0.2를 초과하였으며 이중 48개의 팩터가 통계적으로 유의했다.  

그 다음, 본 논문에서는 각 팩터의 TSFM전략을 결합하여 하나의 포트폴리오로 만든다. 
![image1.png](/assets/images/factor_momentum/image3.png)  
기간 t에서 양수 s를 가지는 팩터는 $TSFM_{j,t}^{Long}$으로, 0이하의 s를 가지는 팩터는 $TSFM_{j,t}^{Short}$으로 할당된다.

![image1.png](/assets/images/factor_momentum/image4.png)   
위 그래프를 통해 전반적으로 단기 모멘텀에서 좋은 성과가 나오는 것을 알 수 있다. 
C에서 나타난 equal-weighted portfolio 대비 알파값 또한 단기 기간에서는 수익률의 약 2%(12-10)를 설명하는 반면 기간이 길어질수록 설명력이 좋아지는 것을 볼 수 있다.  


## Cross Section Factor Momentum  
![image1.png](/assets/images/factor_momentum/image5.png)   
시계열 모멘텀이 아닌 횡단면 팩터 모멘텀 전략의 경우 해당 기간에서 동종업체를 능가하는 팩터 성과를 보이면 매수, 저조한 성과를 보이면 매도하는 방식으로 이루어진다. TSFM과의 차이점은 TSFM의 경우 과거 대비 좋은 성과를 보인 팩터라면 모두 매수하는 방식이지만 CSFM은 과거 대비 좋은 성과를 보임에도 불구하고 중앙값보다 저조한 팩터 성과라면 매도 포지션으로 들어간다는 점이다. 분석 결과 CSFM 또한 TSFM과 유사한 결과를 보이지만 TSFM 대비 약간 저조한 샤프비율과 알파값을 보인다.  


## Factor, Stock, and Industry Momentum  
![image1.png](/assets/images/factor_momentum/image6.png)   
모멘텀 전략의 누적 로그수익률을 비교한 결과 1개월 TSFM 전략이 가장 우수한 성과를 보였다. UMD(주가 모멘텀 전략)은 2009년 3~5월에서 급격한 하락이 존재했으나 팩터 모멘텀 전략은 이를 모두 완화시켰다.  

![image1.png](/assets/images/factor_momentum/image7.png)   
위 표는 CSFM, TSFM 각각 UMD(주식 모멘텀), STR(Reversal 전략), INDMOM(산업 모멘텀 전략)과의 상관관계를 나타낸 표이다.  
- 단기 모멘텀의 경우 CSFM, TSFM 모두 STR전략과 강한 음의 상관관계를 띄고 있으며 기간이 길어질수록 상관관계가 점점 약해지는 것을 알 수 있다. 일반적으로 단기간에서의 모멘텀 효과(즉 Reversal 효과)를 반영한다면 STR전략과 양의 상관관계를 보여야 하지만 음의 상관관계가 나온것을 통해 단기간(1개월)에서 Mementum 효과보단 Reversal 효과를 띈다는 기존 연구결과와 상반되는 결과가 나타났다는 것을 알 수 있다.     
- CSFM과 TSFM은 모든 룩백 기간에서 매우 높은 상관관계를 보인다.  
- UMD, INDMOM의 경우 1-12, 2-12 룩백 기간에서 TSFM, CSFM과 높은 양의 상관관계를 보인다. 이는 6~12개월 사이에서 Momentum 효과가 강하게 띈다는 기존 연구결과를 뒷받침한다고 할 수 있다.  

![image1.png](/assets/images/factor_momentum/image8.png)   
위 그림은 TSFM과 CSFM에 대해 기대수익률과 UMD, INDMOM, STR 전략 대비 알파값을 비교한 그림이다.   
- 패널 A에서 TSFM은 모든 기간에 대해 기대수익률을 가지며 TSFM의 2-12 윈도우의 경우만 UMD로 어느정도 설명이 가능하다.  
- 패널 B에서 CSFM은 모든 기간에 대해 양의 기대수익률을 가지며 UMD, INDMOM이 CSFM의 수익률을 설명할 수 있음을 알 수 있다. 이는 기존의 전통 모멘텀과 유사한 전략을 가진다는 것을 의미한다.    
- STR 대비 알파값은 TSFM과 CSFM 모두 큰 값을 가진다. 이는 TSFM과 CSFM 모두 STR 전략으로 설명되지 않는다는 것을 의미한다. 즉, STR은 이 두 전략과 전혀 다른 방향성을 가진 전략이라는 것을 알 수 있다.  

![image1.png](/assets/images/factor_momentum/image9.png)   
반대로 UMD, INDMOM, STR에 대해 TSFM, CSFM을 통제하여 알파값을 비교한 결과이다.  
- 1-12, 1-36, 1-60 기간동안 UMD, INDMOM의 TSFM 대비 알파값이 낮기 때문에 TSFM이 UMD, INDMOM에 대한 수익률을 상당부분 설명한다. → 1-12 TSFM의 알파값이 0에 가깝고 유의하지 않음을 통해 UMD를 TSFM이 능가한다는 것을 알 수 있다.  
- CSFM은 UMD 수익률을 잘 설명하지는 못하지만 INDMOM의 수익률은 상당 부분 설명이 가능하다. 
- Exhibit 8에서와 마찬가지로 TSFM과 CSFM모두 STR전략의 수익률을 설명하지는 못한다. → STR(단기 Reversal 전략)은 팩터 모멘텀 전략과 독립적인 전략  
- 특히 TSFM, CSFM을 통제한 STR의 알파값이 5%를 초과하여 연평균 수익률 3.4%보다 더 커졌는데 이는 팩터 모멘텀 전략이 첫달을 포함하였음에도 불구하고 Reversal 효과에 더 방해(연구결과와 다르게 음의 상관관계를 보임)가 되었다는 것을 보여준다.  


## Portfolio Combinations  
다음으로는 Mean-Variance Optimization을 통해 팩터 포트폴리오를 구성하고 샤프비율과 각 팩터 가중치를 분석한다.  
![image1.png](/assets/images/factor_momentum/image10.png)   
- 개별 팩터 포트폴리오의 경우 0.84의 샤프비율을 가진 TSFM 1-1 포트폴리오의 성과가 가장 우수했으며 멀티팩터 포트폴리오의 경우 2.62의 샤프비율 값을 가진 6번 포트폴리오(CSFM, TSFM 1-12 제외)가 가장 우수한 성과를 보였다.  
- CSFM 1-12의 경우 3번 포트폴리오에서 -5.38의 가중치를 가진 것을 통해 성과가 좋지 않아 공매도 포지션으로 들어간 것을 알 수 있다.  

![image1.png](/assets/images/factor_momentum/image11.png)   
추가적으로 단순 Fama-French의 HML팩터가 아닌 HML-Devil(HML에 시장 및 산업 Neutralizing까지 수행한 팩터)를 활용했을 때 팩터 모멘텀 포트폴리오의 효과를 높여준다는 것을 Exhibit 12에서 보여준다.  


##  Implementability  
모멘텀 전략은 본질적으로 높은 Turnover를 가지기 때문에 거래비용은 포트폴리오에서 팩터 모멘텀의 실질적 유용성을 평가하는 주요 지표이다.  

![image1.png](/assets/images/factor_momentum/image12.png)   
- Exhibit 13의 패널 A는 각 팩터의 Turnover를 나타낸다. 두 팩터 모멘텀 전략이 다른 팩터보다 훨씬 더 많은 Turnover를 가지고 있는 것을 볼 수 있다.  
- 패널 B는 거래비용을 제외한 팩터 포트폴리오의 연간 샤프비율을 나타낸 그래프이다(매출 1단위당 10bp(0.1%)의 비용으로 계산).   
- 패널 B를 통해 거래비용이 팩터 모멘텀의 성과를 많이 감소시키긴 하지만 여전히 다른 팩터들보다 더 좋은 성과를 기록하는 것을 알 수 있다.  
- STR의 경우 거래비용을 반영한 결과 포트폴리오의 성과가 완전히 지워짐. 


##  Factor Momentum Around the World  
이 섹션에서는 팩터 모멘텀 결과가 다른 국가에서도 적용되는지를 살펴본다.  
- 미국 외 다른 지역에서도 개별 팩터의 과거 수익률이 미래 수익률을 예측하는 현상을 보임.  
- 미국 외에서도 분석 대상으로 삼은 대부분의 국가에서 공통적으로 TSFM 및 CSFM이 효과적으로 작동함.  


# Conclusion  
본 논문은 주식 모멘텀을 넘어 팩터들의 모멘텀 현상을 분석한 논문으로 기존 모멘텀 현상보다 더 높은 수익률과 샤프비율을 지니는 '팩터 모멘텀'이라는 새로운 팩터를 만들었다고 할 수 있다.  
횡단면 팩터 모멘텀과 시계열 팩터 모멘텀 모두 긍정적인 결과가 나왔지만 시계열 팩터 모멘텀의 성과가 더 좋았으며 특히 다양한 룩백 윈도우 기간에서도 모두 좋은 성과를 기록했다.  
팩터 모멘텀 전략은 미국 이외에도 많은 국가에서 같은 현상을 보이며 이는 국제적으로 모두 통용되는 새로운 팩터라는 것을 시사한다.  

**본 논문에서는 한국 시장에서의 팩터 모멘텀 현상을 분석하진 않았다. 개인적으로 한국시장의 팩터 모멘텀 현상이 유사한 결과가 도출될지 궁금하다. 한국의 경우 오히려 역모멘텀 효과가 나타난다는 몇몇 연구결과도 존재하기 때문에 한국시장의 팩터 모멘텀 연구도 굉장히 흥미로운 주제가 될 것이라 생각한다.**    




