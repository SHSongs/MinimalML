# DQN 
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)  

## 요약 

고차원의 센서 인풋으로부터 정책을 배우는 것에 성공한 첫 번째 딥러닝 모델이다.  
convolutional neural network를 사용했고 Q-learning의 변종으로 학습했다.  
input은 원시 픽셀 output은 미래 보상을 추정하는 가치함수.  
Atari 2600 게임에서 architectures나 learning algorithms 조정 없이 사용하였다.  
 여섯 개의 게임에서 이전의 모든 접근법보다 뛰어나다. 그리고 그중 3개는 인간의 실력을 뛰어넘는다.  


## 소개

비전이나 말 같은 고차원의 인풋으로부터 agents를 직접 컨트롤하는 것을 배우는 것은 RL의 긴 난제였다.  
많은 성공적인 RL applications은 이러한 것들을 선형 함수나 정책 표현과 결합한 수작업으로 조작된 기능에 의존했다.  
분명히, 이러한 시스템의 성능은 feature representation에 의존한다.  
최근 딥러닝의 발전은 원시 센서 데이터로부터 높은 수준의 특징을 추출하는것을 가능하게 만들었다.  
컴퓨터 비전과 음성인식의 돌파구이다.  
이 방법들은 convolutional networks, 다층 퍼셉트론, 볼츠만 머신과 반복 neural networks을 포함하여 다양한 neural network architectures 와 지도학습, 비지도학습을 모두 활용했다.  
이런 유사한 기술이 감각 데이터를 가진 RL에 도움이 되는것은 당연하다.  

## Deep Reinforcement Learning

210\*160 pixel image를 흑백과 다운 샘플링 110\*84 한뒤
플레이화면 84\*84 크롭 (정사각형 인풋이 필요한, 2d convolution GPU구현이기 때문에)  
마지막 4 프레임으로  
Q-function에 대한 입력을 생산하기 위해  

neural network 를 사용해 Q를 파라미터화 하는 방법  

상태를 입력받으면 각 행동에 대응되는 스칼라 값 여러개를 돌려준다.  
한번의 forward pass 로 주어진 state에서 가능한 모든 행동 계산 가능.  

### 네트워크 구조

```
84\*84\*4 크기의 input  

8*8 conv stride 4  16개  
relu  

4*4 conv stride 2 32개  
relu  

256 fully-connected layer  
relu  

가능한 각 액션  (4 ~ 16 사이)
```
convolutional networks로 학습된 접근법을 Deep Q-learning(DQN) 이라 한다.  


## 실험 
7개의 ATARI game에서 같은 architectures를 사용했다.  


모든 긍정 reward는 1로 부정 reward는 -1로  

frame-skipping을 사용했다;  
모든 프레임 대신에 k 번째 프레임 사용  
마지막 action은 스킵된 프레임에서 반복되었다.  

emulator를 진행시키는 것은 agent가 action을 선택하는것보다 훨씬 적은 계산한다.  

이 기술은 실행시간을 늘리지 않고, 더 많이 게임을 할 수 있게한다.  

모든 게임은 k = 4 를 이용하였다.  

예외) Space Invanders는 k = 4로 하면 레이저를 볼 수 없어 k = 3으로 했다.  

### Training and Stability
지도 학습은 훈련과 검증 set으로 쉽게 성능을 측정할 수 있다. 강화학습에서는 훈련중에 agent의 진행상황을 정확하게 측정하는것은 어려울 수 있다. 평가 지표로,  agent reward의 합을 수집한다, 게임 평균 episode나 게임,  주기적으로 training중에 성능을 계산한다. 
평균 total reward 지표는 매우 noisy한 경향이 있다. policy의 weight의 작은 변화가 큰 변화를 이끌수 있다; 
Figure 2의 왼쪽 두개의 표에서는 Seaquest와 Breakout game의 training 중에 어떻게 평균 reward가 발달하는지 보여준다. 두개의 평균 reward 표는 정말로 noise하다, 학습 알고리즘이 꾸준히 진전하지 못하다는 인상을 준다.  또 다른, 더 안정적인 지표는 estimated action-value functioni Q; agent가 받을 수 있는 할인된 reward 의 추정을 제공하는; 은 state으로부터 policy를 획득할수 있다. 
수집한다 state의 고정값; 랜덤 policy로 실행할때; 
training 시작과 끝의 maximum 예측 Q 값의 평균 은 TOTAL reward 보다 매우 부드럽게 증가한다 

### Visualizing the Value Funcion
그림 3은 Seaquest game에서 value funcion을 배우는것을 시각화를 보여준다. 이 그림은 적이 나타난 이후 (A). 어뢰를 적에게 발사하고 예측 값이 최고에 이르렀을때(B), 마지막으로 value가 대략 원래 값으로 떨어진다; 적이 사라진 이우에 (C)
그림 3은 입증하다 이 방법이 학습할 수 있다; 상당히 복잡한 sequence의 이벤트에서 value funcion 진화 

### Main Evaluation
DQN의 결과를 가장 좋은 성능의 방법들과 비교한다. 
Sarsa: Linear policies 각 다른 특징은 수작업으로 제작; Arari task 에서 . 
우린 최고 성능 feature set을 보고한다 .
Contingency는  Sarsa와 같은 기본적인 접근 방식을 사용한다.

## Conclusion
강화학습을 위한 새로운 딥러닝 모델을 소개한다. 그리고 Atari 2600에서 어려운 정책 컨트롤 마스터 능력을 입증했다. 오직 raw pixels을 input으로.   
online Q-learning의 변종을 제시한다; 확률적 미니배치 업데이트 RL에서 deep network training을 용이하게하는 경험 리플레이 메모리.  
이 접근법은 테스트된 7개의 게임중 6개에서 state-of-the-art 결과를 준다; architecture or hyperparameters 조정 없이.