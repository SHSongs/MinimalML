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



## Conclusion
강화학습을 위한 새로운 딥러닝 모델을 소개한다. 그리고 Atari 2600에서 어려운 정책 컨트롤 마스터 능력을 입증했다. 오직 raw pixels을 input으로.   
online Q-learning의 변종을 제시한다; 확률적 미니배치 업데이트 RL에서 deep network training을 용이하게하는 경험 리플레이 메모리.  
이 접근법은 테스트된 7개의 게임중 6개에서 state-of-the-art 결과를 준다; architecture or hyperparameters 조정 없이.