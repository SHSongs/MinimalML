## YOLO

[YOLO 원문](https://arxiv.org/abs/1506.02640)

물체 탐지에 대한 새로운 접근 방식

사람은 이미지를 보고 Ojebct의 종류, 위치, 관계를 빠르고 정확하게 알 수 있다.  
그러나 R-CNN 같은 detection system 들은 복잡한 파이프라인으로 인해 Human visual system을 따라하기에는 부족하다. (느린 속도, 각각 최적화 해야 됨)  

YOLO는 이미지 내의 bounding box와 class probability를 
디텍션 파이프라인이 싱글 네트워크이다. 하나의 회귀 문제로 간주해 object의 종류와 위치를 추측한다.   



#### 기존의 object detection method와 비교시 YOLO의 장단점  

#### 장점 
- 매우 빠르다 
- YOLO모델은 45 프레임
- Fast YOLO 는 155 프레임 이상

- Object에 대한 일반화된 특징을 학습한다. natural image로 학습 후 artwork에서 태스트했을때 다른 모델보다 더 높은 성능을 보여준다.

#### 단점 
작은 오브젝트, 새때같이 그룹 지어 나타낸 오브젝트에 대한 낮은 정확도



### Yolo vs Fast R-CNN

Yolo가 더 적은 Background error   
Fast R-CNN이 정확도가 더 높다.   

### YOLO와 Fast R-CNN결합

다른 Fast-RCNN 모델과 결합하였을 때 작은 이점이 있다.  
YOLO와 Fast R-CNN이 결합하였을 때 mAP가 3.2 상승한다.  

## Unified Detection

1. Input Image를 S * S grid로 나눈다
2. 각 grid cell은 B개의 bounding box(x,y,w,h)와 각 bounding box에 대한 confidence score(c)를 갖는다. 
3. 각 grid cell은 C개의 conditional class probability를 갖는다.
4. 각 bounding box는 x, y, w, h, confidence로 구성된다.
   (x, y): Bounding box의 중심점
   (w, h): 전체 이미지의 width, height에 대한 상대값

### Network Design

YOLO의 Network 설계는 GoogleLeNet classification을 기반으로 한다.

논문에서는 7 * 7 grid    
B(bounding box) = 2  
class = 20 개를 사용해  

S * S * (B * 5 + C)  
7 * 7 * 30 의 tensor가 output이다  

98개의 class specific confidence score를 얻을 수 있다

### Loss
기호  
(1) Object가 존재하는 grid cell i의 predictor bounding box j  
(2) Object가 존재하지 않은 grid cell i의 bounding box j  
(3) Object가 존재하는 grid cell i  


1. (1)의, x와 y의 loss
2. (1)의, w와 h의 loss 제곱근을 이용해 큰 box의 error를 작게 한다
3. (1)의, confidence score의 loss. (Ci = 1)
4. (2)의, confidence score의 loss. (Ci = 0)
5. (3)의, conditional class probability의 loss. (Correct class c:pi(c)=1, otherwise:pi(c)=0)   


## Limitation of YOLO


### Reference
[curt-park yolo](https://curt-park.github.io/2017-03-26/yolo/)