# U-net: 바이오메디컬 이미지 세그멘테이션을 위한 Convolutional Network

- End to End 구조로 빠르고, 단순
- data augmentation
- 이웃한 픽셀들 간의 관계 파악 잘 함 
- 의료 데이터 셋에서 state of the art 수준의 성능
  

### Patch: 이미지 인식 단위


## 기존 문제점 개선
### 빠른 속도
#### 문제
sliding window로 했던곳을 또 계산

#### 해결방안
patch 탐색 방식 이용

### context와 localization 간의 Trade-off 해결
#### 문제
큰 이미지를 인식하면 context 파악 용이.  
작은 이미지를 인식하면 세밀한 localization 가능. 그러나 context 인식률은 떨어짐.  

#### 해결방안
Contracting Path 에서는 이미지의 context 포착,  
Expanding Path에서는 feature map Upsampling 후, Contracting Path에서 포착한 context와 결합해 localization의 정확도를 높임  


### Network
- conv 3*3, ReLu
- copy and crop
- max pool 2*2
- up-conv 2*2
- conv 1*1

padding 없이 Convolution 연산으로 작아짐 
```
Input 572*572  
Output 388*388  
```
이를 해결하기 위해 mirroring padding 적용  

#### Contracting Path
context 파악,  
Downsampling으로 해상도가 떨어짐, feature map의 channel 수가 늘어남, 작고 두꺼워짐 
#### Expanding Path
localization,  
Upsampling으로 해상도를 높임, feature map의 크기가 늘어남, 넓고 앏아짐 

## [Result](result/README.md)

Reference :  
https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet