# Practice - number classifier using MNIST
![image](https://user-images.githubusercontent.com/52550295/188263336-17c39c28-71d2-44de-b765-1dd92290341d.png)
    
    
_출처 : https://commons.wikimedia.org/wiki/File:MnistExamples.png_


### MNIST 데이터셋을 학습시켜 0에서 9까지의 숫자를 판별할 수 있는 모델을 만들어보자.
![image](https://user-images.githubusercontent.com/52550295/188263428-b4465a9a-8873-49ec-9229-fd3a02f34591.png)
    
    
---
    
    
    
# Project - Rock vs Scissor vs Paper Classifier
![image](https://user-images.githubusercontent.com/52550295/188263367-14392066-f29d-495f-aa14-25f299594fa4.png)

### 1) (흑백의 데이터에서 확장하여) 가위, 바위, 보 이미지를 학습시킨 모델을 통해 이를 분류할 수 있는 모델을 만들어보자

### 2) 해당 모델을 웹 상에서도 돌아갈 수 있는 서비스를 구현해보자

#### Backbone Network : LeNet

# 회고

32x32 사이즈에서 프로젝트를 수행했으나 다양한 시도에 비해 뚜렷한 성능 향상 결과를 이뤄내지 못함 - 최종 정확도 : 0.5046
(https://github.com/wonderkyeom/AIFFEL/blob/master/Exploration1/Exploration1%20Project%20FINAL(32x32).ipynb)

(32, 32, 3) shape에서 진행했던 이유 : VGG16 input 값의 최소가 32!

이후 112x112 사이즈에서 시도하여 다음과 같은 회고를 남김. - 최종 정확도 : 0.6966



### trainset(1차 2100개 + 2차 2500개)
1차 : 캐글에서 받은 가위바위보 데이터 셋(https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
![image.png](attachment:image.png)


2차 : tf-dataset의 rock_paper_scissors 데이터 셋
![image-5.png](attachment:image-5.png)
![image-4.png](attachment:image-4.png)

- 사진에서 보이는 것 처럼 1차 학습데이터는 초록색 단색의 바탕에 데이터의 분포가 넓지 않고, 2차 학습데이터 역시 크게 다르지는 않음(Data Augmentation 수행 예정)


### testset(4000개)
열 명 정도가 실내에서 촬영한 가위바위보 사진으로 1차 데이터셋보다는 상대적으로 데이터의 분포가 크고, 배경이 다양함
![image-2.png](attachment:image-2.png)
![image-3.png](attachment:image-3.png)

- 실생활에서 찍은 가위바위보 사진으로 배경이 다양하고, 데이터의 분포가 trainset보다 상대적으로 넓은 것을 확인할 수 있음

| 모델명 | trainset < testset | tf-dataset(rock_paper_scissor) 추가학습 | data augmentation | 비고
|---|---|---|---|---|
| LeNet | 0.3027 | 0.3300 | 0.4506 | |
| LeNet(특징을 잡는 파라미터 수↑) | 0.3076 | 0.3872 | 0.4747 | |
| VGG16 | 0.3328 | 0.3328 | 0.3328 | 학습이 전혀 안되고 있는거 같다.. |
| VGG16(pre-trained) | 0.4966 | 0.6369 | 0.6966 | 대략 70%의 성능 |

모델 변경으로 획기적인 성능향상을 기대하기는 어려웠음. (아무래도 SOTA가 아닌, 오버피팅 발생 가능성의 이슈가 있는 VGG여서 그럴 수 있다고 생각)
적은 양의 trainset으로는 올바른 학습을 기대하기 어려웠고, Data augmentation(회전, 색 반전 등)을 수행했을 때, 비로소 오버피팅에서 조금 벗어난 듯한 학습 그래프를 보였다. (나름 의미있는 솔루션이었다고 생각)
그럼에도 pre-trained 된 모델만큼 뛰어난 성능을 보여주는 가정은 없었다.

★★ 데이터 분포를 유사하게 맞춰주면 어떨까? 하는 생각은 해보았음.
예를 들어서, 현재 trainset의 배경이 전부 단색이므로, testset을 모델에 넣기 전에 배경부분과 손 부분을 분리할 수 있는 전처리를 수행해주고 모델에 넣어준다면 더 높은 정확도를 기대해볼 수 있지 않을까?
    
ex) np.where() 등으로 손 영역만 캐치해서 보여준다던지 등등

결론 : pre-trained된 모델이 존재함에 감사하자!
