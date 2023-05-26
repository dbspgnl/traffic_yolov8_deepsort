# 교통량 측정을 위한 오브젝트 트래킹 
## 환경설정
- CUDA
- cnDNN
- TensorFlow

## 참조 코드
- ultralytics YOLOv8
- deep_sort_pytorch
- YOLOv8-DeepSORT-Object-Tracking

## deep_sort_pytorch 다운로드
```
google drive
https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing
git hub
https://github.com/ZQPei/deep_sort_pytorch
```

## traffic_best.pt 다운로드
```
https://universe.roboflow.com/drone-dataset-mvh8i/detection-bzujh
```

## YOLOv8-DeepSORT-Object-Tracking 관련 설치
```
!git clone https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.git
```
```
%cd YOLOv8-DeepSORT-Object-Tracking
!pip install -e .[dev]
```
이후 의존성 관련 에러 발생시
1. **[No module named easydict](https://stackoverflow.com/questions/43732185/importerror-no-module-named-easydict)** 
왜 의존성에 포함되지 않은지 모르겠다만… 따로 설치해준다.
```python
# 가상 환경에서
#sudo pip install easydict
pip install easydict
```
2. **[module 'numpy' has no attribute 'float'"](https://stackoverflow.com/questions/74844262/how-can-i-solve-error-module-numpy-has-no-attribute-float-in-python)** 
numpy 1.20 모듈이 더 이상 지원하지 않음. 따라서 1.24로 업그레이드
```python
# 가상 환경에서
pip install "numpy<1.24"
```

## 실행
경로 이동
```
%cd YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect
```
detect home 경로에서 mp4 영상 소스와 pt 학습 데이터가 있다고 가정
```python
python predict.py model=traffic_best.pt source="test3.mp4"
```
각종 옵션과 설명은 공식에서 확인  
**[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**  
**[ultralytics Quickstart Docs](https://docs.ultralytics.com/quickstart/)** 
**[ultralytics Quickstart Docs OPTION](https://docs.ultralytics.com/usage/cfg/#train)** 
```python
# 작업 과정 보기
python predict.py model=traffic_best.pt source="test3.mp4" show=True
# 프레임 절반으로 줄이기
python predict.py model=traffic_best.pt source="test3.mp4" vid_stride=2
```
