# 예측 프로세스

<p align="center"><img width="800" src="https://raw.githubusercontent.com/PaddlePaddle/Paddle-Inference-Demo/master/docs/images/workflow.png"/></p>

```plaintext
Paddle (파들)  --->  Paddle 추론 모델
                  ( __model__
                    conv1_bn_mean
                    conv1_bn_offset
                    conv1_bn_scale )

TensorFlow / Caffe / ONNX  ---> 모델 변환 도구 (X2Paddle) ---> Paddle 추론 모델

1️⃣ 모델 양자화 / 가지치기 (PaddleSlim)
   ↘︎
    Paddle 추론 모델
    ↓
2️⃣ 환경 준비
   (다운로드 / 설치 / 컴파일
   Paddle 추론 라이브러리)

3️⃣ 예측 프로그램 개발 / 컴파일

입력 데이터  --->  예측 실행  --->  출력 데이터
```
## 1. 모델 준비

Paddle Inference는 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 딥러닝 프레임워크로 훈련된 추론 모델을 네이티브로 지원합니다.  
신버전 PaddlePaddle에서는 추론 모델을 각각 `paddle.jit.save`(동적 그래프), `paddle.static.save_inference_model`(정적 그래프), 또는 `paddle.Model().save`(고수준 API)로 저장합니다.  
구버전 PaddlePaddle에서는 `fluid.io.save_inference_model` API로 추론 모델을 저장합니다.  
자세한 내용은 [여기](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html)를 참고하세요.

Caffe, TensorFlow, PyTorch 등 타 프레임워크에서 만든 모델이라면, [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) 도구를 사용해 PaddlePaddle 형식으로 변환할 수 있습니다.

## 2. 환경 준비

### 1) Python 환경

[Paddle 공식 홈페이지 - 빠른 설치](https://www.paddlepaddle.org.cn/install/quick) 페이지를 참고해 직접 설치 또는 컴파일하세요.  
현재 pip/conda 설치, 도커 이미지, 소스 컴파일 등 다양한 방법으로 Paddle Inference 개발 환경을 준비할 수 있습니다.

### 2) C++ 환경

Paddle Inference는 Ubuntu/Windows/MacOS 플랫폼에 대해 공식 Release 예측 라이브러리를 제공합니다.  
위 플랫폼 중 하나를 사용 중이라면 아래 링크에서 직접 다운로드를 권장하며, 필요 시 [소스 컴파일](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) 방법도 참고하세요.

- [Linux 예측 라이브러리 다운로드 및 설치](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux)  
- [Windows 예측 라이브러리 다운로드 및 설치](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)

## 3. 예측 프로그램 개발

Paddle Inference는 Predictor를 사용해 예측을 수행합니다.  
Predictor는 계산 그래프를 분석해 OP 융합, 메모리 최적화, MKLDNN 및 TensorRT 같은 하위 가속 라이브러리 지원 등 일련의 최적화를 적용해 예측 성능을 크게 향상시키는 고성능 예측 엔진입니다.

<p align="center"><img width="800" src="https://raw.githubusercontent.com/PaddlePaddle/Paddle-Inference-Demo/master/docs/images/predict.png"/></p>

```
1. 예측 설정 관리자 
   `paddle_infer::Config`  
   - 모델 경로 설정  
   - 실행 장치 지정  
   - 최적화 옵션 설정  

2. 예측 엔진 생성  
   `paddle_infer::Predictor`  
   - 예측 모델 계산 그래프 분석 및 최적화:  
     - OP 융합  
     - 메모리 / 캐시 최적화  
     - MKLDNN, TensorRT 등의 가속 라이브러리 지원  

3. 입력 데이터  
   `paddle_infer::Tensor`  

4. 예측 실행
   `predictor->Run()`  

5. 예측 결과  
   `paddle_infer::Tensor`
```

예측 프로그램 개발은 간단한 5단계로 이루어집니다 (여기서는 C++ API 기준):

1. 추론 옵션 설정 (`paddle_infer::Config`)  
   - 모델 경로, 실행 디바이스, 계산 그래프 최적화 활성화 여부, MKLDNN/TensorRT 가속 사용 등 설정
2. 추론 엔진 생성 (`paddle_infer::Predictor`)  
   - `CreatePredictor(Config)` 호출로 엔진 초기화
3. 입력 데이터 준비  
   - `auto input_names = predictor->GetInputNames()`로 모든 입력 Tensor 이름 획득  
   - `auto tensor = predictor->GetInputTensor(input_names[i])`로 입력 Tensor 포인터 획득  
   - `tensor->copy_from_cpu(data)`로 데이터 복사
4. 예측 실행  
   - `predictor->Run()` 호출
5. 예측 결과 획득  
   - `auto out_names = predictor->GetOutputNames()`로 출력 Tensor 이름 획득  
   - `auto tensor = predictor->GetOutputTensor(out_names[i])`로 출력 Tensor 포인터 획득  
   - `tensor->copy_to_cpu(data)`로 데이터 복사

Paddle Inference는 C++ 및 Python API 사용 예제와 개발 문서를 제공합니다.  
예제를 참고해 빠르게 사용법을 익히고, 프로젝트에 통합할 수 있습니다.

- [예측 예제 (C++)](./cpp_demo)  
- [예측 예제 (Python)](./python_demo)
