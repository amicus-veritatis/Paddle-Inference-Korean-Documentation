# X86 CPU에서 양자화 모델 배포

## 1. 개요

모델 양자화는 모델 추론 성능을 효과적으로 향상시킬 수 있는 방법으로 잘 알려져 있으며, PaddlePaddle은 강력한 양자화 기능을 제공합니다.  
이 문서에서는 X86 CPU에서 PaddleSlim이 생성한 양자화 모델을 배포하는 방법을 소개합니다.

일반적인 이미지 분류 모델의 경우, Casecade Lake 아키텍처(예: Intel® Xeon® Gold 6271, 6248, X2XX 등)에서는  
INT8 양자화 모델이 FP32 모델보다 **3~3.7배** 빠른 추론 성능을 보이며,  
자연어 처리 모델의 경우 INT8 모델이 FP32 대비 **1.5~3배** 빠릅니다.  
SkyLake 아키텍처(예: Xeon Gold 6148, 8180 등)에서는 이미지 분류 모델의 INT8 추론 성능이 FP32 대비 **1.5배 수준**입니다.

X86 CPU에서 양자화 모델을 배포하는 절차는 다음과 같습니다:

- **양자화 모델 생성**: PaddleSlim을 사용해 양자화 모델 훈련 및 생성
- **모델 변환**: 생성된 양자화 모델을 최종 배포용 형식으로 변환
- **모델 배포**: Paddle Inference 추론 엔진으로 배포

## 2. Xeon(R) 6271에서 이미지 분류 INT8 모델의 정확도 및 성능

> **정확도**

|     모델     | FP32 Top1 | INT8 Top1 | Top1 차이 | FP32 Top5 | INT8 Top5 | Top5 차이 |
|:------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| MobileNet-V1 |  70.78%   |  70.74%   |  -0.04%   |  89.69%   |  89.43%   |  -0.26%   |
| MobileNet-V2 |  71.90%   |  72.21%   |  +0.31%   |  90.56%   |  90.62%   |  +0.06%   |
| ResNet101    |  77.50%   |  77.60%   |  +0.10%   |  93.58%   |  93.55%   |  -0.03%   |
| ResNet50     |  76.63%   |  76.50%   |  -0.13%   |  93.10%   |  92.98%   |  -0.12%   |
| VGG16        |  72.08%   |  71.74%   |  -0.34%   |  90.63%   |  89.71%   |  -0.92%   |
| VGG19        |  72.57%   |  72.12%   |  -0.45%   |  90.84%   |  90.15%   |  -0.69%   |

> **단일 스레드 성능 (images/sec)**

|     모델     | FP32 | INT8 | 속도 향상 배율 |
|:------------:|:----:|:----:|:--------------:|
| MobileNet-V1 | 74.05 | 216.36 | 2.92x |
| MobileNet-V2 | 88.60 | 205.84 | 2.32x |
| ResNet101    |  7.20 |  26.48 | 3.68x |
| ResNet50     | 13.23 |  50.02 | 3.78x |
| VGG16        |  3.47 |  10.67 | 3.07x |
| VGG19        |  2.83 |   9.09 | 3.21x |

## Xeon(R) 6271에서 자연어처리 INT8 모델 성능 및 정확도 (Ernie, GRU, LSTM)

> **성능**

| Ernie 레이턴시 | FP32 (ms) | INT8 (ms) | 향상 배율 |
|----------------|-----------|-----------|------------|
| 1 thread       | 237.21    | 79.26     | 2.99x      |
| 20 threads     | 22.08     | 12.57     | 1.76x      |

| GRU 성능 (QPS)              | Naive FP32 | INT8  | 향상 배율 |
|-----------------------------|------------|-------|------------|
| bs=1, thread=1              | 1108       | 1393  | 1.26x      |
| repeat=1, bs=50, thread=1   | 2175       | 3199  | 1.47x      |
| repeat=10, bs=50, thread=1  | 2165       | 3334  | 1.54x      |

| LSTM 성능 (QPS) | FP32    | INT8    | 향상 배율 |
|----------------|---------|---------|------------|
| 1 thread       | 4895.65 | 7190.55 | 1.47x      |
| 4 threads      | 6370.86 | 7942.51 | 1.25x      |

> **정확도**

| 모델   | FP32 정확도 | INT8 정확도 | 차이     |
|--------|-------------|-------------|----------|
| Ernie  | 80.20%      | 79.44%      | -0.76%   |

| LAC (GRU) | FP32    | INT8    | 차이      |
|-----------|---------|---------|-----------|
| 정확도    | 0.89326 | 0.89323 | -0.00007  |

| LSTM   | FP32  | INT8  |
|--------|-------|-------|
| HX_ACC | 0.933 | 0.925 |
| CTC_ACC| 0.999 | 1.000 |

**참고 링크:**

- 이미지 분류 데모: [Intel CPU 양자화 이미지 분류 예제](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/mkldnn_quant)  
- Ernie 데모: [ERNIE INT8 정밀도 및 성능 재현](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn)  
- LAC(GRU) 데모: [GRU INT8 재현](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/x86_gru_int8)  
- LSTM 데모: [LSTM INT8 재현](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/x86_lstm_demo)

## 3. PaddleSlim 양자화 모델 생성

X86 CPU 추론 엔진은 PaddleSlim에서 다음 두 방법으로 생성된 모델을 지원합니다:

- 정적 오프라인 양자화
- 양자화 훈련 (QAT)

자세한 내용은 아래 문서 참조:

- [정적 오프라인 양자화 빠른 시작](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/quant_post_static_tutorial.html)
- [양자화 훈련 빠른 시작](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/quant_aware_tutorial.html)
- [객체 탐지 양자화 튜토리얼](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/paddledetection_slim_quantization_tutorial.html)
- [양자화 API 문서](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/quantization_api.html)

주의사항:

- 정적 양자화(`quant_post_static`)는 `conv2d`, `depthwise_conv2d`, `mul`, `matmul` 연산자를 지원합니다.
- 양자화 훈련(`quant_aware`) 또한 위 네 연산자들을 지원하며, 설정 시 `quantize_op_types`에 해당 op들을 명시하면 됩니다.

## 4. 양자화 모델 변환

X86 CPU에서 양자화 모델을 추론 배포하기 전, PaddleSlim 모델을 최적화 및 변환해야 합니다.

### Paddle 설치

[Paddle 공식 홈페이지](https://www.paddlepaddle.org.cn/)를 참고하여 최신 CPU 또는 GPU 버전을 설치합니다.

### 변환 스크립트 준비

다음 스크립트를 다운로드합니다:

```
wget https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py
```

`save_quant_model.py` 스크립트의 파라미터 설명:

- `quant_model_path`: 입력 파라미터로 필수 항목입니다. PaddleSlim에서 생성한 양자화 모델의 경로를 지정합니다.
- `int8_model_save_path`: 변환된 INT8 양자화 모델을 저장할 경로입니다.

### 양자화 모델 변환

스크립트를 사용하여 양자화 모델을 변환하려면 예를 들어 다음과 같이 실행합니다:
```bash

python save_quant_model.py \
    --quant_model_path=/PATH/TO/SAVE/FLOAT32/QUANT/MODEL \
    --int8_model_save_path=/PATH/TO/SAVE/INT8/MODEL
```

## 5. Paddle Inference로 양자화 모델 배포하기

### 시스템 확인

- 터미널에서 `lscpu` 명령어를 입력하면 현재 시스템이 지원하는 명령어 집합을 확인할 수 있습니다.
- `avx512_vnni`를 지원하는 CPU 서버(예: Cascade Lake, 모델명: Intel(R) Xeon(R) Gold X2XX)에서는 INT8 정밀도 및 성능이 가장 우수하며, INT8 모델은 FP32 모델 대비 **3~3.7배** 빠릅니다.
- `avx512`는 지원하지만 `avx512_vnni`는 지원하지 않는 서버(예: SkyLake, 모델명: Intel(R) Xeon(R) Gold X1XX)에서는 INT8 성능이 FP32의 **약 1.5배** 수준입니다.
- 시스템이 **avx512 전체 명령어 집합을 지원**하는지 반드시 확인하세요.

### 추론 배포

모델 배포를 위해 Paddle Inference 추론 라이브러리를 준비합니다.  
다음 문서를 참고하여 X86 환경에서 추론을 구성할 수 있습니다:

- [X86 Linux 환경 추론 배포 예제](../demo_tutorial/x86_linux_demo)  
- [X86 Windows 환경 추론 배포 예제](../demo_tutorial/x86_windows_demo)

> ⚠️ 참고: X86 CPU에서 양자화 모델을 배포할 때는 반드시 **MKLDNN**과 **IrOptim**을 활성화해야 합니다.

#### C++ API 예제:

```cpp
paddle_infer::Config config;
config.SetModel("path/to/model_dir");  // 모델 경로 설정
config.EnableMKLDNN();                 // MKL-DNN 활성화 (필수)
config.SwitchIrOptim(true);           // IR 최적화 활성화 (필수)

// 필요한 경우 INT8 연산자 명시
// config.SetMkldnnCacheCapacity(10);
// config.EnableMkldnnInt8();

auto predictor = paddle_infer::CreatePredictor(config);  // 예측기 생성

```c++
paddle_infer::Config config;
if (FLAGS_model_dir == "") {
config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
} else {
config.SetModel(FLAGS_model_dir); // Load no-combined model
}
config.EnableMKLDNN();
config.SwitchIrOptim(true);
config.SetCpuMathLibraryNumThreads(FLAGS_threads);

auto predictor = paddle_infer::CreatePredictor(config);
```
Python API 예제:

```python
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_mkldnn()
config.switch_ir_optim(True)
config.set_cpu_math_library_num_threads(args.threads)

predictor = create_predictor(config)
```
