# Linux에서의 GPU 추론 배포 예제

## 1. C++ 추론 배포 예제

C++ 예제 코드는 [링크](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/cuda_linux_demo)를 참고하세요. 아래에서는 `프로세스 분석`과 `컴파일 및 실행 예제` 두 부분으로 나누어 설명합니다.

### 1.1 절차 분석

#### 1.1.1 추론 라이브러리 준비

[추론 라이브러리 다운로드 문서](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)를 참고하여 Paddle C++ 추론 라이브러리를 다운로드하세요.  
이름에 `cuda`가 포함된 것이 GPU용입니다.

#### 1.1.2 추론 모델 준비

Paddle로 학습을 완료한 후 얻은 모델은 추론 배포에 사용할 수 있습니다.  
이 예제에서는 MobileNetV1 모델을 사용하며, [링크](https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz)에서 다운받거나 아래 명령어로 다운로드할 수 있습니다.

```bash
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz
```

#### 1.1.3 헤더 파일 포함

Paddle 추론 라이브러리를 사용하려면 `paddle_inference_api.h` 헤더 파일만 포함하면 됩니다.

```cpp
#include "paddle_inference_api.h"
```

#### 1.1.4 Config 설정

추론 배포의 실제 상황에 따라 Config를 설정하고, 이후 Predictor를 생성하는 데 사용합니다.  

기본적으로 Config는 CPU 추론을 사용합니다. GPU를 사용하려면 수동으로 활성화하고 GPU 번호와 초기 GPU 메모리를 설정해야 합니다.  
TensorRT 가속, IR 최적화, 메모리 최적화도 설정할 수 있습니다. 자세한 내용은 [문서](https://www.paddlepaddle.org.cn/inference/master/guides/nv_gpu_infer/gpu_trt_infer.html)를 참고하세요.

```cpp
paddle_infer::Config config;
if (FLAGS_model_dir == "") {
    config.SetModel(FLAGS_model_file, FLAGS_params_file); // 결합 모델
} else {
    config.SetModel(FLAGS_model_dir); // 비결합 모델
}
config.EnableUseGpu(500, 0);
config.SwitchIrOptim(true);
config.EnableMemoryOptim();
config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 10, PrecisionType::kFloat32, false, false);
```

#### 1.1.5 Predictor 생성

```cpp
std::shared_ptr<paddle_infer::Predictor> predictor = paddle_infer::CreatePredictor(config);
```

#### 1.1.6 입력 설정

Predictor에서 입력의 names와 handle을 가져온 다음 입력 데이터를 설정합니다.

```cpp
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
std::vector<int> input_shape = {1, 3, 224, 224};
std::vector<float> input_data(1 * 3 * 224 * 224, 1);
input_t->Reshape(input_shape);
input_t->CopyFromCpu(input_data.data());
```

#### 1.1.7 추론 실행

```cpp
predictor->Run();
```

#### 1.1.8 출력 가져오기

```cpp
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputHandle(output_names[0]);
std::vector<int> output_shape = output_t->shape();
int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
std::vector<float> out_data(out_num);
output_t->CopyToCpu(out_data.data());
```

### 1.2 컴파일 및 실행 예제

#### 1.2.1 컴파일 예제

`model_test.cc` 파일은 추론 샘플 프로그램입니다(프로그램의 입력은 고정값이므로, opencv나 다른 방식으로 데이터를 읽을 필요가 있다면 프로그램을 수정해야 합니다).
`CMakeLists.txt` 파일은 컴파일 빌드 파일입니다.
`run_impl.sh` 스크립트는 서드파티 라이브러리, 사전 컴파일된 라이브러리의 정보 설정을 포함합니다.

앞서 말한 단계에 따라 Paddle 추론 라이브러리와 mobilenetv1 모델을 다운로드합니다.
`run_impl.sh` 파일을 열고, LIB_DIR을 다운로드한 추론 라이브러리 경로로 설정합니다. 예: `LIB_DIR=/work/Paddle/build/paddle_inference_install_dir.`
`sh run_impl.s`h를 실행하면 현재 디렉토리에 build 디렉토리가 컴파일되어 생성됩니다.

#### 1.2.2 실행 예제

build 디렉토리로 이동하여 예제를 실행합니다.

```bash
cd build
./model_test --model_dir=mobilenetv1_fp32_dir
```

실행이 성공적으로 완료되면 결과가 화면에 출력됩니다.

---

## 2. Python 추론 배포 예제

Python 예제 코드는 [링크](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/cuda_linux_demo)를 참고하세요. 아래에서는 `프로세스 분석`과 `컴파일 및 실행 예제` 두 부분으로 나누어 설명합니다.

### 2.1 프로세스 분석

#### 2.1.1 환경 준비

- PaddlePaddle-gpu 설치: [공식 사이트](https://www.paddlepaddle.org.cn/)에서 2.0 이상 버전 설치
- Python에서 OpenCV 설치:

```bash
pip install opencv-python
```

#### 2.1.2 추론 모델 준비

Paddle 훈련이 완료되면 추론 모델을 얻게 되며, 이를 추론 배포에 사용할 수 있습니다.
본 예제에서는 mobilenet_v1 추론 모델을 준비했으며, [링크](https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz)에서 다운로드하거나 wget으로 다운로드할 수 있습니다.

```bash
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz
tar zxf mobilenetv1_fp32.tar.gz
```

#### 2.1.3 Python 모듈 임포트

```python
import paddle.inference as paddle_infer
```

#### 2.1.4 Config 설정

추론 배포의 실제 상황에 따라 Config를 설정하고, 이후 Predictor를 생성하는 데 사용합니다.  

기본적으로 Config는 CPU 추론을 사용합니다. GPU를 사용하려면 수동으로 활성화하고 GPU 번호와 초기 GPU 메모리를 설정해야 합니다.  
TensorRT 가속, IR 최적화, 메모리 최적화도 설정할 수 있습니다. 자세한 내용은 [문서](https://www.paddlepaddle.org.cn/inference/master/guides/nv_gpu_infer/gpu_trt_infer.html)를 참고하세요.

```python
if args.model_dir == "":
    config = paddle_infer.Config(args.model_file, args.params_file)
else:
    config = paddle_infer.Config(args.model_dir)

config.enable_use_gpu(500, 0)
config.switch_ir_optim()
config.enable_memory_optim()
config.enable_tensorrt_engine(
    workspace_size=1 << 30,
    precision_mode=paddle_infer.PrecisionType.Float32,
    max_batch_size=1,
    min_subgraph_size=5,
    use_static=False,
    use_calib_mode=False
)
```

#### 2.1.5 Predictor 생성

```python
predictor = paddle_infer.create_predictor(config)
```

#### 2.1.6 입력 설정

Predictor에서 입력의 names와 handle을 가져온 다음 입력 데이터를 설정합니다.

```python
img = cv2.imread(args.img_path)
img = preprocess(img)
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])
input_tensor.reshape(img.shape)
input_tensor.copy_from_cpu(img)
```

#### 2.1.7 추론 실행

```python
predictor.run()
```

#### 2.1.8 출력 가져오기

```python
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])
output_data = output_tensor.copy_to_cpu()
```

### 2.2 실행 예제

`img_preprocess.py` 파일은 이미지 전처리를 담당합니다.
`model_test.py` 파일은 예제 프로그램입니다.
앞선 단계를 참조하여 환경을 준비하고 추론 모델을 다운로드합니다.

추론용 이미지를 다운로드합니다.

```bash
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
```

추론 명령을 실행합니다.

```bash
python model_test.py --model_dir mobilenetv1_fp32 --img_path ILSVRC2012_val_00000247.jpeg
```

실행이 성공적으로 완료되면 결과가 화면에 출력됩니다.
