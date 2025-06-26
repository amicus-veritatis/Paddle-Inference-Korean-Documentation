# Windows에서의 GPU 추론 배포 예제 

## 1 C++ 추론 배포 예제 

C++ 예제 코드는 [링크](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/cuda_linux_demo)를 참고하세요. 아래에서는 `프로세스 분석`과 `컴파일 및 실행 예제` 두 부분으로 나누어 설명합니다.

### 1.1 프로세스 분석

#### 1.1.1 추론 라이브러리 준비

[추론 라이브러리 다운로드 문서](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/windows_cpp_inference.html)를 참고하여 Windows 플랫폼용 Paddle GPU C++ 추론 라이브러리를 다운로드하세요.

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

기본적으로 Config는 CPU 추론을 사용합니다. GPU를 사용하려면 수동으로 활성화하고 GPU 번호와 초기 GPU 메모리를 설정해야 합니다. IR 최적화와 메모리 최적화를 활성화할 수 있습니다.

```cpp
paddle_infer::Config config;
if (FLAGS_model_dir == "") {
config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
} else {
config.SetModel(FLAGS_model_dir); // Load no-combined model
}
config.EnableUseGpu(500, 0);
config.SwitchIrOptim(true);
config.EnableMemoryOptim();
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
int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                              std::multiplies<int>());
std::vector<float> out_data;
out_data.resize(out_num);
output_t->CopyToCpu(out_data.data());
```

### 1.2 컴파일 및 실행 예제

#### 1.2.1 컴파일 예제

`model_test.cc` 파일은 추론 샘플 프로그램입니다(프로그램의 입력은 고정값이므로, opencv나 다른 방식으로 데이터를 읽을 필요가 있다면 프로그램을 수정해야 합니다).
`CMakeLists.txt` 파일은 컴파일 빌드 파일입니다.

앞서 말한 단계에 따라 Paddle 추론 라이브러리와 mobilenetv1 모델을 다운로드합니다.

cmake-gui 프로그램을 사용하여 Visual Studio 프로젝트를 생성:

- 소스 코드 경로와 빌드 결과물이 저장될 경로를 선택합니다. 아래 그림과 같이 설정하세요.

![win_x86_cpu_cmake_1](./images/win_x86_cpu_cmake_1.png)

- Configure 버튼을 클릭하고, Visual Studio를 선택한 후 x64 버전을 선택합니다. 아래 그림처럼 Finish를 클릭하세요.
필수 CMake 옵션을 아직 추가하지 않았기 때문에, 이 단계에서 Configure는 실패할 것입니다. 다음 단계로 진행하세요.

![win_x86_cpu_cmake_2](./images/win_x86_cpu_cmake_2.png)

- CMake 옵션을 설정합니다. Add Entry 버튼을 클릭해 PADDLE_LIB, CMAKE_BUILD_TYPE, DEMO_NAME 등의 항목을 추가합니다.
아래 그림처럼 설정하며, PADDLE_LIB 항목은 다운로드한 추론 라이브러리 경로로 지정합니다.

![win_x86_cpu_cmake_3](./images/win_x86_cpu_cmake_3.png)

- Configure를 클릭합니다. 로그 정보에 "Configure done"이 표시되면 설정이 성공한 것입니다. 다음으로 Generate를 클릭하여 VS 프로젝트를 생성합니다. 로그 정보에 "Generate done"이 표시되면 생성이 성공한 것입니다. 마지막으로 Open Project를 클릭하여 Visual Studio를 엽니다.

- Release/x64로 설정하고 컴파일합니다. 컴파일된 결과물은 build/Release 디렉터리에 생성됩니다.

![win_x86_cpu_vs_1](./images/win_x86_cpu_vs_1.png)

#### 1.2.2 실행 예제 

먼저 model_test 프로젝트를 시작 프로젝트로 설정하세요.

![win_x86_cpu_vs_2](./images/win_x86_cpu_vs_2.png)

입력 flags를 설정합니다. 즉, 이전에 다운로드한 모델 경로를 설정합니다. Debug 탭의 `model_test Properties..`를 클릭합니다.

![win_x86_cpu_vs_3](./images/win_x86_cpu_vs_3.png)

Debug 탭에서 Start Without Debugging 옵션을 클릭하여 프로그램 실행을 시작합니다.

![win_x86_cpu_vs_4](./images/win_x86_cpu_vs_4.png)

## 2. Python 추론 배포 예제

Python 예제 코드는 [링크](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/x86_linux_demo)를 참고하세요. 아래에서는 `프로세스 분석`과 `컴파일 및 실행 예제` 두 부분으로 나누어 설명합니다.

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

```shell
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz
tar zxf mobilenetv1_fp32.tar.gz
```

#### 2.1.3 Python 모듈 임포트 

```
from paddle.inference import Config
from paddle.inference import create_predictor
```

#### 2.1.4 Config 설정

추론 배포의 실제 상황에 따라 Config를 설정하고, 이후 Predictor를 생성하는 데 사용합니다.  

기본적으로 Config는 CPU 추론을 사용합니다. GPU를 사용하려면 수동으로 활성화하고 GPU 번호와 초기 GPU 메모리를 설정해야 합니다. IR 최적화와 메모리 최적화를 활성화할 수 있습니다.

```python
# args 파싱된 입력 인자
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_use_gpu(500, 0)
config.switch_ir_optim()
config.enable_memory_optim()
```

#### 2.1.5 Predictor 생성

```python
predictor = create_predictor(config)
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
predictor.run();
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

```shell
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
```

추론 명령을 실행합니다.

```
python model_test.py --model_dir mobilenetv1_fp32 --img_path ILSVRC2012_val_00000247.jpeg
```

실행이 성공적으로 완료되면 결과가 화면에 출력됩니다.
