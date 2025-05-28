# 예측 예제 (C++)

본 장은 두 부분으로 구성됩니다:  
(1) [C++ 예제 프로그램 실행하기](#id1)  
(2) [C++ 예측 프로그램 개발 안내](#id5)

## C++ 예제 프로그램 실행하기

### 1. 사전 빌드된 C++ 예측 라이브러리 다운로드

Paddle Inference는 Ubuntu/Windows/MacOS 플랫폼에 대해 공식 릴리스된 예측 라이브러리를 제공합니다. 위 플랫폼 중 하나를 사용 중이라면 아래 링크에서 직접 다운로드를 권장하며, 필요시 [소스 컴파일](../user_guides/source_compile.html) 방법도 참고할 수 있습니다.

- [Linux 예측 라이브러리 다운로드 및 설치](../user_guides/download_lib.html#linux)  
- [Windows 예측 라이브러리 다운로드 및 설치](../user_guides/download_lib.html#windows)

다운로드 후 압축을 해제하면, `paddle_inference_install_dir` 폴더가 C++ 예측 라이브러리이며, 폴더 구조는 다음과 같습니다:

```bash
paddle_inference/paddle_inference_install_dir/
├── CMakeCache.txt
├── paddle
│   ├── include                                    # C++ 예측 라이브러리 헤더 파일 디렉터리
│   │   ├── crypto
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 # C++ 예측 라이브러리 헤더 파일
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   └── paddle_pass_builder.h
│   └── lib
│       ├── libpaddle_inference.a                   # C++ 정적 예측 라이브러리 파일
│       └── libpaddle_inference.so                  # C++ 동적 예측 라이브러리 파일
├── third_party
│   ├── install                                    # 서드파티 링크 라이브러리 및 헤더 파일
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── openvino        # OpenVINO 추론 백엔드
│   │   ├── tbb             # OpenVINO 멀티스레드 백엔드
│   │   ├── protobuf
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt
```

`version.txt` 파일에는 예측 라이브러리의 버전 정보가 기록되어 있습니다.  
여기에는 Git 커밋 ID, OpenBlas 또는 MKL 수학 라이브러리 사용 여부, CUDA/CUDNN 버전 정보 등이 포함됩니다.  
예시는 다음과 같습니다:

```bash
GIT COMMIT ID: 1bf4836580951b6fd50495339a7a75b77bf539f6
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 9.0
CUDNN version: v7.6
CXX compiler version: 4.8.5
WITH_OPENVINO: ON
OpenVINO version: 2024.5.0
WITH_TENSORRT: ON
TensorRT version: v6
```
### 2. 예측 예제 코드 다운로드 및 컴파일

본 장의 C++ 예측 예제 코드는 [Paddle-Inference-Demo/c++/resnet50](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/resnet50) 경로에 있습니다. 디렉터리에는 다음과 같은 파일들이 포함되어 있습니다:

```bash
Paddle-Inference-Demo/c++/resnet50/
├── resnet50_test.cc   # 예측용 C++ 소스 코드
├── README.md          # README 설명 파일
├── compile.sh         # 컴파일 스크립트
└── run.sh             # 실행 스크립트
```

컴파일 및 예측 예제 실행 전에, 실행 환경에 맞게 컴파일 스크립트 `compile.sh`를 설정해야 합니다.

```bash
# 사전 빌드된 라이브러리의 version.txt 정보를 참고하여 아래 네 개 플래그를 ON/OFF 설정
WITH_MKL=ON       
WITH_GPU=ON         
WITH_OPENVINO=OFF    
USE_TENSORRT=OFF

# 예측 라이브러리의 루트 디렉터리 경로 설정 (본 장 1단계에서 다운로드/컴파일한 C++ 예측 라이브러리)
# paddle_inference로 이름 변경 후 ../lib 디렉터리 아래에 위치 가능
LIB_DIR=${work_path}/../lib/paddle_inference

# WITH_GPU 또는 USE_TENSORRT가 ON일 경우 CUDA, CUDNN, TENSORRT 경로를 설정
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-6.0.1.5
```

스크립트를 실행하여 컴파일하면, 현재 디렉터리에 build 폴더가 생성되고, 그 안에 build/resnet50_test 실행 파일이 만들어집니다.

```bash
bash compile.sh
```

### 3. 예측 프로그램 실행

**주의**: Paddle Inference에서 제공하는 C++ 예측 라이브러리는 사용자의 컴퓨터에 설치된 GCC 버전과 일치해야 합니다. 버전이 다를 경우 알 수 없는 오류가 발생할 수 있습니다.

`run.sh` 스크립트를 실행하여 예측 프로그램을 실행합니다.

```bash
bash run.sh
```

스크립트 설명:

```bash
# run.sh 스크립트는 먼저 예측 배포 모델을 다운로드합니다.
# 모델 구조를 확인하려면 `inference.pdmodel` 파일을 Netron과 같은 시각화 도구에서 열 수 있습니다.
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 다운로드한 모델을 로드하고 예측 프로그램을 실행합니다.
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```

성공적으로 실행된 후, 얻은 예측 출력 결과는 다음과 같습니다:

```bash
# 프로그램 출력 결과 예시
I1202 06:53:18.979496  3411 resnet50_test.cc:73] run avg time is 257.678 ms
I1202 06:53:18.979645  3411 resnet50_test.cc:88] 0 : 0
I1202 06:53:18.979676  3411 resnet50_test.cc:88] 100 : 2.04164e-37
I1202 06:53:18.979728  3411 resnet50_test.cc:88] 200 : 2.12382e-33
I1202 06:53:18.979768  3411 resnet50_test.cc:88] 300 : 0
I1202 06:53:18.979779  3411 resnet50_test.cc:88] 400 : 1.68493e-35
I1202 06:53:18.979794  3411 resnet50_test.cc:88] 500 : 0
I1202 06:53:18.979802  3411 resnet50_test.cc:88] 600 : 1.05767e-19
I1202 06:53:18.979810  3411 resnet50_test.cc:88] 700 : 2.04094e-23
I1202 06:53:18.979820  3411 resnet50_test.cc:88] 800 : 3.85254e-25
I1202 06:53:18.979828  3411 resnet50_test.cc:88] 900 : 1.52393e-30
```
## C++ 예측 프로그램 개발 안내

Paddle Inference를 사용하여 C++ 예측 프로그램을 개발하려면 다음 다섯 단계만 따르면 됩니다.

(1) 헤더 파일 포함

```c++
#include "paddle_inference_api.h"
```

(2) 설정 객체 생성 및 필요에 따라 설정  
자세한 내용은 [C++ API 문서 - Config](../api_reference/cxx_api_doc/Config_index) 참고

```c++
// 기본 설정 객체 생성
paddle_infer::Config config;

// 예측 모델 경로 설정 (본 장 2단계에서 다운로드한 모델)
config.SetModel(FLAGS_model_file, FLAGS_params_file);

// GPU 및 MKLDNN 예측 활성화
config.EnableUseGpu(100, 0);
config.EnableMKLDNN();

// 메모리 및 비디오 메모리 재사용 활성화
config.EnableMemoryOptim();
```

(3) Config를 기반으로 예측 객체 생성  
자세한 내용은 [C++ API 문서 - Predictor](../api_reference/cxx_api_doc/Predictor) 참고
```c++
auto predictor = paddle_infer::CreatePredictor(config);
```

(4) 모델 입력 Tensor 설정  
자세한 내용은 [C++ API 문서 - Tensor](../api_reference/cxx_api_doc/Tensor) 참고

```c++
// 입력 Tensor 가져오기
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);

// 입력 Tensor 차원 설정
std::vector<int> INPUT_SHAPE = {1, 3, 224, 224};
input_tensor->Reshape(INPUT_SHAPE);

// 입력 데이터 준비
int input_size = 1 * 3 * 224 * 224;
std::vector<float> input_data(input_size, 1);

// 입력 Tensor에 데이터 복사
input_tensor->CopyFromCpu(input_data.data());
```

5. 예측 실행  
자세한 내용은 [C++ API 문서 - Predictor](../api_reference/cxx_api_doc/Predictor) 참고

```c++
// 예측 실행
predictor->Run();
```

6. 예측 결과 획득  
자세한 내용은 [C++ API 문서 - Tensor](../api_reference/cxx_api_doc/Tensor) 참고

```c++
// 출력 Tensor 가져오기
auto output_names = predictor->GetOutputNames();
auto output_tensor = predictor->GetOutputHandle(output_names[0]);

// 출력 Tensor 차원 정보 가져오기
std::vector<int> output_shape = output_tensor->shape();
int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());

// 출력 Tensor 데이터 가져오기
std::vector<float> output_data;
output_data.resize(output_size);
output_tensor->CopyToCpu(output_data.data());
```
