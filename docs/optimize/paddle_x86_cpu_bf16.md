# X86 CPU에서 BF16 추론 배포

## 1. 개요

bfloat16 (Brain float Point) 부동소수점 형식은 컴퓨터 메모리에서 16비트를 사용하는 수치 데이터 형식입니다.  
이 형식은 32비트 IEEE 754 단정밀도 부동소수점 형식(float32)을 16비트로 절단한 버전으로, 부호 비트 1개, 지수 8비트, 가수 7비트를 유지하고, float32의 23비트 중 중요하지 않은 하위 16비트 가수를 제거합니다.  
bfloat16은 저장 공간을 줄이고, 머신러닝 알고리즘의 계산 속도를 높이기 위해 사용됩니다.  
bfloat16 데이터 형식에 대한 더 많은 정보는 [여기](https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf)에서 확인할 수 있습니다.  

현재 X86 CPU에서 bfloat16 추론은 PaddlePaddle에서 이미 지원되며, 그 결과는 아래와 같습니다.  
X86 CPU에서의 bfloat16 훈련은 개발 중입니다.

![](images/bfloat16.jpg)

## 2. Intel(R) CPU에서의 이미지 분류 및 자연어 처리 모델의 bfloat16 추론 정확도 및 성능

> **이미지 분류 모델 - Intel(R) Xeon(R) Platinum 8371HC CPU @ 3.30GHz 에서의 정확도 및 성능**

| 전체 데이터셋 | BF16 FPS 향상 (MKLDNN FP32 대비) | TOP1 정확도 (FP32) | TOP1 정확도 (BF16) | 정확도 감소 |
|---------------|:-------------------------------:|:------------------:|:------------------:|:------------:|
| resnet50      | 1.85배                          | 0.7663             | 0.7656             | 0.00091      |
| googlenet     | 1.61배                          | 0.705              | 0.7049             | 0.00014      |
| mobilenetV1   | 1.71배                          | 0.7078             | 0.7071             | 0.00099      |
| mobilenetV2   | 1.52배                          | 0.719              | 0.7171             | 0.00264      |

**Note:** Clas 모델 기준, batch_size=1, nr_threads=1

> **자연어 처리 모델 - Intel(R) Xeon(R) Platinum 8371HC CPU @ 3.30GHz 에서의 정확도 및 성능**

| GRU 정확도   | FP32     | BF16     | 차이      |
|--------------|----------|----------|-----------|
| Precision    | 0.89211  | 0.89225  | -0.00014  |
| Recall       | 0.89442  | 0.89457  | -0.00015  |
| F1 score     | 0.89326  | 0.89341  | -0.00015  |

| GRU 성능 (QPS)      | Naive FP32 | FP32     | BF16     | (BF16/FP32) |
|---------------------|------------|----------|----------|-------------|
| thread = 1          | 2794.97    | 2700.45  | 4210.27  | 1.56배      |
| thread = 4          | 3076.66    | 4756.45  | 6186.94  | 1.30배      |

**Note:** GRU 모델 기준, batch size = 50, iterations = 160

## 3. Paddle BF16 추론 재현

### 3.1 Paddle 설치

최신 CPU 또는 GPU 버전의 Paddle은 [Paddle 공식 홈페이지](https://www.paddlepaddle.org.cn/)를 참고하여 설치하세요.

### 3.2 시스템 확인

- 터미널에서 `lscpu` 명령어를 입력하면 현재 시스템이 지원하는 명령어 집합을 확인할 수 있습니다.
- Intel이 `avx512_bf16` 명령어를 지원하는 경우, (현재 Cooper Lake 계열에서 지원됨 — 예: Intel(R) Xeon(R) Platinum 8371HC, Intel(R) Xeon(R) Gold 6348H 등), 위 표와 같은 BF16 성능 향상을 기대할 수 있습니다.  
  👉 [Cooper Lake 제품 목록](https://ark.intel.com/content/www/us/en/ark/products/codename/189143/products-formerly-cooper-lake.html?wapkw=cooper%20lake)
- `avx512bw`, `avx512vl`, `avx512dq`는 지원하지만 `avx512_bf16`은 지원하지 않는 SkyLake, CasCade Lake 등의 시스템에서는 BF16 추론은 가능하나 성능 향상은 기대할 수 없습니다.
- 비호환 시스템에서의 테스트를 방지하려면 아래와 같은 방식으로 사전 체크하는 것이 좋습니다.

```
Python
import paddle
paddle.fluid.core.supports_bfloat16() // 如果为true, bf16可以顺利运行不报错，性能未知。
paddle.fluid.core.supports_bfloat16_fast_performance() // 如果为true, bf16可以顺利运行，且可获得上表所示的性能。

c++
#include "paddle/fluid/platform/cpu_info.h"
platform::MayIUse(platform::cpu_isa_t::avx512_core) // 如果为true, bf16可以顺利运行不报错，性能未知。
platform::MayIUse(platform::cpu_isa_t::avx512_bf16) // 如果为true, bf16可以顺利运行，且可获得上表所示的性能。
```

### 3.3 추론 배포

C++ API 예시는 다음과 같습니다:

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
// 将所可转为BF16的op转为BF16
config.EnableMkldnnBfloat16();
// 如果您想自己决定要替换哪些操作符，可以使用SetBfloat16Op选项
//config.SetBfloat16Op({“conv2d”、“pool2d”})

auto predictor = paddle_infer::CreatePredictor(config);
```

Python API举例如下:

```python
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_mkldnn()
config.switch_ir_optim(True)
config.set_cpu_math_library_num_threads(args.threads)
config.enable_mkldnn_bfloat16()
# 如果您想自己决定要替换哪些操作符，可以使用set_bfloat16_op选项
# config.set_bfloat16_op({"conv2d", "pool2d"})
predictor = create_predictor(config)
```
