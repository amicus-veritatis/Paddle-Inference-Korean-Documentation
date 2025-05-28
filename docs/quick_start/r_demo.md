# 예측 예제 (R)

본 장은 두 부분으로 구성됩니다:  
(1) [R 예제 프로그램 실행하기](#id1)  
(2) [R 예측 프로그램 개발 안내](#id6)

## R 예제 프로그램 실행하기

### 1. R 예측 환경 설치

**방법 1:**  
Paddle Inference의 R 언어 예측은 Paddle Python 환경에 의존합니다.  
먼저 [공식 홈페이지 - 빠른 설치](https://www.paddlepaddle.org.cn/install/quick) 페이지를 참고하여 직접 설치 또는 컴파일하세요.  
현재 pip/conda 설치, 도커 이미지, 소스 컴파일 등 다양한 방식으로 Paddle Inference Python 환경을 준비할 수 있습니다.  
이후 R에서 Paddle 예측을 실행하는 데 필요한 라이브러리를 설치해야 합니다.

```bash
Rscript -e 'install.packages("reticulate", repos="https://cran.rstudio.com")'
```

**방법 2:**  
[Paddle/r/Dockerfile](https://github.com/PaddlePaddle/Paddle/blob/develop/r/Dockerfile)을 로컬로 다운로드한 후, 아래 명령어로 Docker 이미지를 빌드하고 컨테이너를 실행하세요:

```bash
# Docker 이미지 빌드
docker build -t paddle-rapi:latest .

# Docker 컨테이너 실행
docker run --rm -it paddle-rapi:latest bash
```

### 2. 예측 배포 모델 준비

[ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 모델을 다운로드한 후 압축을 해제하면, Paddle 예측 포맷 모델이 `resnet50` 폴더 내에 생성됩니다.  
모델 구조를 확인하려면 `inference.pdmodel` 파일명을 `__model__`로 변경한 뒤, 모델 시각화 도구 Netron으로 열면 됩니다.

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 생성된 모델 디렉터리 및 파일 목록
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 예측 배포 프로그램 준비

아래 코드를 `r_demo.r` 파일로 저장하고 실행 권한을 부여하세요:

```r
#!/usr/bin/env Rscript

library(reticulate) # Python 라이브러리 호출
use_python("/opt/python3.7/bin/python")

np <- import("numpy")
paddle_infer <- import("paddle.inference")

predict_run_resnet50 <- function() {
    # Config 생성
    config <- paddle_infer$Config("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
    
    # Config 기반으로 predictor 생성
    predictor <- paddle_infer$create_predictor(config)

    # 입력 이름 가져오기
    input_names <- predictor$get_input_names()
    input_handle <- predictor$get_input_handle(input_names[1])

    # 입력 설정
    input_data <- np$random$randn(as.integer(1 * 3 * 318 * 318))
    input_data <- np_array(input_data, dtype="float32")$reshape(as.integer(c(1, 3, 318, 318)))
    input_handle$reshape(as.integer(c(1, 3, 318, 318)))
    input_handle$copy_from_cpu(input_data)

    # 예측 실행
    predictor$run()

    # 출력 가져오기
    output_names <- predictor$get_output_names()
    output_handle <- predictor$get_output_handle(output_names[1])
    output_data <- output_handle$copy_to_cpu()
    output_data <- np_array(output_data)$reshape(as.integer(-1))
    print(paste0("Output data size is: ", output_data$size))
    print(paste0("Output data shape is: ", output_data$shape))
}

if (!interactive()) {
    predict_run_resnet50()
}
```

```r
# `use_python`에서 Python 실행 파일 경로 지정하기
# `reticulate` 패키지의 `use_python()` 함수를 사용하면 R에서 사용할 Python 실행 파일의 경로를 직접 지정할 수 있습니다.

use_python("/opt/python3.7/bin/python")
```

### 4. 예측 프로그램 실행

```bash
# 본 장 2단계에서 다운로드한 모델 폴더를 현재 작업 디렉터리로 이동한 후
./r_demo.r
```

성공적으로 실행된 후, 예측 출력 결과는 다음과 같습니다:

```bash
# 프로그램 출력 결과 예시
--- Running analysis [ir_graph_build_pass]
W1202 07:44:14.075577  6224 allocator_facade.cc:145] FLAGS_use_stream_safe_cuda_allocator is invalid for naive_best_fit strategy
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [layer_norm_fuse_pass]
---    Fused 0 subgraphs into layer_norm op.
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
--- Running IR pass [seqpool_cvm_concat_fuse_pass]
--- Running IR pass [mul_lstm_fuse_pass]
--- Running IR pass [fc_gru_fuse_pass]
---    fused 0 pairs of fc gru patterns
--- Running IR pass [mul_gru_fuse_pass]
--- Running IR pass [seq_concat_fc_fuse_pass]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
W1202 07:44:14.165925  6224 op_compat_sensible_pass.cc:219]  Check the Attr(transpose_Y) of Op(matmul) in pass(reshape2_matmul_fuse_pass) failed!
W1202 07:44:14.165951  6224 map_matmul_to_mul_pass.cc:668] Reshape2MatmulFusePass in op compat failed.
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
I1202 07:44:14.169189  6224 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [fc_fuse_pass]
I1202 07:44:14.170653  6224 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [repeated_fc_relu_fuse_pass]
--- Running IR pass [squared_mat_sub_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
I1202 07:44:14.219425  6224 fuse_pass_base.cc:57] ---  detected 53 subgraphs
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [conv_transpose_bn_fuse_pass]
--- Running IR pass [conv_transpose_eltwiseadd_bn_fuse_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I1202 07:44:14.268868  6224 analysis_predictor.cc:717] ======= optimize end =======
I1202 07:44:14.272181  6224 naive_executor.cc:98] ---  skip [feed], feed -> inputs
I1202 07:44:14.273878  6224 naive_executor.cc:98] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
[1] "Output data size is: 1000"
[1] "Output data shape is: (1000,)"
```

## R 예측 프로그램 개발 안내

Paddle Inference를 사용하여 R 예측 프로그램을 개발하려면 다음 다섯 단계만 따르면 됩니다.

1. R에서 Paddle Python 예측 라이브러리 불러오기

```r
library(reticulate) # Paddle 호출
use_python("/opt/python3.7/bin/python")

np <- import("numpy")
paddle_infer <- import("paddle.inference")
```

2. Config 객체 생성 및 필요에 따라 설정  
자세한 내용은 [Python API 문서 - Config](../api_reference/python_api_doc/Config_index) 참고

```r
# Config 생성 및 예측 모델 경로 설정
config <- paddle_infer$Config("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
```

3. Config를 기반으로 예측 객체 생성  
자세한 내용은 [Python API 문서 - Predictor](../api_reference/python_api_doc/Predictor) 참고

```r
predictor <- paddle_infer$create_predictor(config)
```

4. 모델 입력 Tensor 설정  
자세한 내용은 [Python API 문서 - Tensor](../api_reference/python_api_doc/Tensor) 참고

```r
# 입력 이름 가져오기
input_names <- predictor$get_input_names()
input_handle <- predictor$get_input_handle(input_names[1])

# 입력 데이터 설정
input_data <- np$random$randn(as.integer(1 * 3 * 318 * 318))
input_data <- np_array(input_data, dtype="float32")$reshape(as.integer(c(1, 3, 318, 318)))
input_handle$reshape(as.integer(c(1, 3, 318, 318)))
input_handle$copy_from_cpu(input_data)
```

5. 예측 실행  
자세한 내용은 [Python API 문서 - Predictor](../api_reference/python_api_doc/Predictor) 참고

```r
predictor$run()
```

6. 예측 결과 획득  
자세한 내용은 [Python API 문서 - Tensor](../api_reference/python_api_doc/Tensor) 참고

```r
output_names <- predictor$get_output_names()
output_handle <- predictor$get_output_handle(output_names[1])
output_data <- output_handle$copy_to_cpu()
output_data <- np_array(output_data)$reshape(as.integer(-1)) # numpy.ndarray类型
```
