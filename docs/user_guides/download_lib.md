# Linux 추론 라이브러리 다운로드 및 설치  
## C++ 추론 라이브러리

- 사전 컴파일된 패키지 사용 방법은 다음을 참조하세요: [추론 예제(C++)](../quick_start/cpp_demo.md)

| 하드웨어 백엔드 | AVX 활성화 여부 | 수학 라이브러리 | gcc 버전 | CUDA/cuDNN/TensorRT 버전 | 추론 라이브러리(버전 2.4.0) |
|--------------|--------------|--------------|--------------|--------------|:-----------------|
|CPU|예|MKL|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|
|CPU|예|MKL|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference.tgz)|
|CPU|예|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference.tgz)|
|CPU|예|OpenBLAS|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz)|
|CPU|아니오|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference.tgz)|
|CPU|아니오|OpenBLAS|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference.tgz)|
|GPU|예|MKL|8.2|CUDA10.2/cuDNN7.6/TensorRT7.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddle_inference.tgz)|
|GPU|예|MKL|8.2|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|예|MKL|5.4|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|예|MKL|8.2|CUDA11.2/cuDNN8.1/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.1.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|예|MKL|8.2|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|예|MKL|5.4|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|예|MKL|8.2|CUDA11.6/cuDNN8.4/TensorRT8.4|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference.tgz)|
|GPU|예|MKL|8.2|CUDA11.7/cuDNN8.4/TensorRT8.4|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.7_cudnn8.4.1-trt8.4.2.4/paddle_inference.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.5_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.5_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.5_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.5_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack5.0.2_gcc9.4/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack5.0.2_gcc9.4/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack5.0.2_gcc9.4/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Jetson/jetpack5.0.2_gcc9.4/xavier/paddle_inference_install_dir.tgz)|

## C 추론 라이브러리

- 사전 컴파일된 패키지 사용 방법은 다음을 참조하세요: [추론 예제(C)](../quick_start/c_demo.md)

| 하드웨어 백엔드 | AVX 활성화 여부 | 수학 라이브러리 | gcc 버전 | CUDA/cuDNN/TensorRT 버전 | 추론 라이브러리(버전 2.4.0) |
|----------|----------|----------|----------|:---------|:--------------|
|CPU|예|MKL|8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference_c.tgz)|
|CPU|예|MKL|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference_c.tgz)|
|CPU|예|OpenBLAS|8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference_c.tgz)|
|CPU|예|OpenBLAS|5.4| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference_c.tgz)|
|CPU|아니오|OpenBLAS|8.2| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference_c.tgz)|
|CPU|아니오|OpenBLAS|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference_c.tgz)|
|GPU|예|예|8.2|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|예|예|8.2|CUDA10.2/cuDNN7.6/TensorRT7.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddle_inference_c.tgz)|
|GPU|예|예|5.4|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|예|예|8.2|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|
|GPU|예|예|8.2|CUDA11.2/cuDNN8.1/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.1.1_trt8.0.3.4/paddle_inference_c.tgz)|
|GPU|예|예|5.4|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|
|GPU|예|예|8.2|CUDA11.6/cuDNN8.4/TensorRT8.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference_c.tgz)|
|GPU|예|예|8.2|CUDA11.7/cuDNN8.4/TensorRT8.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.7_cudnn8.4.1-trt8.4.2.4/paddle_inference_c.tgz)|

## Python 추론 라이브러리

- 사전 컴파일된 패키지 사용 방법은 다음을 참조하세요: [추론 예제(Python)](../quick_start/python_demo.md)

| 버전 설명 |     python3.6  |   python3.7   |     python3.8     |     python3.9   |     python3.10   |
|:---------|:----------------|:-------------|:-------------------|:----------------|:----------------|
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-all|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.5_gcc7.5/all/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-nano|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.5_gcc7.5/nano/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-tx2|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.5_gcc7.5/tx2/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-xavier|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.5_gcc7.5/xavier/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-all|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6_gcc7.5/all/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-nano|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6_gcc7.5/nano/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-tx2|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6_gcc7.5/tx2/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-xavier|[paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6_gcc7.5/xavier/paddlepaddle_gpu-2.4.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-all||[paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6.1_gcc7.5/all/paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-nano||[paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6.1_gcc7.5/nano/paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-tx2||[paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6.1_gcc7.5/tx2/paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-xavier||[paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack4.6.1_gcc7.5/xavier/paddlepaddle_gpu-2.4.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-all|||[paddlepaddle_gpu-2.4.0-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack5.0.2_gcc9.4/all/paddlepaddle_gpu-2.4.0-cp38-cp38-linux_aarch64.whl)||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-nano|||[paddlepaddle_gpu-2.4.0-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack5.0.2_gcc9.4/nano/paddlepaddle_gpu-2.4.0-cp38-cp38-linux_aarch64.whl)||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-tx2|||[paddlepaddle_gpu-2.4.0-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack5.0.2_gcc9.4/tx2/paddlepaddle_gpu-2.4.0-cp38-cp38-linux_aarch64.whl)||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-xavier|||[paddlepaddle_gpu-2.4.0-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.0/python/Jetson/jetpack5.0.2_gcc9.4/xavier/paddlepaddle_gpu-2.4.0-cp38-cp38-linux_aarch64.whl)||


# Windows 추론 라이브러리 다운로드 및 설치

환경 하드웨어 구성:

| 운영체제      |    Windows 10 홈 버전      |
|:---------|:--------------------|
| CPU      |      I7-8700K       |
| 메모리   | 16G                 |
| 저장장치 | 1T HDD + 256G SSD   |
| 그래픽카드 | GTX1080 8G        |

## C++ 추론 라이브러리

- 사전 컴파일된 패키지 사용 방법은 다음을 참조하세요: [추론 예제(C++)](../quick_start/cpp_demo.md)

| 하드웨어 백엔드 | AVX 사용 여부 | 컴파일러 | CUDA/cuDNN/TensorRT 버전 | 수학 라이브러리 | 추론 라이브러리(버전 2.4.0) |
|--------------|--------------|:----------------|:--------|:-------------|:-----------------|
| CPU | 예 |  MSVC 2017 | - |MKL|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2017/paddle_inference.zip)|
| CPU | 예 | MSVC 2017 | - |OpenBLAS|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Windows/CPU/x86-64_avx-openblas-vs2017/paddle_inference.zip)|
| GPU | 예 | MSVC 2017  | CUDA10.2/cuDNN7.6/TensorRT7.0 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/Windows/GPU/x86-64_cuda10.2_cudnn7.6.5_trt7.0.0.11_mkl_avx_vs2017/paddle_inference.zip)|
| GPU | 예 | MSVC 2019  | CUDA11.2/cuD오 |OpenBLAS|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.0/cxx_c/MacOS/m1_clang_noavx_openblas/paddle_inference_c_install_dir.tgz)|
