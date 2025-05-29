# Paddle Inference FAQ (한글 번역)

## 1. 컴파일 에러, 명백한 문법 오류 없음  
**답:** 사용 중인 GCC 버전을 확인하세요. PaddlePaddle은 GCC 4.8.2와 8.2.0을 지원합니다.

## 2. 컴파일 에러: `No CMAKE_CUDA_COMPILER could be found`  
**답:** `nvcc`를 찾지 못한 것입니다. `-DCMAKE_CUDA_COMPILER=nvcc의_경로`를 설정하세요. CUDA 버전과 일치해야 합니다.

## 3. 런타임 에러: `cudaErrorNoKernelImageForDevice`  
**답:** 컴파일 시와 실행 시 GPU 아키텍처가 다르거나 CMake에서 CUDA 아키텍처를 지정하지 않았을 때 발생합니다. `-DCUDA_ARCH_NAME=All` 또는 `Turing`, `Volta` 등으로 지정하세요.

## 4. 런타임 에러: `id:0 >= GetCUDADeviceCount():0`  
**답:** 드라이버(libcuda.so)를 찾을 수 없습니다. `LD_LIBRARY_PATH`에 `/usr/lib64`를 추가하세요.

## 5. 에러: `Cannot load cudnn shared library.`  
**답:** cuDNN 라이브러리 경로를 `LD_LIBRARY_PATH`에 추가하세요.

## 6. `libstdc++.so.6`의 GLIBCXX 버전 오류  
**답:** 잘못된 glibc가 링크된 것입니다.  
- 컴파일 시:  -Wl,--rpath=/opt/compiler/gcc-8.2/lib,--dynamic-linker=/opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2
- 실행 시: 올바른 glibc 경로를 `LD_LIBRARY_PATH`에 추가하세요.

## 7. MKLDNN 사용 시 CPU 메모리 급증  
**답:** 입력이 가변 길이인 경우 `config.set_mkldnn_cache_capacity(capacity)`로 캐시 수를 제한하세요.

## 8. 에러: `CUDNN_STATUS_NOT_INITIALIZED`  
**답:** 실행 시 연결된 cuDNN 버전과 컴파일 시 사용된 cuDNN 버전이 일치하는지 확인하세요.

## 9. 에러: `OMP: Error #100 Fatal system error detected`  
**답:** OpenMP 관련 문제입니다. [참고 링크](https://unix.stackexchange.com/questions/302683/omp-error-bash-on-ubuntu-on-windows)

## 10. SIGILL 에러  
**답:** AVX 명령어를 지원하지 않는 머신에서 AVX 기반 예측 라이브러리를 사용한 경우입니다.

## 11. 에러: `an illegal memory access was encountered`  
**답:** 입력 Tensor에서 포인터가 잘못된 메모리를 참조했는지 확인하세요.

## 12. Predictor에 프로파일 기능이 있나요?  
**답:** `config.EnableProfile()`을 사용하면 op별 실행 시간을 출력합니다. [API 문서](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/Config_index.html)를 참고하세요.

## 13. 모델 추론 시간이 불안정함  
**답:**  
1) 하드웨어 자원이 공유되고 있는지 확인  
2) 입력이 일정한지 확인  
3) TensorRT의 초기 최적화 시간이 포함된 것인지 확인하고 warm-up 수행

## 14. ZeroCopyTensor 및 ZeroCopyRun 관련 문서  
**답:** ZeroCopyTensor는 추론 시 복사가 없지만, 생성 시 복사가 필요합니다. 2.0rc1 이후에는 해당 API가 숨겨졌으며 [공식 문서](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)를 참고하세요.

## 15. Jetson + JetPack 4.4에서 `std::logic_error` 발생  
**답:** cuDNN 8.0에서 SM_72 아키텍처를 사용할 때의 버그입니다. [관련 포럼](https://forums.developer.nvidia.com/t/nx-jp44-cudnn-internal-logic-error/124805) 참조  
- 해결 방법 1: 다음 PASS 제거
```cpp
config.pass_builder()->DeletePass("conv_elementwise_add_act_fuse_pass");
config.pass_builder()->DeletePass("conv_elementwise_add2_act_fuse_pass");
config.pass_builder()->DeletePass("conv_elementwise_add_fuse_pass");
```
- 해결 방법 2: cuDNN 7.6으로 다운그레이드

## 16. 에러: col >= feed_list.size()
**답:** 2.0 rc1 이전에는 config.SwitchUseFeedFetchOps(false)를 명시해야 합니다. 이후 버전에서는 필요 없습니다.
## 17. CPU 예측에서 멀티스레드 가속을 설정하려면?
답: 다음과 같이 설정합니다:

```cpp
config.EnableMKLDNN();
config.SetCpuMathLibraryNumThreads(스레드_수);
```
