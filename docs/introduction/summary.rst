개요
====

Paddle Inference는 **파들(飞桨, PaddlePaddle)** 핵심 프레임워크의 추론 엔진입니다.  
Paddle Inference는 기능이 풍부하고 성능이 뛰어나며, 다양한 플랫폼과 응용 시나리오에 대해 깊이 있는 최적화를 제공하여 **높은 처리량(Throughput), 낮은 지연시간(Latency)** 을 달성합니다. 이를 통해 Paddle 모델은 서버 환경에서 학습 후 즉시 사용할 수 있고, 빠르게 배포할 수 있습니다.

특징
----

- **범용성**: Paddle에서 학습한 모든 모델을 추론에 사용할 수 있습니다.

- **메모리/비디오 메모리 재사용**: 추론 초기화 시, 모델 내 OP 출력 Tensor의 의존성을 분석하여 서로 의존성이 없는 Tensor들을 동일한 메모리/비디오 메모리 공간에 재사용합니다. 이로 인해 계산 병렬성이 증가하고, 서비스 처리량이 향상됩니다.

- **세분화된 OP Fusion (연산자 결합)**: 추론 초기화 단계에서 사전 정의된 Fusion 패턴에 따라 여러 OP를 하나의 OP로 결합합니다. 이를 통해 계산량과 커널 실행 횟수를 줄여 추론 성능을 개선할 수 있습니다. Paddle Inference는 수십 가지 이상의 Fusion 패턴을 지원합니다.

- **고성능 CPU/GPU 커널**: Intel, NVIDIA와 공동 개발한 고성능 커널이 내장되어 있어, 모델 추론 시 높은 성능을 보장합니다.

- **TensorRT 하위 그래프 통합**  
  [`TensorRT`](https://developer.nvidia.com/tensorrt)는 NVIDIA의 고속 추론 엔진이며, Paddle Inference는 하위 그래프(subgraph) 방식으로 이를 통합합니다. GPU 추론 환경에서 TensorRT는 연산자의 가로/세로 방향 Fusion, 불필요한 OP 제거, 최적 커널 자동 선택 등의 최적화를 수행해 추론 속도를 높입니다.

- **MKLDNN 통합 지원**

- **PaddleSlim으로 양자화/경량화된 모델 로드 지원**  
  [`PaddleSlim`](https://github.com/PaddlePaddle/PaddleSlim)은 파들 모델을 양자화, 압축, 지식 증류 등으로 경량화하는 도구입니다. Paddle Inference는 PaddleSlim과 연동되어 양자화/프루닝/증류된 모델을 직접 로드 및 배포할 수 있습니다.  
  특히 X86 CPU 환경에서 양자화 모델에 대해 심층적인 최적화가 적용되어,  
  [분류 모델은 단일 스레드에서 최대 3배](https://github.com/PaddlePaddle/PaddleSlim/tree/80c9fab3f419880dd19ca6ea30e0f46a2fedf6b3/demo/mkldnn_quant/quant_aware),  
  ERNIE 모델은 2.68배의 성능 향상이 가능합니다.

지원 시스템 및 하드웨어
--------------------

- 서버용 X86 CPU 및 NVIDIA GPU 지원
- Linux, macOS, Windows 운영체제 지원
- NVIDIA Jetson 임베디드 플랫폼도 지원

지원 언어
--------

- Python
- C++
- Go
- R

다음 단계
--------

- Paddle Inference가 처음이라면 [`빠른 시작 가이드`](./quick_start.html)를 참고하세요.
