# Paddle Inference 소개

Paddle Inference는 플라잉페이(PaddlePaddle)의 네이티브 추론 라이브러리로, 서버 및 클라우드 환경에서 고성능 추론 기능을 제공한다.

Paddle Inference는 플라잉페이의 훈련 연산자(operator)를 기반으로 하기에, 플라잉페이로 훈련된 모든 모델을 범용적으로 지원한다.

Paddle Inference는 기능이 풍부하고 성능이 우수하며, 다양한 플랫폼과 애플리케이션 시나리오에 맞춰 깊이 있게 최적화되어 높은 처리량과 낮은 지연 시간을 구현한다. 이를 통해 플라잉페이 모델을 서버에서 즉시 훈련 후 바로 사용하고 빠르게 배포할 수 있다.

---

## Paddle Inference의 고성능 구현

### 메모리/그래픽 메모리 재사용으로 서비스 처리량 향상  
추론 초기화 단계에서 모델 내 연산(OP)의 출력 Tensor에 대해 의존성 분석을 수행하여, 서로 의존하지 않는 Tensor끼리 메모리/그래픽 메모리 공간을 재사용한다. 이를 통해 병렬 계산량을 늘리고 서비스 처리량을 향상시킨다.

### 세분화된 OP의 가로·세로 융합으로 계산량 감소  
추론 초기화 시 기존 융합 패턴에 따라 모델 내 여러 OP를 하나로 융합한다. 계산량과 Kernel Launch 횟수를 줄여 추론 성능을 개선한다. 현재 Paddle Inference는 수십 개에 달하는 융합 패턴을 지원한다.

### 내장 고성능 CPU/GPU Kernel  
Intel, Nvidia와 공동 개발한 고성능 커널을 내장하여 모델 추론의 높은 실행 성능을 보장한다.

### 서브그래프 형태로 TensorRT 통합, GPU 추론 속도 향상  
Paddle Inference는 GPU 추론 환경에서 TensorRT를 서브그래프 형태로 통합한다. TensorRT는 일부 서브그래프에 대해 OP의 가로·세로 융합, 불필요 OP 필터링, 최적 커널 자동 선택 등을 수행해 추론 속도를 가속화한다.

### 서브그래프 형태로 Paddle Lite 경량 추론 엔진 통합  
Paddle Lite는 플라잉페이의 경량 추론 엔진으로, 모바일뿐만 아니라 서버에서도 사용할 수 있다. Paddle Inference는 Paddle Lite를 서브그래프 형태로 통합하여, 기존 서버 추론 방식을 약간만 변경해도 Paddle Lite 추론을 사용할 수 있다. 이를 통해 추론 속도가 빨라지며, Baidu Kunlun 등 고성능 AI 칩에서도 실행 가능하다.

### PaddleSlim 양자화 및 압축 모델 지원  
PaddleSlim은 플라잉페이의 딥러닝 모델 압축 도구로, Paddle Inference는 PaddleSlim과 연동해 양자화, 가지치기, 증류한 모델을 불러와 배포할 수 있다. 이를 통해 모델 저장 공간을 줄이고 계산 메모리 점유를 낮추며 추론 속도를 높인다. 특히 X86 CPU에서 양자화 최적화를 심층 적용해, 대표적인 분류 모델의 단일 스레드 성능은 약 3배, ERNIE 모델은 약 2.68배 향상된다.

---

## Paddle Inference의 범용성

### 주요 소프트웨어 및 하드웨어 환경 호환  
서버 X86 CPU, NVIDIA GPU 칩을 지원하며 Linux, Mac, Windows 운영체제에 대응한다. 플라잉페이로 훈련된 모든 모델을 즉시 사용 가능하다.

### 다양한 언어 환경 지원 및 유연한 인터페이스  
C++, Python, C, Golang을 지원하며, 간단하고 유연한 API를 제공해 20줄 코드로도 배포할 수 있다. 그 외 언어의 경우 안정적인 ABI를 갖춘 C API를 제공하여 확장성을 보장한다.


---
원본


# Paddle Inference 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。

由于能力直接基于飞桨的训练算子，因此Paddle Inference 可以通用支持飞桨训练出的所有模型。

Paddle Inference 功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。

## Paddle Inference的高性能实现

### 内存/显存复用提升服务吞吐量

在推理初始化阶段，对模型中的OP输出Tensor 进行依赖分析，将两两互不依赖的Tensor在内存/显存空间上进行复用，进而增大计算并行量，提升服务吞吐量。

### 细粒度OP横向纵向融合减少计算量

在推理初始化阶段，按照已有的融合模式将模型中的多个OP融合成一个OP，减少了模型的计算量的同时，也减少了 Kernel Launch的次数，从而能提升推理性能。目前Paddle Inference支持的融合模式多达几十个。

### 内置高性能的CPU/GPU Kernel

内置同Intel、Nvidia共同打造的高性能kernel，保证了模型推理高性能的执行。

### 子图集成TensorRT加快GPU推理速度

Paddle Inference采用子图的形式集成TensorRT，针对GPU推理场景，TensorRT可对一些子图进行优化，包括OP的横向和纵向融合，过滤冗余的OP，并为OP自动选择最优的kernel，加快推理速度。

### 子图集成Paddle Lite轻量化推理引擎

Paddle Lite 是飞桨深度学习框架的一款轻量级、低框架开销的推理引擎，除了在移动端应用外，还可以使用服务器进行 Paddle Lite 推理。Paddle Inference采用子图的形式集成 Paddle Lite，以方便用户在服务器推理原有方式上稍加改动，即可开启 Paddle Lite 的推理能力，得到更快的推理速度。并且，使用 Paddle Lite 可支持在百度昆仑等高性能AI芯片上执行推理计算。

### 支持加载PaddleSlim量化压缩后的模型

PaddleSlim是飞桨深度学习模型压缩工具，Paddle Inference可联动PaddleSlim，支持加载量化、裁剪和蒸馏后的模型并部署，由此减小模型存储空间、减少计算占用内存、加快模型推理速度。其中在模型量化方面，Paddle Inference在X86 CPU上做了深度优化，常见分类模型的单线程性能可提升近3倍，ERNIE模型的单线程性能可提升2.68倍。

## Paddle Inference的通用性

### 主流软硬件环境兼容适配

支持服务器端X86 CPU、NVIDIA GPU芯片，兼容Linux/Mac/Windows系统。支持所有飞桨训练产出的模型，完全做到即训即用。

### 多语言环境丰富接口可灵活调用

支持 C++, Python, C, Golang，接口简单灵活，20行代码即可完成部署。对于其他语言，提供了 ABI 稳定的 C API, 用户可以很方便地扩展。

