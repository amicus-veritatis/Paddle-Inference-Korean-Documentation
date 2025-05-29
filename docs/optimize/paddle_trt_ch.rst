사용 Paddle-TensorRT 라이브러리를 활용한 추론
===========================================

NVIDIA TensorRT는 NVIDIA 하드웨어에서의 딥러닝 모델 추론을 빠르고 효율적으로 수행하기 위한 고성능 머신러닝 추론 SDK입니다.  
PaddlePaddle은 TensorRT를 서브그래프 방식으로 통합하여, TensorRT로 가속할 수 있는 연산자들을 하나의 서브그래프로 구성해 TensorRT에 전달함으로써, PaddlePaddle의 즉시 학습-즉시 추론 기능을 유지하면서도 TensorRT의 가속 성능을 얻을 수 있습니다.  
이 문서에서는 Paddle-TRT를 사용하여 추론을 가속화하는 방법을 소개합니다.

만약 `TensorRT <https://developer.nvidia.com/nvidia-tensorrt-6x-download>`_ 를 설치해야 한다면, `TRT 공식 문서 <https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html>`_ 를 참고하세요.

개요
----

모델이 로드되면, 신경망은 변수와 연산 노드들로 구성된 계산 그래프로 표현됩니다.  
TRT 서브그래프 모드를 활성화하면, Paddle은 그래프 분석 단계에서 TensorRT로 최적화할 수 있는 서브그래프를 감지하여 이를 TensorRT 노드로 대체합니다.  
모델 추론 도중 TensorRT 노드를 만나면, Paddle은 해당 노드에 대해 TensorRT 라이브러리를 호출하여 추론을 수행하고, 나머지 노드는 Paddle의 기본 구현으로 처리합니다.

현재 Paddle-TRT는 **정적 shape** 과 **동적 shape** 두 가지 실행 방식을 지원합니다.

- *정적 shape*: 입력 size의 batch 차원을 제외한 나머지 차원이 변하지 않는 경우에 사용됩니다.
- *동적 shape*: 입력 size가 자유롭게 변할 수 있는 모델 (예: NLP, OCR 등)에 적합하며, 정적 shape을 지원하는 모델에도 적용 가능합니다.

정적 shape과 동적 shape 모두에서 다음과 같은 정밀도를 지원합니다:

- FP32
- FP16
- INT8

지원 하드웨어:

- 서버용 GPU: T4, A10 등
- 엣지 장비: Jetson NX, Jetson Nano, Jetson TX2 등
- 게이밍 GPU: RTX2080, RTX3090 등
- 엣지 장비에서는 DLA(Device Level Accelerator)도 지원됨

TensorRT는 최초 추론 시 다음과 같은 최적화를 수행합니다:

- 연산자 통합
- 메모리 재사용
- 커널 선택

이로 인해 첫 프레임의 지연 시간이 길 수 있습니다.  
이를 해결하기 위해 Paddle-TRT는 **직렬화(Serialization) 인터페이스**를 제공하여, TensorRT 분석 정보를 저장하고 이후 추론 시 재활용할 수 있도록 합니다.

.. note::

   1. 소스 코드에서 직접 빌드하는 경우, TensorRT 예측 라이브러리는 GPU 기반으로만 컴파일을 지원하며, 반드시 `TENSORRT_ROOT` 옵션을 TensorRT 설치 경로로 지정해야 합니다.
   2. Windows 환경에서는 TensorRT 5.0 이상이 필요합니다.
   3. 동적 shape 기능을 사용하려면 TensorRT 6.0 이상이 필요합니다.

1. 환경 준비
------------

Paddle-TRT 기능을 사용하려면 TensorRT가 포함된 Paddle 실행 환경이 필요하며, 다음과 같은 방법들이 제공됩니다:

1) Linux에서 pip 설치

`whl 리스트 <https://www.paddlepaddle.org.cn/inference/master/guides/install/download_lib.html>`_ 에서 TensorRT가 포함되어 있고 사용자의 환경에 맞는 whl 패키지를 다운로드한 후 pip로 설치하세요.

2) Docker 이미지 사용

.. code:: shell

    # Paddle 2.2 Python 환경이 사전 설치되어 있으며,
    # C++ 사전 컴파일 라이브러리(lib)는 홈 디렉토리 ~/ 에 저장됩니다.
    docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82

    sudo nvidia-docker run --name your_name -v $PWD:/paddle --network=host -it paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82 /bin/bash

3) 수동 컴파일  
컴파일 방법은 `컴파일 문서 <../user_guides/source_compile.html>`_ 를 참고하세요.

**Note1:** cmake 실행 시 다음 옵션을 설정하세요:  
- `TENSORRT_ROOT`: TensorRT 라이브러리 경로  
- `WITH_PYTHON`: Python whl 패키지 생성 여부 (ON으로 설정)

2. API 사용 소개
-----------------

`추론 절차 <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ 에 따르면, Paddle Inference를 사용한 추론은 다음과 같은 절차를 포함합니다:

- 추론 옵션 구성
- predictor 생성
- 모델 입력 준비
- 모델 추론 실행
- 모델 출력 획득

Paddle-TRT를 사용할 때도 이와 동일한 절차를 따릅니다.  
이제 간단한 예제를 통해 이 과정을 설명하겠습니다.  
(여기서는 사용자가 Paddle Inference에 어느 정도 익숙하다고 가정합니다. 만약 Paddle Inference가 처음이라면, `이곳 <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ 에서 기본 개념을 먼저 익히는 것을 추천합니다.)

.. code:: python

    import numpy as np
    import paddle.inference as paddle_infer
    
    def create_predictor():
        config = paddle_infer.Config("./resnet50/model", "./resnet50/params")
        config.enable_memory_optim()
        config.enable_use_gpu(1000, 0)
        
        # 打开TensorRT。此接口的详细介绍请见下文
        config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                      max_batch_size = 1, 
                                      min_subgraph_size = 3, 
                                      precision_mode=paddle_infer.PrecisionType.Float32, 
                                      use_static = False, use_calib_mode = False)

        predictor = paddle_infer.create_predictor(config)
        return predictor

    def run(predictor, img):
        # 准备输入
        input_names = predictor.get_input_names()
        for i,  name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)   
            input_tensor.copy_from_cpu(img[i])
        # 预测
        predictor.run()
        results = []
        # 获取输出
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    if __name__ == '__main__':
        pred = create_predictor()
        img = np.ones((1, 3, 224, 224)).astype(np.float32)
        result = run(pred, [img])
        print ("class index: ", np.argmax(result[0][0]))


이제 해당 인터페이스의 각 파라미터가 어떤 역할을 하는지 살펴보겠습니다:

- **workspace_size**: 타입: int, 기본값: 1 << 30 (1GB)  
  TensorRT가 사용하는 워크스페이스(작업 공간) 크기를 지정합니다. TensorRT는 이 제한된 크기 내에서 최적의 커널을 선택하여 추론을 수행합니다.

- **max_batch_size**: 타입: int, 기본값: 1  
  최대 배치 크기를 사전에 설정해야 하며, 실행 시 배치 크기는 이 값을 초과할 수 없습니다.

- **min_subgraph_size**: 타입: int, 기본값: 3  
  Paddle-TRT는 서브그래프 단위로 실행됩니다. 서브그래프 내부의 노드 개수가 `min_subgraph_size`보다 클 경우에만 Paddle-TRT로 실행되어 성능 손실을 방지합니다.

- **precision_mode**: 타입: **paddle_infer.PrecisionType**, 기본값: **paddle_infer.PrecisionType.Float32**  
  TensorRT가 사용할 정밀도를 지정합니다. FP32 (Float32), FP16 (Half), INT8 (Int8) 세 가지를 지원합니다.  
  Paddle-TRT의 INT8 오프라인 양자화 보정을 사용하려면 `precision_mode`를 **paddle_infer.PrecisionType.Int8** 로 설정하고, `use_calib_mode`를 True로 설정해야 합니다.

- **use_static**: 타입: bool, 기본값: False  
  True로 설정하면 첫 실행 시 TensorRT의 최적화 정보를 디스크에 직렬화(serialize)하고, 이후 실행 시 이를 직접 불러와 재생성 없이 빠르게 실행할 수 있습니다.

- **use_calib_mode**: 타입: bool, 기본값: False  
  Paddle-TRT의 INT8 오프라인 양자화 보정을 사용하려면 이 옵션을 True로 설정해야 합니다.

Int8 양자화 추론
>>>>>>>>>>>>>>>

딥러닝 모델의 가중치 파라미터는 어느 정도 중복성이 존재합니다. 많은 작업에서 모델을 양자화해도 계산 정확도에 거의 영향을 주지 않으며, 다음과 같은 이점이 있습니다:

- 메모리 접근량 감소
- 계산 효율 증가
- GPU 메모리 사용량 절감

INT8 양자화를 활용한 추론은 다음 두 단계로 진행됩니다:

1. 양자화 모델 생성
2. 양자화 모델을 로드하여 추론 실행

아래에서는 Paddle-TRT를 사용한 INT8 양자화 추론의 전체 절차를 설명합니다.

1. 양자화 모델 생성

현재 두 가지 방법으로 양자화 모델을 생성할 수 있습니다:

a. **TensorRT 자체의 INT8 오프라인 양자화 보정 기능 사용**  
   보정(Calibration)이란, 학습이 완료된 FP32 모델과 소량의 보정용 데이터(예: 이미지 500~1000장)를 기반으로 **보정 테이블 (Calibration Table)** 을 생성하는 과정입니다.  
   추론 시, 이 보정 테이블과 FP32 모델을 함께 로드하면 INT8 정밀도로 추론할 수 있습니다.  
   보정 테이블을 생성하려면 다음과 같이 설정합니다:

   - TensorRT 설정 시:
     - `precision_mode` 를 **paddle_infer.PrecisionType.Int8** 로 설정하고,
     - `use_calib_mode` 를 **True** 로 설정합니다.

    .. code:: python

      config.enable_tensorrt_engine(
        workspace_size=1<<30,
        max_batch_size=1, min_subgraph_size=5,
        precision_mode=paddle_infer.PrecisionType.Int8,
        use_static=False, use_calib_mode=True)

  - 약 500장 정도의 실제 입력 데이터를 준비한 후, 위의 설정을 적용하여 모델을 실행합니다.  
  (Paddle-TRT는 이 실행 과정에서 각 텐서의 값 범위를 수집하고, 이를 **보정 테이블(Calibration Table)** 에 기록합니다. 실행이 완료되면, 이 테이블은 모델 디렉토리 하위의 `_opt_cache` 디렉토리에 저장됩니다.)

  TensorRT의 INT8 오프라인 양자화 보정 기능을 사용하여 보정 테이블을 생성하는 전체 예제 코드를 확인하려면  
  `<https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/c%2B%2B/paddle-trt/trt_gen_calib_table_test.cc>`_ 의 데모를 참고하세요.

b. **PaddleSlim** 모델 압축 도구 라이브러리를 사용하여 양자화 모델 생성  
   PaddleSlim은 오프라인 양자화와 온라인 양자화 기능을 모두 지원합니다.

   - **오프라인 양자화**는 TensorRT 오프라인 보정 방식과 유사하게, 사전 학습된 모델과 소량의 데이터로 양자화 테이블을 생성하는 방식입니다.
   - **온라인 양자화(QAT, Quantization Aware Training)**는 다량의 데이터(예: 5000장 이상의 이미지)를 사용하여 학습 중에 양자화를 시뮬레이션하며, 가중치를 조정하여 양자화 오차를 줄이는 방식입니다.

   PaddleSlim을 활용한 양자화 모델 생성은 아래 문서를 참고하세요:

   - 오프라인 양자화 `빠른 시작 튜토리얼 <https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_post_tutorial.html>`_
   - 오프라인 양자화 `API 문서 <https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post>`_
   - 오프라인 양자화 `데모 코드 <https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.1.0/demo/quant/quant_post>`_
   - 양자화 훈련 `빠른 시작 튜토리얼 <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/dygraph/dygraph_quant_aware_training_tutorial.md>`_
   - 양자화 훈련 `API 문서 <https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-aware>`_
   - 양자화 훈련 `데모 코드 <https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.1.0/demo/quant/quant_aware>`_

오프라인 양자화의 장점은 재학습이 필요 없어 간단하고 사용이 쉬우며, 단점은 정밀도가 약간 감소할 수 있다는 점입니다.  
반면, 양자화 훈련은 정밀도 손실이 적지만 모델 재학습이 필요하기 때문에 사용 진입 장벽이 다소 높습니다.  
실제 사용에서는 먼저 **TensorRT의 오프라인 양자화 보정 기능**으로 양자화 모델을 생성해보고, 정밀도가 부족할 경우 **PaddleSlim**으로 모델을 재양자화하는 것을 추천합니다.

2. 양자화 모델을 로드하여 Int8 예측 수행
----------------------------------------

양자화 모델을 로드하여 Int8 추론을 수행하려면 TensorRT 설정 시 다음과 같이 지정해야 합니다:

- **precision_mode** 를 **paddle_infer.PrecisionType.Int8** 으로 설정합니다.

만약 사용 중인 양자화 모델이 TensorRT 오프라인 보정을 통해 생성된 모델이라면, 다음도 함께 설정해야 합니다:

- **use_calib_mode** 를 **True** 로 설정합니다.

  .. code:: python

    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=True)

전체 데모는 `여기 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E5%8A%A0%E8%BD%BD%E6%A0%A1%E5%87%86%E8%A1%A8%E6%89%A7%E8%A1%8Cint8%E9%A2%84%E6%B5%8B>`_ 를 참고하세요.

사용 중인 양자화 모델이 **PaddleSlim**에서 생성된 경우에는  
**use_calib_mode** 를 **False** 로 설정해야 합니다.


  .. code:: python

    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=False)

전체 데모는 `여기 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E4%B8%89%E4%BD%BF%E7%94%A8trt-%E5%8A%A0%E8%BD%BDpaddleslim-int8%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B>`_ 를 참고하세요.

Dynamic shape 실행
>>>>>>>>>>>>>>>>>>>

Paddle 1.8 버전부터 TRT 서브그래프에 대해 **Dynamic shape** 기능을 지원합니다.  
사용하는 인터페이스는 아래와 같습니다:

.. code:: python

	config.enable_tensorrt_engine(
		workspace_size = 1<<30,
		max_batch_size=1, min_subgraph_size=5,
		precision_mode=paddle_infer.PrecisionType.Float32,
		use_static=False, use_calib_mode=False)
		  
	min_input_shape = {"image":[1,3, 10, 10]}
	max_input_shape = {"image":[1,3, 224, 224]}
	opt_input_shape = {"image":[1,3, 100, 100]}

	config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)



위 사용 방식에서 알 수 있듯이, `config.enable_tensorrt_engine` 인터페이스를 호출한 후,  
새롭게 추가된 `config.set_trt_dynamic_shape_info` 인터페이스를 사용해야 합니다.

이 인터페이스는 모델 입력의 최소(min), 최대(max), 최적(opt) shape를 설정하는 데 사용됩니다.  
여기서 "최적(opt) shape"는 최소와 최대 사이의 값으로 설정되며, 추론 초기화 시 해당 shape를 기준으로 연산자(op)의 최적 커널(kernel)을 선택하게 됩니다.

`config.set_trt_dynamic_shape_info` 인터페이스를 호출하면, 예측기는 TRT 서브그래프에 대해 **동적 입력 모드(Dynamic Shape Mode)** 를 활성화하게 되며,  
실행 중에는 설정된 최소~최대 shape 범위 내의 어떤 입력도 받을 수 있습니다.

3. 테스트 예제
-------------

GitHub에서는 TRT 서브그래프 추론 사용을 보여주는 다양한 예제를 제공합니다:

- **Python 예제**: `링크 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/paddle_trt>`_
- **C++ 예제**: `링크 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt>`_

4. Paddle-TRT 서브그래프 실행 원리
-------------------------------

PaddlePaddle은 TensorRT를 서브그래프 방식으로 통합하여 사용합니다.  
모델이 로드되면, 신경망은 변수와 연산 노드로 구성된 계산 그래프(computational graph)로 표현됩니다.  
Paddle TensorRT의 기능은 이 전체 그래프를 스캔하여 TensorRT로 최적화할 수 있는 서브그래프를 찾아 해당 부분을 TensorRT 노드로 대체하는 것입니다.

모델 추론 중 TensorRT 노드를 만나면, Paddle은 TensorRT 라이브러리를 호출하여 해당 노드를 실행하고, 나머지 노드는 Paddle의 기본 연산으로 처리됩니다.

TensorRT는 추론 중 다음과 같은 최적화를 수행합니다:

- 연산자(Op)의 **가로 및 세로 방향 병합 (fuse)**  
- 불필요한 연산자 제거 (op 제거)  
- 특정 플랫폼에 최적화된 커널(kernel) 선택  

이러한 최적화는 모델 추론 속도를 상당히 향상시킬 수 있습니다.

아래 그림은 간단한 모델을 통해 이 과정을 시각적으로 보여줍니다:

**원본 네트워크**

    .. image:: https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png

**변환된 네트워크**

    .. image:: https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png

위 원본 네트워크에서 다음과 같은 색상 표현을 볼 수 있습니다:

- **녹색 노드**: TensorRT가 지원하는 노드
- **빨간색 노드**: 네트워크 내 변수
- **노란색 노드**: PaddlePaddle만이 처리할 수 있는 노드 (TensorRT가 지원하지 않음)

원본 네트워크의 녹색 노드들은 추출되어 하나의 서브그래프로 통합되고, 변환된 네트워크에서는 **block-25**라는 단일 TensorRT 노드로 대체됩니다.  
추론 중 이 노드에 도달하면, Paddle은 TensorRT 라이브러리를 호출하여 이 노드를 실행하게 됩니다.

