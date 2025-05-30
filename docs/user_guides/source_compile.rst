소스 코드 컴파일
========

언제 소스 코드 컴파일이 필요한가요?
--------------

딥러닝 기술은 매우 빠르게 발전하고 있으며, 연구자나 엔지니어들은 종종 사용자 정의 OP를 개발해야 하는 상황을 마주합니다. 이러한 경우 Python 레벨에서 OP를 작성할 수 있지만, 성능에 대한 요구가 높은 경우 C++ 레벨에서 개발해야 합니다. 이럴 때는 PaddlePaddle을 소스 코드로 직접 컴파일해야 적용할 수 있습니다.  
또한 대부분의 C++ 환경에서 모델을 배포하는 사용자라면, 공식 웹사이트에서 사전 컴파일된 예측 라이브러리를 직접 다운로드하여 바로 사용할 수 있습니다. [`PaddlePaddle 공식 홈페이지 <https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html>`_] 에는 다양한 환경에 맞춘 사전 컴파일된 예측 라이브러리가 제공되어 있습니다. 만약 사용자의 환경이 공식 사이트의 환경과 다르거나(CUDA, cuDNN, TensorRT 버전 차이 등), 소스 코드를 수정하거나 커스터마이징이 필요할 경우, 본 문서를 참고하여 소스 코드로 직접 컴파일하여 예측 라이브러리를 얻을 수 있습니다.

컴파일 원리
---------

**1: 최종 산출물**

PaddlePaddle 프레임워크의 소스 컴파일은 소스 코드의 컴파일과 링크를 포함하며, 최종 산출물은 다음과 같습니다:

 - C++ 인터페이스를 포함한 헤더 파일과 바이너리 라이브러리: C++ 환경에서 사용 가능하며, 지정된 경로에 파일을 배치하면 바로 사용할 수 있습니다.
 - Python Wheel 형식의 설치 패키지: Python 환경에서 사용. 앞서 설명한 pip 설치는 온라인 설치이며, 여기서는 로컬 설치 방식입니다.

**2: 기본 개념**

PaddlePaddle은 주로 C++로 작성되어 있으며, pybind 도구를 통해 Python 인터페이스를 제공합니다. PaddlePaddle의 소스 컴파일은 주로 컴파일과 링크 두 단계로 이루어집니다.  
* 컴파일 단계는 컴파일러가 수행하며, `.cc` 또는 `.cpp` 확장자의 파일 단위로 C++ 소스 코드를 바이너리 형태의 오브젝트 파일로 변환합니다. 일반적으로 여러 개의 소스 파일로 구성되기 때문에, 여러 오브젝트 파일이 생성됩니다.  
* 링크 단계는 이러한 오브젝트 파일을 하나의 실행 가능한 바이너리 파일로 조합하며, 외부 참조도 해결합니다. 이 바이너리에는 외부에서 재사용 가능한 함수 인터페이스도 포함될 수 있어, 라이브러리라고도 부릅니다. 링크 방식에 따라 정적 링크와 동적 링크로 나뉘며, 정적 링크는 오브젝트 파일을 아카이브하는 방식이고, 동적 링크는 로딩 시에 링크를 수행합니다.  
헤더 파일(`.h` 또는 `.hpp`)과 함께 사용하면 라이브러리 코드를 재사용하여 응용 프로그램을 개발할 수 있습니다. 정적 링크로 만들어진 프로그램은 독립 실행 가능하고, 동적 링크는 실행 시 지정된 경로에 있는 의존 라이브러리를 필요로 합니다.

**3: 컴파일 방식**

PaddlePaddle은 다양한 플랫폼을 지원하는 것을 목표로 설계되었습니다. 그러나 각 운영 체제는 사용되는 컴파일러와 링크 방식이 다릅니다. 예를 들어, Linux는 일반적으로 GCC를 사용하고, Windows는 MSVC를 사용합니다. 이를 통일하기 위해 PaddlePaddle은 CMake를 사용하여 다양한 컴파일러에 필요한 Makefile 또는 프로젝트 파일을 생성합니다.  
편리한 컴파일을 위해 CMake 명령어를 래핑하여 `cc_binary`, `cc_library` 등의 인터페이스를 제공하며, 이는 Bazel의 구조를 참조한 것입니다. 자세한 구현은 `cmake/generic.cmake`에서 확인할 수 있습니다. 또한 Python Wheel 패키지를 생성하는 로직도 CMake에 통합되어 있으며, 이에 대한 내용은 [`패키징 가이드 <https://packaging.python.org/tutorials/packaging-projects/>`] 를 참고하시기 바랍니다.

컴파일 단계
-----------

PaddlePaddle은 CPU 버전과 GPU 버전으로 나뉩니다. 사용자의 컴퓨터에 Nvidia GPU가 없다면 CPU 버전으로 설치하세요. 만약 CUDA / CuDNN이 설치된 Nvidia GPU가 있다면 GPU 버전으로 설치할 수 있습니다.

**권장 사양 및 의존 항목**

1. 안정적인 GitHub 연결, 1GHz 이상의 멀티코어 CPU, 9GB 이상의 디스크 공간  
2. GCC 4.8 또는 8.2 버전, 또는 Visual Studio 2015 Update 3  
3. Python 2.7 또는 3.5 이상, pip 9.0 이상, CMake 3.10 이상, Git 2.17 이상 (실행 편의를 위해 환경 변수에 등록)  
4. GPU 버전은 추가로 Nvidia CUDA 9/10, CuDNN v7 이상 필요 (필요 시 TensorRT도 필요)

Ubuntu 18.04 기반
------------

**1: 환경 준비**

위에서 언급한 의존 항목 외에 Ubuntu에서는 GCC8 컴파일러 등 도구도 필요합니다. 아래 명령어로 설치할 수 있습니다:

.. code:: shell

	sudo apt-get install gcc g++ make cmake git vim unrar python3 python3-dev python3-pip swig wget patchelf libopencv-dev
	pip3 install numpy protobuf wheel setuptools


CUDA 가속을 원한다면 CUDA와 cuDNN 환경을 구축해야 합니다. 아래는 CUDA 10.1, cuDNN 7.6 기준 설정 예입니다:

.. code:: shell

	# cuda
	sh cuda_10.1.168_418.67_linux.run
	export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda-10.1/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

	# cudnn
	tar -xzvf cudnn-10.1-linux-x64-v7.6.4.38.tgz
	sudo cp -a cuda/include/cudnn.h /usr/local/cuda/include/
	sudo cp -a cuda/lib64/libcudnn* /usr/local/cuda/lib64/

Ubuntu 18.04에서는 기본적으로 열 수 있는 파일 수가 1024로 제한되어 있어 컴파일 시 문제 발생 가능성이 있습니다.

/etc/security/limits.conf 파일에 다음 줄을 추가하세요:

.. code:: shell
 
	* hard noopen 102400
	* soft noopen 102400

컴퓨터 재부팅 후 다음 명령어로 현재 사용자로 재진입하여 설정을 적용하세요 (${user} 자리에 사용자명 입력):

.. code:: shell

	su ${user}
	ulimit -n 102400

TensorRT를 사용하는 경우, 가상 소멸자 오류가 발생할 수 있으니 NvInfer.h 파일의 IPluginFactory, IGpuAllocator 클래스에 가상 소멸자를 추가해야 합니다:

.. code-block:: c++

	virtual ~IPluginFactory() {};
	virtual ~IGpuAllocator() {};

2: 컴파일 명령어

PaddlePaddle 코드를 Git으로 클론하고 안정 버전으로 전환합니다 (예: release/2.0).
develop 브랜치는 최신 기능 개발용이며, release 브랜치는 안정 버전입니다. GitHub의 Releases에서 버전 기록을 확인하세요.

.. code:: shell

	git clone https://github.com/PaddlePaddle/Paddle.git
	cd Paddle
	git checkout release/2.0  

아래는 GPU 버전 예시입니다. CPU 버전은 WITH_GPU=OFF로 설정하세요.

.. code:: shell

	# build 디렉토리 생성
	mkdir build_cuda && cd build_cuda
	# cmake 실행
	cmake .. -DPY_VERSION=3 \
		-DWITH_TESTING=OFF \
		-DWITH_MKL=ON \
		-DWITH_GPU=ON \
		-DON_INFER=ON \
		..

**make로 컴파일**

make -j4

**Wheel 패키지 설치 (dist 디렉토리에서)**

pip3 install python/dist/paddlepaddle-2.0.0-cp38-cp38-linux_x86_64.whl

**예측 라이브러리 컴파일**

make inference_lib_dist -j4

**cmake 컴파일 환경 표**

다음은 공통된 cmake 빌드 방법으로, 환경에 따라 cmake 옵션을 수정하면 됩니다.

================  ============================================================================  =============================================================
      옵션                                             설명                                                                  기본값
================  ============================================================================  =============================================================
WITH_GPU          GPU를 지원할지 여부                                                                 ON
WITH_AVX          AVX 명령어 집합이 포함된 Paddle 바이너리를 컴파일할지 여부                        ON
WITH_PYTHON       PYTHON 인터프리터를 내장하고 Wheel 설치 패키지를 컴파일할지 여부                 ON
WITH_TESTING      유닛 테스트를 활성화할지 여부                                                      OFF
WITH_MKL          MKL 수학 라이브러리를 사용할지 여부 (아니면 OpenBLAS 사용)                         ON
WITH_SYSTEM_BLAS  시스템 기본 BLAS를 사용할지 여부                                                   OFF
WITH_DISTRIBUTE   분산 버전으로 컴파일할지 여부                                                       OFF
WITH_BRPC_RDMA    BRPC 및 RDMA를 RPC 프로토콜로 사용할지 여부                                        OFF
ON_INFER          추론 최적화를 활성화할지 여부                                                      OFF
CUDA_ARCH_NAME    현재 CUDA 아키텍처만 대상으로 컴파일할지 여부                                     All: 지원 가능한 모든 CUDA 아키텍처 컴파일; Auto: 현재 환경 자동 인식
WITH_TENSORRT     TensorRT를 활성화할지 여부                                                         OFF
TENSORRT_ROOT     TensorRT 라이브러리 경로, 지정하면 TRT 서브그래프 기능을 컴파일함 예: /paddle/nvidia/TensorRT/  /usr
================  ============================================================================  =============================================================

**三：NVIDIA Jetson 임베디드 하드웨어 추론 라이브러리 소스 코드 컴파일**

NVIDIA Jetson은 NVIDIA에서 출시한 임베디드 AI 플랫폼이며, Paddle Inference는 NVIDIA Jetson 플랫폼에서 추론 라이브러리를 컴파일하는 것을 지원합니다. 구체적인 단계는 다음과 같습니다:

1、환경 준비:

.. code:: shell

	# 하드웨어 성능 모드 활성화
	sudo nvpmodel -m 0 && sudo jetson_clocks
	# DDR 사용 가능 공간 증가. Xavier는 기본 메모리가 16 GB이므로 충분하지만, Nano에서는 다음 작업을 수행하세요.
	sudo fallocate -l 5G /var/swapfile
	sudo chmod 600 /var/swapfile
	sudo mkswap /var/swapfile
	sudo swapon /var/swapfile
	sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'

2、추론 라이브러리 컴파일:

.. code:: shell

	cd Paddle
	mkdir build
	cd build
	cmake .. \
	-DWITH_CONTRIB=OFF \
	-DWITH_MKL=OFF  \
	-DWITH_MKLDNN=OFF \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DON_INFER=ON \
	-DWITH_PYTHON=OFF \
	-DWITH_XBYAK=OFF  \
	-DWITH_NV_JETSON=ON
	make -j4
	
	# 추론 lib 생성
	make inference_lib_dist -j4

3、`공식 예제 <https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/inference_improving/paddle_tensorrt_infer.html#id2>`_ 를 참고하여 테스트하세요.


Windows 10 기반
-------------------

**1: 환경 준비**

이 절 앞에서 언급한 의존성 외에도, Windows 10에서 Paddle을 컴파일하려면 Visual Studio 2015 Update 3이 필요합니다. Paddle은 더 높은 버전에 대한 컴파일 지원을 지속적으로 개선하고 있습니다.

명령 프롬프트에 다음 명령어를 입력하여 필요한 Python 패키지를 설치하세요:

.. code:: shell

	pip3 install numpy protobuf wheel 

**2: 컴파일 명령어**
 
Git을 사용하여 Paddle 코드를 로컬로 클론하고 디렉터리에 들어가서 안정된 버전으로 전환합니다(git tag로 표시된 태그 이름, 예: release/2.0).  
**Paddle은 최신 기능 개발을 위해 develop 브랜치를 사용하고, 안정 버전은 release 브랜치에서 릴리스합니다. GitHub의 Releases 탭에서 Paddle 버전 릴리스 내역을 확인할 수 있습니다.**

.. code:: shell

	git clone https://github.com/PaddlePaddle/Paddle.git
	cd Paddle
	git checkout release/2.0
	
빌드 디렉터리를 만들고 그 안에서 CMake를 실행하여 솔루션 파일(Solution File)을 생성합니다. 아래는 CPU 버전 컴파일 예시이며, 다른 환경은 "CMake 컴파일 옵션 표"를 참고하여 cmake 옵션을 수정하세요.

.. code:: shell

	mkdir build
	cd build
	cmake .. -G "Visual Studio 14 2015 Win64" -A x64 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DON_INFER=ON 
		-DCMAKE_BUILD_TYPE=Release -DPY_VERSION=3

Visual Studio로 솔루션 파일을 열고, 상단의 빌드 구성 메뉴에서 Release x64를 선택한 뒤 "솔루션 빌드"를 클릭해 빌드를 완료하세요.  

**cmake 컴파일 환경 표**

================  ============================================================================  =============================================================
      옵션                                             설명                                                                  기본값
================  ============================================================================  =============================================================
WITH_GPU          GPU를 지원할지 여부                                                                 ON
WITH_AVX          AVX 명령어 집합이 포함된 Paddle 바이너리를 컴파일할지 여부                        ON
WITH_PYTHON       PYTHON 인터프리터를 내장하고 Wheel 설치 패키지를 컴파일할지 여부                 ON
WITH_TESTING      유닛 테스트를 활성화할지 여부                                                      OFF
WITH_MKL          MKL 수학 라이브러리를 사용할지 여부 (아니면 OpenBLAS 사용)                         ON
WITH_SYSTEM_BLAS  시스템 기본 BLAS를 사용할지 여부                                                   OFF
WITH_DISTRIBUTE   분산 버전으로 컴파일할지 여부                                                       OFF
WITH_BRPC_RDMA    BRPC 및 RDMA를 RPC 프로토콜로 사용할지 여부                                        OFF
ON_INFER          추론 최적화를 활성화할지 여부                                                      OFF
CUDA_ARCH_NAME    현재 CUDA 아키텍처만 대상으로 컴파일할지 여부                                     All: 지원 가능한 모든 CUDA 아키텍처 컴파일; Auto: 현재 환경 자동 인식
WITH_TENSORRT     TensorRT를 활성화할지 여부                                                         OFF
TENSORRT_ROOT     TensorRT 라이브러리 경로. 지정하면 TRT 서브그래프 기능을 컴파일함 예: /paddle/nvidia/TensorRT/  /usr
================  ============================================================================  =============================================================

**결과 확인**

**1: Python whl 패키지**

컴파일이 완료되면, `python/dist` 디렉터리에 Python Wheel 설치 패키지가 생성됩니다. 설치 및 테스트 명령은 다음과 같습니다:

.. code:: shell

	pip3 install paddlepaddle-2.0.0-cp38-cp38-win_amd64.whl  

설치가 완료되면, python3로 파이썬 인터프리터에 진입하여 아래 명령어를 입력합니다.  
`Your Paddle Fluid is installed successfully!` 문구가 출력되면 설치에 성공한 것입니다.

.. code:: python

	import paddle.fluid as fluid
	fluid.install_check.run_check()


**2: C++ 라이브러리**

추론 라이브러리 컴파일 후, 모든 출력물은 `build` 디렉터리 내 `paddle_inference_install_dir` 폴더에 생성됩니다.  
`version.txt` 파일에는 해당 추론 라이브러리의 버전 정보(Git Commit ID, OpenBLAS 또는 MKL 수학 라이브러리 사용 여부, CUDA/CUDNN 버전 등)가 기록되어 있습니다.

.. code:: shell

	build/paddle_inference_install_dir
	├── CMakeCache.txt
	├── paddle
	│   ├── include
	│   │   ├── paddle_anakin_config.h
	│   │   ├── paddle_analysis_config.h
	│   │   ├── paddle_api.h
	│   │   ├── paddle_inference_api.h
	│   │   ├── paddle_mkldnn_quantizer_config.h
	│   │   └── paddle_pass_builder.h
	│   └── lib
	│       ├── libpaddle_inference.a (Linux)
	│       ├── libpaddle_inference.so (Linux)
	│       └── libpaddle_inference.lib (Windows)
	├── third_party
	│   ├── boost
	│   │   └── boost
	│   ├── eigen3
	│   │   ├── Eigen
	│   │   └── unsupported
	│   └── install
	│       ├── gflags
	│       ├── glog
	│       ├── mkldnn
	│       ├── mklml
	│       ├── protobuf
	│       ├── xxhash
	│       └── zlib
	└── version.txt

`include` 디렉터리에는 Paddle 추론 라이브러리를 사용할 때 필요한 헤더 파일이 포함되어 있고, `lib` 디렉터리에는 생성된 정적/동적 라이브러리가 포함되어 있으며, `third_party` 디렉터리에는 추론 라이브러리가 의존하는 외부 라이브러리들이 포함되어 있습니다.

응용 프로그램 코드를 작성하여 이 추론 라이브러리와 함께 컴파일 및 결과 테스트를 할 수 있습니다.  
자세한 내용은 [`C++ 추론 라이브러리 API 사용법 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html>`_] 문서를 참고하세요.

기반: MacOSX 10.14
------------

**1: 환경 준비**

Paddle을 컴파일하기 전에, MacOSX에 Apple Clang 11.0과 Python 3.8, 그리고 python-pip이 사전 설치되어 있어야 합니다.  
다음 명령어를 사용해 Paddle 컴파일에 필요한 Python 패키지를 설치하세요.

.. code:: shell

	pip3 install numpy protobuf wheel setuptools


**2: 컴파일 명령어**

Git을 사용하여 Paddle 소스를 로컬에 클론하고, 디렉터리로 이동한 후, 안정적인 릴리스 버전으로 전환합니다 (git tag로 표시되는 태그명, 예: release/2.0).  
**Paddle은 최신 기능 개발을 위해 develop 브랜치를 사용하며, 안정 버전 배포는 release 브랜치를 통해 진행됩니다. GitHub의 Releases 탭에서 Paddle 버전 기록을 확인할 수 있습니다.**

.. code:: shell

	git clone https://github.com/PaddlePaddle/Paddle.git
	cd Paddle
	git checkout release/2.0    

다음은 CPU-MKL 버전을 예로 든 컴파일 명령어입니다.

.. code:: shell

	# build 디렉터리 생성 및 진입
	mkdir build && cd build
	# cmake 명령 실행
	cmake .. -DPY_VERSION=3 \
		-DWITH_TESTING=OFF \
		-DWITH_MKL=ON \
		-DON_INFER=ON \
		..

**make를 사용한 컴파일**

make -j4

**컴파일이 성공하면 dist 디렉터리에서 생성된 .whl 패키지를 확인할 수 있습니다**

pip3 install python/dist/paddlepaddle-2.0.0-cp38-cp38-macosx_10_14_x86_64.whl

**추론 라이브러리 컴파일**

make inference_lib_dist -j4


**cmake 컴파일 환경표**

다음은 공통적으로 사용되는 컴파일 설정 옵션입니다. 환경에 따라 해당 cmake 옵션을 조정하세요.

================  ============================================================================  =============================================================
      옵션                                            설명                                                                 기본값
================  ============================================================================  =============================================================
WITH_GPU          GPU 지원 여부                                                                  ON
WITH_AVX          AVX 명령어가 포함된 Paddle 바이너리 컴파일 여부                                 ON
WITH_PYTHON       Python 인터프리터 내장 및 Wheel 패키지 컴파일 여부                              ON
WITH_TESTING      단위 테스트 활성화 여부                                                        OFF
WITH_MKL          MKL 수학 라이브러리 사용 여부, 아니면 OpenBLAS 사용                              ON
WITH_SYSTEM_BLAS  시스템 기본 BLAS 사용 여부                                                      OFF
WITH_DISTRIBUTE   분산 컴파일 버전 활성화 여부                                                    OFF
WITH_BRPC_RDMA    BRPC, RDMA를 RPC 프로토콜로 사용할지 여부                                       OFF
ON_INFER          추론 최적화 기능 활성화 여부                                                    OFF
CUDA_ARCH_NAME    현재 CUDA 아키텍처만을 대상으로 컴파일할지 여부                                 All: 모든 지원 아키텍처 컴파일; Auto: 현재 환경 자동 감지
WITH_TENSORRT     TensorRT 활성화 여부                                                            OFF
TENSORRT_ROOT     TensorRT 라이브러리 경로 지정, 예: /paddle/nvidia/TensorRT/                    /usr
================  ============================================================================  =============================================================
