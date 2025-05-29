# **兆芯下从源码编译**

## 검증된 모델 목록

- resnet50
- mobilenetv1
- ernie
- ELMo

## 환경 준비

* **프로세서：ZHAOXIN KaiSheng KH-37800D**
* **운영 체제：centos7**
* **Python 버전：2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip 또는 pip3 버전：9.0.1+ (64 bit)**

ZHAOXIN은 x86 아키텍처이며, 컴파일 방법은 [Linux에서 CPU 버전 소스 컴파일](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html#compile_from_host)과 동일합니다.

## 설치 단계

이 문서에서는 ZHAOXIN 프로세서 환경에서 Paddle을 설치하는 전 과정을 설명합니다.

<a name="zhaoxin_source"></a>
### **소스 코드 컴파일**

1. Paddle은 cmake를 통해 컴파일되며, cmake 버전은 3.10 이상이어야 합니다. 운영 체제의 패키지 소스에 적합한 버전이 있다면 해당 버전을 설치하면 됩니다. 없을 경우 [소스 설치](https://github.com/Kitware/CMake)를 진행하세요.

    ```
    wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
    tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
    ./bootstrap && make && sudo make install
    ```

2. Paddle은 내부적으로 patchelf를 사용하여 동적 라이브러리의 rpath를 수정합니다. 운영 체제의 소스에 patchelf가 포함되어 있다면 설치만 하면 되고, 그렇지 않으면 [patchelf 공식 문서](https://github.com/NixOS/patchelf)를 참고하여 소스로 설치해야 합니다.

    ```
    ./bootstrap.sh
    ./configure
    make
    make check
    sudo make install
    ```

3. Paddle의 소스 코드를 현재 디렉터리에 `Paddle` 폴더로 클론한 후, 해당 폴더로 이동합니다.

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    cd Paddle
    ```

4. 안정적인 release 브랜치로 전환하여 컴파일합니다.

    ```
    git checkout [브랜치/태그명]
    ```

    예시:

    ```
    git checkout release/2.0-rc1
    ```

5. [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)를 참고하여 Python 의존 패키지를 설치합니다.

    ```
    pip install -r python/requirments.txt
    ```

6. `build`라는 이름의 디렉터리를 생성하고 진입합니다.

    ```
    mkdir build && cd build
    ```

7. 컴파일 시 열리는 파일 수가 많아 시스템 기본 제한을 초과할 수 있으므로, 최대 파일 수를 늘려줍니다.

    ```
    ulimit -n 4096
    ```

8. cmake 실행:

    > 각 컴파일 옵션에 대한 설명은 [컴파일 옵션표](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)를 참고하세요.

    ```
    # For Python2:
    cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MKL=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON
    # For Python3:
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON
    ```

9. 컴파일:

    ```
    make -j$(nproc)
    ```

10. 컴파일이 성공하면 `Paddle/build/python/dist` 디렉터리로 이동하여 생성된 `.whl` 파일을 찾습니다.

11. 현재 또는 대상 머신에 `.whl` 패키지를 설치합니다.

    ```
    python2 -m pip install -U (whl 파일 이름) 또는 python3 -m pip install -U (whl 파일 이름)
    ```

축하합니다! 이제 FT 환경에서 PaddlePaddle 컴파일 및 설치가 완료되었습니다.

## **설치 확인**
설치가 완료되면 `python` 또는 `python3`을 실행한 후 아래 명령어를 입력하세요:

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```
Your Paddle Fluid is installed succesfully!라는 메시지가 나타나면 설치가 성공한 것입니다.

mobilenetv1 및 resnet50 모델 테스트:

    wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
    tar xf profile.tar && cd profile
    python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
    # 正确输出应为：[0.0002414  0.00022418 0.00053661 0.00028639 0.00072682 0.000213
    #              0.00638718 0.00128127 0.00013535 0.0007676 ]
    python mobilenetv1.py --model_file mobilenetv1/model --params_file mobilenetv1/params
    # 正确输出应为：[0.00123949 0.00100392 0.00109539 0.00112206 0.00101901 0.00088412
    #              0.00121536 0.00107679 0.00106071 0.00099605]
    python ernie.py --model_dir ernieL3H128_model/
    # 正确输出应为：[0.49879393 0.5012061 ]

## **제거 방법**

다음 명령어로 PaddlePaddle을 제거할 수 있습니다:
```
python3 -m pip uninstall paddlepaddle` 或 `python3 -m pip uninstall paddlepaddle
```

## **비고**

ZHAOXIN 아키텍처에서 resnet50, mobilenetv1, ernie, ELMo 등의 모델을 테스트한 결과, 예측에 사용되는 연산자의 정확성은 기본적으로 확인되었습니다.
사용 중 계산 오류나 컴파일 실패 등이 발생하면 [issue](https://github.com/PaddlePaddle/Paddle/issues)에 남겨주세요. 최대한 신속히 대응하겠습니다.

예측 문서[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)

사용 예제[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
