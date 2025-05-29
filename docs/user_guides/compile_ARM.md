# **소스 코드로부터 페이텅/쿤펑 컴파일**

## 검증된 모델 목록

- resnet50  
- mobilenetv1  
- ernie  
- ELMo  

## 환경 준비

* **프로세서: FT2000+/Kunpeng 920 2426SK**
* **운영 체제: Kylin v10 / UOS**
* **Python 버전: 2.7.15+/3.5.1+/3.6/3.7/3.8 (64비트)**
* **pip 또는 pip3 버전: 9.0.1+ (64비트)**

FT2000+와 Kunpeng920 프로세서는 모두 ARMV8 아키텍처이며, 이 아키텍처에서의 Paddle 소스 컴파일 방식은 동일합니다. 본 문서에서는 FT2000+를 예로 들어 Paddle의 소스 컴파일 과정을 소개합니다.

## 설치 단계

현재 FT2000+ 프로세서와 국산 운영 체제(예: Kylin UOS) 환경에서는 Paddle을 소스 컴파일 방식으로만 설치할 수 있습니다. 다음은 자세한 단계입니다.

### **소스 컴파일**

1. Paddle은 cmake를 통해 빌드됩니다. cmake 버전은 >=3.10 이어야 하며, 운영 체제의 패키지 관리자가 해당 버전을 제공한다면 그대로 설치하면 됩니다. 그렇지 않으면 [소스 코드로 설치](https://github.com/Kitware/CMake)해야 합니다.

    ```bash
    wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
    tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
    ./bootstrap && make && sudo make install
    ```

2. Paddle은 내부적으로 patchelf를 사용하여 동적 라이브러리의 rpath를 수정합니다. 시스템에서 patchelf를 제공한다면 설치하면 되고, 없다면 [공식 문서](https://github.com/NixOS/patchelf)를 참고하여 소스 설치하세요.

    ```bash
    ./bootstrap.sh
    ./configure
    make
    make check
    sudo make install
    ```

3. [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)에 따라 Python 의존 패키지를 설치하세요. FT2000+ 및 국산 운영 체제에서는 pip 설치가 실패할 수 있어, 시스템 패키지 관리자나 소스 코드로 직접 설치하는 방식을 추천합니다.

4. Paddle 소스 코드를 현재 디렉터리에 복제하고 해당 폴더로 이동합니다.

    ```bash
    git clone https://github.com/PaddlePaddle/Paddle.git
    cd Paddle
    ```

5. 안정적인 release 브랜치로 전환 후 컴파일하세요:

    ```bash
    git checkout [브랜치 이름]
    ```

    예시:

    ```bash
    git checkout release/2.0-rc1
    ```

6. `build` 디렉터리를 만들고 진입하세요:

    ```bash
    mkdir build && cd build
    ```

7. 파일 연결 수 제한으로 인해 컴파일 중 오류가 발생할 수 있으므로, 열 수 있는 최대 파일 수를 증가시켜 주세요:

    ```bash
    ulimit -n 4096
    ```

8. cmake 실행:

    > 각 옵션의 의미는 [컴파일 옵션 표](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)를 참고하세요.

    Python2 용:

    ```bash
    cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF
    ```

    Python3 용:

    ```bash
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF
    ```

9. 다음 명령어로 컴파일합니다:

    ```bash
    make TARGET=ARMV8 -j$(nproc)
    ```

10. 컴파일 성공 후, `Paddle/build/python/dist` 디렉터리에서 `.whl` 패키지를 확인합니다.

11. 현재 시스템 또는 대상 시스템에 `.whl` 패키지를 설치합니다:

    ```bash
    pip install -U (whl 파일 이름) 또는 pip3 install -U (whl 파일 이름)
    ```

축하합니다! FT 환경에서 PaddlePaddle의 컴파일 및 설치가 완료되었습니다.

## **설치 확인**

설치 후 `python` 또는 `python3`으로 Python 인터프리터에 진입한 뒤, 아래 명령어를 입력합니다:

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```
Your Paddle Fluid is installed succesfully!라는 메시지가 뜨면 설치가 성공한 것입니다.

mobilenetv1 및 resnet50 모델에서 테스트:
```
wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
tar xf profile.tar && cd profile

python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
# 출력 예시:
# [0.0002414  0.00022418 0.00053661 0.00028639 0.00072682 0.000213
#  0.00638718 0.00128127 0.00013535 0.0007676 ]

python mobilenetv1.py --model_file mobilenetv1/model --params_file mobilenetv1/params
# 출력 예시:
# [0.00123949 0.00100392 0.00109539 0.00112206 0.00101901 0.00088412
#  0.00121536 0.00107679 0.00106071 0.00099605]

python ernie.py --model_dir ernieL3H128_model/
# 출력 예시:
# [0.49879393 0.5012061 ]
```
## **제거 방법**

다음 명령어를 사용하여 PaddlePaddle을 삭제할 수 있습니다:
```
pip uninstall paddlepaddle 또는 pip3 uninstall paddlepaddle
```
## **비고**

ARM 아키텍처 환경에서 resnet50, mobilenetv1, ernie, ELMo 등의 모델을 테스트하였으며, 예측을 위한 연산자들의 정확성은 기본적으로 검증되었습니다.  
사용 중 계산 결과 오류나 컴파일 실패 등의 문제가 발생하면, [issue 페이지](https://github.com/PaddlePaddle/Paddle/issues)에 남겨주세요. 최대한 빠르게 지원해드리겠습니다.

- 예측 관련 문서는 [여기에서 확인](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)할 수 있습니다.  
- 예제 사용법은 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)를 참고하세요.
- 
