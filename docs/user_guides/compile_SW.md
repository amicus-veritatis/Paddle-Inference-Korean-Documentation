# **申威下从源码编译**

## 검증된 모델 목록

- resnet50
- mobilenetv1
- ernie
- ELMo

## 환경 준비

* **프로세서：SW6A**
* **운영 체제：Phytium iSoft Linux 5**
* **Python 버전：2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip 또는 pip3 버전：9.0.1+ (64 bit)**

신웨이 머신은 SW 아키텍처 기반으로, 현재 지원되는 생태계 소프트웨어가 제한적입니다. 이 문서에서는 다소 복잡한 방법을 통해 신웨이 머신에서 Paddle을 소스 컴파일하는 방법을 설명합니다. 향후 신웨이 생태계가 발전함에 따라 이 문서도 계속 업데이트될 예정입니다.

## 설치 단계

이 문서는 신웨이 프로세서 환경에서 Paddle을 설치하는 전 과정을 안내합니다.

<a name="sw_source"></a>
### **소스 코드 컴파일**

1. Paddle의 소스 코드를 현재 디렉터리의 `Paddle` 폴더에 클론한 후, 해당 디렉터리로 진입합니다.

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    cd Paddle
    ```

2. 비교적 안정적인 release 브랜치로 전환하여 컴파일을 진행합니다.

    ```
    git checkout [브랜치/태그명]
    ```

    예시:

    ```
    git checkout release/2.0-rc1
    ```

3. Paddle은 cmake를 통해 컴파일됩니다. cmake 버전은 최소 3.10 이상이 필요하며, OS에서 제공하는 cmake 버전을 확인해 주세요. `apt install cmake`로 설치하고, `cmake --version`으로 확인하세요. 만약 cmake가 3.10 미만이라면, `CMakeLists.txt`의 `cmake_minimum_required(VERSION 3.10)`을 `cmake_minimum_required(VERSION 3.0)`으로 수정해야 합니다.

4. 신웨이 환경은 아직 openblas를 지원하지 않기 때문에, 대신 blas + cblas 조합을 사용해야 하며, 이 둘을 소스에서 컴파일해야 합니다.

    ```
    pushd /opt
    wget http://www.netlib.org/blas/blas-3.8.0.tgz
    wget http://www.netlib.org/blas/blast-forum/cblas.tgz
    tar xzf blas-3.8.0.tgz
    tar xzf cblas.tgz
    pushd BLAS-3.8.0
    make
    popd
    pushd CBLAS
    # Makefile.in에서 BLLIB 항목을 BLAS-3.8.0에서 컴파일된 blas_LINUX.a로 수정
    make
    pushd lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
    ln -s cblas_LINUX.a libcblas.a
    cp ../../BLAS-3.8.0/blas_LINUX.a .
    ln -s blas_LINUX.a libblas.a
    popd
    popd
    popd
    ```

5. [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)를 참고하여 Python 의존성을 설치합니다.  
   주의: 신웨이 시스템에서는 pip 또는 소스 빌드 방식으로 의존 패키지를 설치하는 것이 거의 불가능합니다. 따라서 OS의 패키지 관리자(yum, apt 등)를 통해 설치하는 것을 권장합니다. 만약 일부 패키지가 설치되지 않는 경우 OS 공급업체에 문의하거나, `pip install --no-deps` 방식으로 의존성 설치를 건너뛸 수 있으나 이 경우 일부 기능이 동작하지 않을 수 있습니다.

6. `build` 디렉터리를 만들고 진입합니다.

    ```
    mkdir build && cd build
    ```

7. 빌드 도중 너무 많은 파일을 열게 되어 오류가 날 수 있으므로, 파일 오픈 수 제한을 늘립니다.

    ```
    ulimit -n 4096
    ```

8. cmake 실행:

    > 각 옵션의 의미는 [컴파일 옵션 표](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile) 참고

    ```
    CBLAS_ROOT=/opt/CBLAS
    # For Python2:
    cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MKL=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON -DREFERENCE_CBLAS_ROOT=${CBLAS_ROOT} -DWITH_CRYPTO=OFF -DWITH_XBYAK=OFF -DWITH_SW=ON -DCMAKE_CXX_FLAGS="-Wno-error -w"
    # For Python3:
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON -DREFERENCE_CBLAS_ROOT=${CBLAS_ROOT} -DWITH_CRYPTO=OFF -DWITH_XBYAK=OFF -DWITH_SW=ON -DCMAKE_CXX_FLAGS="-Wno-error -w"
    ```

9. 컴파일:

    ```
    make -j$(nproc)
    ```

10. 컴파일이 완료되면 `Paddle/build/python/dist` 경로에 `.whl` 패키지가 생성됩니다.

11. 생성된 `.whl` 패키지를 설치합니다.

    ```
    python2 -m pip install -U (whl 파일 이름) 또는 python3 -m pip install -U (whl 파일 이름)
    ```

축하합니다! 이제 신웨이 환경에서 PaddlePaddle 소스 컴파일 및 설치가 완료되었습니다.

## **설치 확인**
설치가 완료되면 `python` 또는 `python3`으로 Python 인터프리터에 진입한 후 다음 명령을 입력하세요:

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```
Your Paddle Fluid is installed succesfully!라는 메시지가 출력되면 설치가 성공한 것입니다.

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

```
python3 -m pip uninstall paddlepaddle
# 또는
python2 -m pip uninstall paddlepaddle
```
## **비고**

신웨이 아키텍처에서 resnet50, mobilenetv1, ernie, ELMo 등의 모델을 테스트하였으며, 예측에 사용되는 연산자의 정확성은 기본적으로 검증되었습니다.
다만 부동 소수점 예외(Floating Point Exception) 문제가 발생할 수 있으며, 이 문제는 추후 신웨이 개발팀과 함께 해결할 예정입니다.
사용 중 컴파일 오류나 계산 오류가 발생하면 [issue](https://github.com/PaddlePaddle/Paddle/issues)에 남겨 주세요. 최대한 빠르게 대응하겠습니다.

예측 문서[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)

사용 예제[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
