# **龙芯下从源码编译**

## 검증된 모델 목록

- resnet50
- mobilenetv1
- ernie
- ELMo

## 환경 준비

* **프로세서：Loongson-3A R4 (Loongson-3A4000)**
* **운영 체제：Loongnix release 1.0**
* **Python 버전：2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip 또는 pip3 버전：20.2.2+ (64 bit)**

이 문서는 Loongson-3A4000을 예시로 하여 MIPS 아키텍처에서 Paddle을 소스 코드로 컴파일하는 방법을 소개합니다.

## 설치 단계

현재 MIPS 기반 Loongson 프로세서와 Loongson 국산 운영 체제에서 Paddle을 설치할 때는 소스 코드 컴파일 방식만 지원됩니다. 아래는 각 단계에 대한 자세한 설명입니다.

<a name="mips_source"></a>
### **소스 코드 컴파일**

1. Loongnix 1.0에는 기본적으로 gcc 4.9가 설치되어 있습니다. 하지만 yum 소스에서 gcc-7 도구 체인을 제공하므로, gcc-7을 설치합니다. 자세한 내용은 [Loongnix 커뮤니티 문서](http://www.loongnix.org/index.php/Gcc7.3.0)를 참고하세요.

    ```
    sudo yum install devtoolset-7-gcc.mips64el devtoolset-7-gcc-c++.mips64el devtoolset-7.mips64el
    ```

    gcc-7을 활성화하려면 환경변수를 설정하세요:

    ```
    source /opt/rh/devtoolset-7/enable
    ```

2. 시스템에 기본으로 설치된 Python은 gcc 4.9를 기반으로 하기 때문에, 위에서 설치한 gcc-7.3에 맞춰 Python도 소스로 설치해야 합니다. 여기서는 Python 3.7을 예시로 설명합니다.

    ```
    sudo yum install libffi-devel.mips64el openssl-devel.mips64el libsqlite3x-devel.mips64el sqlite-devel.mips64el lbzip2-utils.mips64el lzma.mips64el tk.mips64el uuid.mips64el gdbm-devel.mips64el gdbm.mips64el openjpeg-devel.mips64el zlib-devel.mips64el libjpeg-turbo-devel.mips64el openjpeg-devel.mips64el
    ```

    ```
    wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz && tar xzf Python-3.7.5.tgz && cd Python-3.7.5
    ```

    ```
    ./configure –prefix $HOME/python37–enable−shared
    ```

    ```
    make -j
    ```

    ```
    make install
    ```

    Python 3.7이 적용되도록 환경 변수를 설정합니다:

    ```
    export PATH=$HOME/python37/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/python37/lib:$LD_LIBRARY_PATH
    ```

3. Paddle은 cmake를 사용하여 컴파일되며, cmake 3.10 이상이 필요합니다. 하지만 현재 Loongnix에서 제공하는 cmake 버전은 3.9이며, 소스 컴파일이 실패할 수 있습니다. 임시 해결책으로는 `CMakeLists.txt`에서 `cmake_minimum_required(VERSION 3.10)`를 `cmake_minimum_required(VERSION 3.9)`로 수정하는 것입니다. 향후 cmake ≥ 3.10이 지원되면 수정은 필요하지 않습니다.

4. Paddle은 동적 라이브러리의 rpath 수정을 위해 patchelf를 사용합니다. 운영 체제의 패키지 소스에 patchelf가 포함되어 있으므로 직접 설치하면 됩니다. 향후 MIPS에서는 해당 의존을 제거할 예정입니다.

    ```
    sudo yum install patchelf.mips64el
    ```

5. Paddle의 소스 코드를 현재 디렉터리에 클론한 후 Paddle 디렉터리로 진입합니다:

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    cd Paddle
    ```

6. [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)를 참고하여 Python 의존 라이브러리를 설치합니다.

7. `develop` 브랜치로 전환:

    ```
    git checkout develop
    ```

8. `build` 디렉터리를 만들고 진입합니다:

    ```
    mkdir build && cd build
    ```

9. 파일 디스크립터 제한을 늘립니다:

    ```
    ulimit -n 4096
    ```

10. cmake 실행:

    > 각 컴파일 옵션에 대한 설명은 [컴파일 옵션 문서](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)를 참고하세요.

    Python2:

    ```
    cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MIPS=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_MKL=OFF
    ```

    Python3:

    ```
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MIPS=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_MKL=OFF
    ```

11. 컴파일 실행:

    ```
    make -j$(nproc)
    ```

12. 컴파일이 완료되면 `Paddle/build/python/dist` 경로에서 생성된 `.whl` 파일을 확인합니다.

13. 현재 또는 대상 머신에 `.whl` 패키지를 설치합니다:

    ```
    python -m pip install -U (whl 파일명)
    # 또는
    python3 -m pip install -U (whl 파일명)
    ```

---

## **설치 확인**

설치가 완료되면 `python` 또는 `python3`을 실행한 뒤 아래 코드를 입력하세요:

```python
import paddle
paddle.utils.run_check()
```
PaddlePaddle is installed successfully! 메시지가 출력되면 설치가 완료된 것입니다.
mobilenetv1 및 resnet50 테스트:
```
wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
tar xf profile.tar && cd profile
```
```
python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
# 예상 출력:
# [0.0002414  0.00022418 0.00053661 0.00028639 0.00072682 0.000213
#  0.00638718 0.00128127 0.00013535 0.0007676 ]
```
```
python ernie.py --model_dir ernieL3H128_model/
# 正确输出应为：[0.49879393 0.5012061 ]
```
## **제거 방법**

다음 명령어로 PaddlePaddle을 삭제할 수 있습니다:
```
python -m pip uninstall paddlepaddle
# 또는
python3 -m pip uninstall paddlepaddle
```
## **비고**

MIPS 아키텍처 환경에서 resnet50, mobilenetv1, ernie, ELMo 모델로 테스트를 완료하였으며, 예측 관련 연산자의 정확성은 기본적으로 보장됩니다.
사용 중 계산 오류나 빌드 실패 등의 문제가 발생할 경우, [issue](https://github.com/PaddlePaddle/Paddle/issues) 페이지에 남겨주세요. 최대한 빠르게 대응하겠습니다.

예측 문서 참조[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)

사용 예시[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
