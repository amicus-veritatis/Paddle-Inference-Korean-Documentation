# **龙芯下从源码编译**

## 검증된 모델 목록

- resnet50  
- mobilenetv1  
- ernie  
- ELMo  

## 환경 준비

* **프로세서: Loongson-3A R4 (Loongson-3A4000)**
* **운영 체제: Loongnix release 1.0**
* **Python 버전: 2.7.15+/3.5.1+/3.6/3.7/3.8 (64비트)**
* **pip 또는 pip3 버전: 20.2.2+ (64비트)**

이 문서는 Loongson-3A4000을 예시로 하여 MIPS 아키텍처 환경에서 Paddle을 소스 코드로 컴파일하는 방법을 설명합니다.

---

## 설치 단계

현재 MIPS 기반의 Loongson 프로세서와 국산 운영체제 환경에서는 Paddle을 소스 코드로만 설치할 수 있습니다. 아래는 상세한 설치 방법입니다.

### **소스 컴파일**

1. Loongnix 1.0에는 기본적으로 gcc 4.9가 설치되어 있으나, yum 소스에서 gcc-7 도구 체인을 설치할 수 있습니다:

    ```bash
    sudo yum install devtoolset-7-gcc.mips64el devtoolset-7-gcc-c++.mips64el devtoolset-7.mips64el
    source /opt/rh/devtoolset-7/enable
    ```

2. Python도 gcc 7.3에 맞춰 소스 설치가 필요합니다 (Python 3.7 예시):

    ```bash
    sudo yum install libffi-devel.mips64el openssl-devel.mips64el libsqlite3x-devel.mips64el sqlite-devel.mips64el \
    lbzip2-utils.mips64el lzma.mips64el tk.mips64el uuid.mips64el gdbm-devel.mips64el gdbm.mips64el \
    openjpeg-devel.mips64el zlib-devel.mips64el libjpeg-turbo-devel.mips64el openjpeg-devel.mips64el
    ```

    ```bash
    wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
    tar xzf Python-3.7.5.tgz && cd Python-3.7.5
    ./configure --prefix=$HOME/python37 --enable-shared
    make -j
    make install
    ```

    환경 변수 설정:

    ```bash
    export PATH=$HOME/python37/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/python37/lib:$LD_LIBRARY_PATH
    ```

3. CMake는 기본 버전이 3.9이므로, `CMakeLists.txt`에서 다음 줄을 수정하세요:

    ```cmake
    cmake_minimum_required(VERSION 3.10)
    ```
    를
    ```cmake
    cmake_minimum_required(VERSION 3.9)
    ```
    로 변경

4. patchelf 설치:

    ```bash
    sudo yum install patchelf.mips64el
    ```

5. Paddle 소스 코드 클론:

    ```bash
    git clone https://github.com/PaddlePaddle/Paddle.git
    cd Paddle
    ```

6. Python 의존성 설치:

    [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)를 참고해 수동 설치

7. develop 브랜치로 전환:

    ```bash
    git checkout develop
    ```

8. build 디렉터리 생성 및 진입:

    ```bash
    mkdir build && cd build
    ```

9. 파일 열기 제한 설정:

    ```bash
    ulimit -n 4096
    ```

10. CMake 실행 (Python3 기준):

    ```bash
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MIPS=ON -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_MKL=OFF
    ```

    > Python2 사용 시 PY_VERSION=2로 설정

11. 컴파일:

    ```bash
    make -j$(nproc)
    ```

12. `.whl` 패키지 확인:

    ```bash
    cd ../python/dist
    ```

13. 패키지 설치:

    ```bash
    python -m pip install -U <whl파일이름>
    # 또는
    python3 -m pip install -U <whl파일이름>
    ```

---

## **설치 확인**

```python
import paddle
paddle.utils.run_check()
# **龙芯下从源码编译**

## 검증된 모델 목록

- resnet50  
- mobilenetv1  
- ernie  
- ELMo  

## 환경 준비

* **프로세서: Loongson-3A R4 (Loongson-3A4000)**
* **운영 체제: Loongnix release 1.0**
* **Python 버전: 2.7.15+/3.5.1+/3.6/3.7/3.8 (64비트)**
* **pip 또는 pip3 버전: 20.2.2+ (64비트)**

이 문서는 Loongson-3A4000을 예시로 하여 MIPS 아키텍처 환경에서 Paddle을 소스 코드로 컴파일하는 방법을 설명합니다.

---

## 설치 단계

현재 MIPS 기반의 Loongson 프로세서와 국산 운영체제 환경에서는 Paddle을 소스 코드로만 설치할 수 있습니다. 아래는 상세한 설치 방법입니다.

### **소스 컴파일**

1. Loongnix 1.0에는 기본적으로 gcc 4.9가 설치되어 있으나, yum 소스에서 gcc-7 도구 체인을 설치할 수 있습니다:

    ```bash
    sudo yum install devtoolset-7-gcc.mips64el devtoolset-7-gcc-c++.mips64el devtoolset-7.mips64el
    source /opt/rh/devtoolset-7/enable
    ```

2. Python도 gcc 7.3에 맞춰 소스 설치가 필요합니다 (Python 3.7 예시):

    ```bash
    sudo yum install libffi-devel.mips64el openssl-devel.mips64el libsqlite3x-devel.mips64el sqlite-devel.mips64el \
    lbzip2-utils.mips64el lzma.mips64el tk.mips64el uuid.mips64el gdbm-devel.mips64el gdbm.mips64el \
    openjpeg-devel.mips64el zlib-devel.mips64el libjpeg-turbo-devel.mips64el openjpeg-devel.mips64el
    ```

    ```bash
    wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
    tar xzf Python-3.7.5.tgz && cd Python-3.7.5
    ./configure --prefix=$HOME/python37 --enable-shared
    make -j
    make install
    ```

    환경 변수 설정:

    ```bash
    export PATH=$HOME/python37/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/python37/lib:$LD_LIBRARY_PATH
    ```

3. CMake는 기본 버전이 3.9이므로, `CMakeLists.txt`에서 다음 줄을 수정하세요:

    ```cmake
    cmake_minimum_required(VERSION 3.10)
    ```
    를
    ```cmake
    cmake_minimum_required(VERSION 3.9)
    ```
    로 변경

4. patchelf 설치:

    ```bash
    sudo yum install patchelf.mips64el
    ```

5. Paddle 소스 코드 클론:

    ```bash
    git clone https://github.com/PaddlePaddle/Paddle.git
    cd Paddle
    ```

6. Python 의존성 설치:

    [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)를 참고해 수동 설치

7. develop 브랜치로 전환:

    ```bash
    git checkout develop
    ```

8. build 디렉터리 생성 및 진입:

    ```bash
    mkdir build && cd build
    ```

9. 파일 열기 제한 설정:

    ```bash
    ulimit -n 4096
    ```

10. CMake 실행 (Python3 기준):

    ```bash
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MIPS=ON -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_MKL=OFF
    ```

    > Python2 사용 시 PY_VERSION=2로 설정

11. 컴파일:

    ```bash
    make -j$(nproc)
    ```

12. `.whl` 패키지 확인:

    ```bash
    cd ../python/dist
    ```

13. 패키지 설치:

    ```bash
    python -m pip install -U <whl파일이름>
    # 또는
    python3 -m pip install -U <whl파일이름>
    ```

---

## **설치 확인**

```python
import paddle
paddle.utils.run_check()
