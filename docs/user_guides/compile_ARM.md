# **Feiteng/Kunpeng 환경에서 소스코드 컴파일**

## 검증된 모델 목록

- resnet50
- mobilenetv1
- ernie
- ELMo

## 환경 준비

* **프로세서: FT2000+/Kunpeng 920 2426SK**
* **운영체제: Kylin v10/UOS**
* **Python 버전: 2.7.15+/3.5.1+/3.6/3.7/3.8 (64비트)**
* **pip 또는 pip3 버전: 9.0.1+ (64비트)**

Feiteng FT2000+와 Kunpeng 920 프로세서는 모두 ARMV8 아키텍처로, 해당 아키텍처에서의 Paddle 컴파일 방식은 동일합니다. 본 문서는 FT2000+를 예시로 하여 Paddle의 소스코드 컴파일을 소개합니다.

## 설치 단계

현재 FT2000+ 프로세서 및 국산 운영체제(Kylin UOS)에서 Paddle을 설치하는 방법은 소스코드 컴파일 방식만 지원되며, 다음은 각 단계에 대한 상세 설명입니다.

### **소스코드 컴파일**

1. Paddle은 cmake를 사용하여 컴파일하며, cmake 버전은 3.10 이상이 필요합니다. 운영체제가 적절한 버전의 cmake를 제공하는 경우 직접 설치 가능하며, 그렇지 않은 경우 [소스 설치](https://github.com/Kitware/CMake)를 참고하십시오.

2. Paddle 내부에서는 patchelf를 사용하여 동적 라이브러리의 rpath를 수정합니다. 운영체제에서 patchelf를 제공하는 경우 직접 설치 가능하며, 그렇지 않으면 [patchelf 공식 문서](https://github.com/NixOS/patchelf)를 참고하십시오. ARM 환경에서의 종속성 제거도 고려 중입니다.

3. [requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)에 따라 Python 의존 라이브러리를 설치합니다. Feiteng 및 국산 운영체제 환경에서는 pip 설치가 실패하거나 제대로 작동하지 않을 수 있어, 소스 또는 시스템 패키지 설치 방식을 권장합니다.

4. Paddle 소스코드를 현재 디렉토리의 Paddle 폴더에 클론하고 해당 디렉토리로 이동합니다.

5. 안정적인 release 브랜치로 전환합니다. 예: `git checkout release/2.0-rc1`

6. build 디렉토리를 만들고 이동합니다.

7. 파일 열기 제한으로 인해 컴파일 에러가 발생할 수 있어 최대 파일 열기 수를 설정합니다: `ulimit -n 4096`

8. cmake 실행 (자세한 컴파일 옵션은 [컴파일 옵션표](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile) 참고)

9. 컴파일 명령 실행 시 `TARGET=ARMV8`을 반드시 추가해야 합니다.

10. 컴파일 성공 후 `Paddle/build/python/dist` 디렉토리에서 `.whl` 패키지를 확인합니다.

11. 현재 또는 대상 머신에 `.whl` 패키지를 설치합니다.

## **설치 확인**
`python` 또는 `python3`에서 `import paddle.fluid as fluid` 입력 후 `fluid.install_check.run_check()`를 실행하여 `Your Paddle Fluid is installed succesfully!` 메시지가 출력되면 설치가 완료된 것입니다.

resnet50, mobilenetv1 모델을 테스트할 수 있습니다.

## **제거 방법**
PaddlePaddle 제거는 다음 명령을 사용합니다:

```
pip uninstall paddlepaddle 또는 pip3 uninstall paddlepaddle
```

## **비고**
ARM 아키텍처에서 resnet50, mobilenetv1, ernie, ELMo 모델의 테스트를 완료하였으며, 예측에 필요한 연산자의 정확성을 기본적으로 보장합니다. 사용 중 오류 발생 시 [issue](https://github.com/PaddlePaddle/Paddle/issues)에 문의 바랍니다.

예측 관련 문서는 [문서](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)를, 사용 예시는 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)를 참고하십시오.
