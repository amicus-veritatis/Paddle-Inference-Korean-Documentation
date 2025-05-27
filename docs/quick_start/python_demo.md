# 예측 예제 (Python)

본 장은 두 부분으로 구성됩니다:  
(1) [Python 예제 프로그램 실행하기](#id1)  
(2) [Python 예측 프로그램 개발 안내](#id6)

## Python 예제 프로그램 실행하기

### 1. Python 예측 라이브러리 설치

[Paddle 공식 홈페이지 - 빠른 설치](https://www.paddlepaddle.org.cn/install/quick) 페이지를 참고하여 직접 설치 또는 컴파일을 진행하세요. 현재 pip/conda 설치, 도커 이미지, 소스 컴파일 등 다양한 방법으로 Paddle Inference 개발 환경을 준비할 수 있습니다.

### 2. 예측 배포 모델 준비

[ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 모델을 다운로드한 뒤 압축을 해제하면, Paddle 예측 포맷 모델이 `resnet50` 폴더 내에 생성됩니다. 모델 구조를 확인하려면 `inference.pdmodel` 파일을 모델 시각화 도구 Netron으로 열면 됩니다.

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 모델 디렉터리 및 파일 목록
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 예측 배포 프로그램 준비

다음 코드를 `python_demo.py` 파일로 저장하세요:

```python
import argparse
import numpy as np

# paddle inference 예측 라이브러리 임포트
import paddle.inference as paddle_infer

def main():
    args = parse_args()

    # Config 객체 생성
    config = paddle_infer.Config(args.model_file, args.params_file)

    # Config를 기반으로 predictor 생성
    predictor = paddle_infer.create_predictor(config)

    # 입력 이름 가져오기
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 입력 데이터 설정
    fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 3, 318, 318])
    input_handle.copy_from_cpu(fake_input)

    # 예측 실행
    predictor.run()

    # 출력 가져오기
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()  # numpy.ndarray 타입
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="모델 파일명")
    parser.add_argument("--params_file", type=str, help="파라미터 파일명")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    return parser.parse_args()

if __name__ == "__main__":
    main()
```

### 4. 예측 프로그램 실행

```bash
# 인자는 본 장 2단계에서 다운로드한 ResNet50 모델을 사용
python python_demo.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --batch_size 2
```

성공적으로 실행된 후, 예측 출력 결과는 다음과 같습니다:

```bash
# 프로그램 출력 결과 예시
Output data size is 2000
Output data shape is (2, 1000)
```

## Python 예측 프로그램 개발 안내

Paddle Inference를 사용하여 Python 예측 프로그램을 개발하려면 다음 다섯 단계만 따르면 됩니다.

1. paddle inference 예측 라이브러리 import

```python
import paddle.inference as paddle_infer
```

2. Config 객체 생성 및 필요에 따라 설정  
자세한 내용은 [Python API 문서 - Config](../api_reference/python_api_doc/Config_index) 참고

```python
# Config 생성 및 예측 모델 경로 설정
config = paddle_infer.Config(args.model_file, args.params_file)
```

3. Config를 기반으로 예측 객체 생성  
자세한 내용은 [Python API 문서 - Predictor](../api_reference/python_api_doc/Predictor) 참고

```python
predictor = paddle_infer.create_predictor(config)
```

4. 모델 입력 Tensor 설정  
자세한 내용은 [Python API 문서 - Tensor](../api_reference/python_api_doc/Tensor) 참고

```python
# 입력 이름 가져오기
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 입력 데이터 설정
fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
input_handle.reshape([args.batch_size, 3, 318, 318])
input_handle.copy_from_cpu(fake_input)
```

5. 예측 실행  
자세한 내용은 [Python API 문서 - Predictor](../api_reference/python_api_doc/Predictor) 참고

```python
predictor.run()
```

6. 예측 결과 획득  
자세한 내용은 [Python API 문서 - Tensor](../api_reference/python_api_doc/Tensor) 참고

```python
# 출력 이름 가져오기
output_names = predictor.get_output_names()
# 출력 핸들 가져오기
output_handle = predictor.get_output_handle(output_names[0])
# 출력 데이터를 CPU로 복사 (numpy.ndarray 타입)
output_data = output_handle.copy_to_cpu()
```
