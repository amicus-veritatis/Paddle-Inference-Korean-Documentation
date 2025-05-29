# 훈련 및 추론 예제 설명

이 문서는 **Paddle 2.0의 새로운 인터페이스**를 사용해 모델을 훈련하고 추론하는 방법을 소개합니다.

## 1. Paddle 2.0 인터페이스로 간단한 모델 훈련

[LeNet을 이용한 MNIST 이미지 분류 예제](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/cv_case/image_classification/image_classification.html#lenetmnist)를 참고하여, Paddle 2.0 인터페이스로 간단한 모델을 훈련하고, 훈련 모델을 저장합니다. 특히, 모델 파일을 생성하는 과정을 강조합니다.

### - 필수 패키지 임포트

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor
```

- Paddle 버전 확인

```
print(paddle.__version__)
```

- 데이터셋 준비

```
train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
```

- LeNet 네트워크 구성

```
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
```

- 모델 훈련

```
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
model = LeNet()
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
train(model, optim)
```

- 훈련 모델 저장 (훈련용 포맷): 매개변수 저장을 참고하여, 동적 그래프(dygraph) 모드에서 훈련 포맷의 모델을 저장하는 방법을 확인할 수 있습니다. paddle.save 인터페이스만 호출하면 됩니다.

```
paddle.save(model.state_dict(), 'lenet.pdparams')
paddle.save(optim.state_dict(), "lenet.pdopt")
```

## 2. 사전 훈련된 모델을 추론 배포용 모델로 변환하는 방법

- 사전 훈련 모델 불러오기: 매개변수 불러오기를 참고하면, 동적 그래프(dygraph) 모드에서 훈련 포맷의 모델을 로드하는 방법을 알 수 있습니다. 이 방법은 훈련 중단 시점으로 모델 상태를 복원하는 데 유용하며, 복원 후에도 훈련의 경로(gradient flow)가 그대로 이어집니다.
paddle.load 인터페이스로 훈련 포맷 모델을 불러온 뒤, set_state_dict를 호출하여 훈련 중단 당시의 상태를 복원하면 됩니다.

```
model_state_dict = paddle.load('lenet.pdparams')
opt_state_dict = paddle.load('lenet.pdopt')
model.set_state_dict(model_state_dict)
optim.set_state_dict(opt_state_dict)
```

- 추론 배포용 모델로 저장하기: 실제 서비스에 배포할 때는 추론용 포맷의 모델을 사용하는 것이 필요합니다. 이 추론 포맷의 모델은 훈련 포맷과 비교했을 때 그래프 구조가 간소화되어 있으며, 예측에 필요 없는 연산자(OP)들이 제거되어 있습니다.
InputSpec 문서를 참고하여 동적 그래프(DyGraph)를 정적 그래프(Static Graph)로 변환할 수 있습니다.
InputSpec으로 모델 입력을 지정한 후, paddle.jit.to_static과 paddle.jit.save를 호출하면 추론용 모델을 저장할 수 있습니다.

```
net = to_static(model, input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
paddle.jit.save(net, 'inference_model/lenet')
```

### 참고 코드

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor


class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1,
                                      out_channels=6,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16 * 5 * 5,
                                        out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # x = x.reshape((-1, 1, 28, 28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # calc loss
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                    epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()


if __name__ == '__main__':
    # paddle version
    print(paddle.__version__)

    # prepare datasets
    train_dataset = MNIST(mode='train', transform=ToTensor())
    test_dataset = MNIST(mode='test', transform=ToTensor())

    # load dataset
    train_loader = paddle.io.DataLoader(train_dataset,
                                        batch_size=64,
                                        shuffle=True)

    # build network
    model = LeNet()
    # prepare optimizer
    optim = paddle.optimizer.Adam(learning_rate=0.001,
                                  parameters=model.parameters())

    # train network
    train(model, optim)

    # save training format model
    paddle.save(model.state_dict(), 'lenet.pdparams')
    paddle.save(optim.state_dict(), "lenet.pdopt")

    # load training format model
    model_state_dict = paddle.load('lenet.pdparams')
    opt_state_dict = paddle.load('lenet.pdopt')
    model.set_state_dict(model_state_dict)
    optim.set_state_dict(opt_state_dict)

    # save inferencing format model
    net = to_static(model,
                    input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
    paddle.jit.save(net, 'inference_model/lenet')
```

Paddle 2.0은 기본적으로 *.pdiparams 확장자를 가진 통합된 형식의 가중치 파일로 모델을 저장합니다.
하지만 특별한 요구 사항이 있어 이전 버전의 분리된 가중치 저장 방식을 사용하고자 한다면, 아래 예제를 참고하여 따로 저장할 수 있습니다.
Paddle 2.0은 이러한 구형 분리 포맷의 추론 모델도 호환하여 불러올 수 있습니다.

```
import paddle

if __name__ == '__main__':
    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    # load combined params and model
    program, _, _ = paddle.static.load_inference_model(
        path_prefix='inference_model',
        executor=exe,
        model_filename='lenet.pdmodel',
        params_filename='lenet.pdiparams')

    # save as separate persistables
    paddle.static.save_vars(
        executor=exe,
        dirname="separate_persistables",
        main_program=program,
        vars=None,
        predicate=paddle.static.io.is_persistable)
```


## 3. Paddle 2.0 Python 인터페이스를 이용한 추론 배포

저장된 추론용 모델을 사용하여, Python 2.0 인터페이스를 통해 추론을 수행합니다.

### 추론 모델 로드 및 설정 구성

먼저, 추론 모델을 로드하고, 예측 시 사용할 몇 가지 설정을 구성합니다. 그런 다음 설정에 따라 추론 엔진을 생성합니다:

```python
config = Config("inference_model/lenet/lenet.pdmodel", "inference_model/lenet/lenet.pdiparams")  # 모델 및 파라미터 경로로 로드
config.disable_gpu()  # CPU로 추론 수행
predictor = create_predictor(config)  # 설정에 따라 예측 엔진 생성
```
더 많은 설정 옵션은 공식 문서를 참고하세요.

### 입력 설정

입력 Tensor의 이름을 가져오고, 해당 이름을 이용해 입력 핸들을 획득합니다.

```python
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])
```

다음으로 입력 데이터를 준비하고, 예측 장치(CPU)에 복사합니다. 여기서는 임의 데이터를 사용하지만, 실제 사용 시에는 실제 이미지 데이터를 사용할 수 있습니다.。

```python
fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
input_handle.reshape([1, 1, 28, 28])
input_handle.copy_from_cpu(fake_input)
```

### 추론 실행

```python
predictor.run()
```

### 출력 획득

```python
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()
```
입출력 핸들 설정 방식은 동일하며, 출력 결과는 numpy.ndarray 형식으로 획득되어 numpy를 사용해 후속 처리가 가능합니다.

### 전체 실행 가능한 코드
```python
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

# 모델 및 파라미터 경로 설정
config = Config("inference_model/lenet/lenet.pdmodel", "inference_model/lenet/lenet.pdiparams")
config.disable_gpu()  # GPU 비활성화, CPU 사용

# Paddle 예측기 생성
predictor = create_predictor(config)

# 입력 이름 가져오기
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 입력 설정 (임의의 테스트 데이터 사용)
fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
input_handle.reshape([1, 1, 28, 28])
input_handle.copy_from_cpu(fake_input)

# 예측 실행
predictor.run()

# 출력 이름 가져오기 및 출력값 복사
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()  # numpy.ndarray 타입

print(output_data)

```

## 4. Paddle 2.0 C++ 인터페이스를 이용한 추론 배포

저장해둔 모델을 Paddle 2.0의 C++ 인터페이스를 통해 추론 환경에 배포하고 실행하는 방법을 설명합니다.

### 추론 라이브러리 준비

먼저, 모델 추론을 위한 Paddle Inference 라이브러리를 다운로드해야 합니다.  
다음 명령어를 사용해 x86 CPU용 Paddle Inference 2.0.0 버전을 다운로드합니다:

```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.0.0-cpu-avx-mkl/paddle_inference.tgz
```

### 추론 예제 코드 다운로드
예제 코드를 포함한 패키지를 다운로드합니다:

```shell
wget https://paddle-inference-dist.bj.bcebos.com/lenet_demo.tgz
```

압축을 풀면 다음과 같은 파일이 포함되어 있습니다: lenet_infer_test.cc: 추론 실행 C++ 코드, run.sh: 빌드 및 실행 스크립트
 
### 의존 라이브러리 경로 설정

run.sh 스크립트에서 추론 라이브러리의 경로를 지정해줘야 합니다. 예:

```shell
LIB_DIR=/path/to/paddle_inference
```

### 모델 로드 및 예측 엔진 설정

먼저 모델을 로드하고, 예측에 필요한 설정을 구성합니다. 아래는 C++ 코드 예시입니다:

```c++
std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.DisableGpu();
  return CreatePredictor(config);
}
```

더 많은 설정 항목은 공식 문서를 참고하세요.

### 입력 설정

입력 텐서의 이름을 가져오고 해당 입력 핸들을 이용해 데이터를 전달합니다:

```c++
std::vector<float> input(1 * 1 * 28 * 28, 0);  // 입력 데이터를 전부 0으로 초기화
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
input_t->Reshape({1, 1, 28, 28});  // 입력 shape 설정
input_t->CopyFromCpu(input.data());  // CPU 데이터 복사

```

여기에서는 모든 값이 0인 데이터를 사용했지만, 실제 사용 시에는 예측하고자 하는 실제 이미지 데이터로 대체할 수 있습니다.

### 예측 실행

```c++
predictor->Run();
```

### 출력 가져오기

```c++
# 출력 변수 이름 가져오기
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
```
out_data바로 원하는 출력 결과이며, 이후 분석 및 처리에 활용할 수 있습니다.

### 추론 실행

구성이 완료되면, 추론 예제 파일이 있는 디렉터리에서 아래 명령어를 사용하여 예제를 컴파일하고 실행합니다.

```shell
sh run.sh
./build/lenet_infer_test --model_file=inference_model/lenet/lenet.pdmodel --params_file=inference_model/lenet/lenet.pdiparams
```
