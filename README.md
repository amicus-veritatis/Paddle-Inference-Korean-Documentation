# Paddle Inference 데모

**Paddle Inference**는 PaddlePaddle 핵심 프레임워크의 추론 엔진입니다.
Paddle Inference는 풍부한 기능과 우수한 성능을 갖추고 있으며, 서버 측 응용 시나리오에 대해 깊이 있는 최적화가 이루어져 고처리량(High Throughput), 저지연(Low Latency)을 달성합니다. 이를 통해 Paddle 모델을 서버에서 바로 훈련 후 사용할 수 있으며, 빠르게 배포할 수 있습니다.

많은 사용자들이 Paddle Inference를 이용해 빠르게 애플리케이션을 배포할 수 있도록, 이 저장소(Repo)에서는 C++ 및 Python을 이용한 사용 예제들을 제공합니다.

**이 저장소에서는 사용자가 이미 Paddle Inference에 대해 일정 수준의 이해가 있다고 가정합니다.**

**Paddle Inference를 처음 접하셨다면, [이곳](https://www.paddlepaddle.org.cn/inference/master/guides/introduction/index_intro.html)을 방문하여 Paddle Inference에 대한 기초적인 이해를 먼저 하시는 것을 권장드립니다.**

---

## 테스트 예제

1. **python 디렉토리**에서는 실제 입력을 기반으로 다양한 테스트 예제를 나열하였습니다.
   여기에는 이미지 분류, 분할, 감지(Detection) 및 NLP 관련 Ernie/Bert 등 Python 예제들이 포함되어 있으며, Paddle-TRT 및 멀티스레드 사용 예제도 포함되어 있습니다.

2. **c++ 디렉토리**에서는 단위 테스트(Unit Test) 방식으로 다양한 테스트 예제를 제공하고 있습니다.
   여기에는 이미지 분류, 분할, 감지(Detection) 및 NLP 관련 Ernie/Bert 등 C++ 예제들이 포함되어 있으며, Paddle-TRT 및 멀티스레드 사용 예제도 포함되어 있습니다.

> ⚠️ 주의: 만약 Paddle 2.0 이전 버전을 사용하고 있다면, [release/1.8 브랜치](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/release/1.8)를 참고해주세요.
