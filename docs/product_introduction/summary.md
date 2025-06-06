# 플라잉페이 추론 제품 소개

플라잉페이 생태계의 중요한 구성 요소로서, 플라잉페이는 딥러닝 모델 응용의 마지막 단계를 완벽히 지원하는 여러 추론 제품을 제공한다.

전체적으로 추론 제품은 다음과 같은 하위 제품들로 구성된다.

| 명칭             | 영어 명칭         | 적용 시나리오                    |
|------------------|------------------|----------------------------------|
| 플라잉페이 네이티브 추론 라이브러리 | Paddle Inference | 고성능 서버 및 클라우드 추론        |
| 플라잉페이 서비스화 추론 프레임워크 | Paddle Serving   | 자동 서비스, 모델 관리 등 고급 기능  |
| 플라잉페이 경량 추론 엔진          | Paddle Lite      | 모바일, 사물인터넷(IoT) 등           |
| 플라잉페이 프론트엔드 추론 엔진    | Paddle.js        | 브라우저 추론, 미니 프로그램 등       |

---

## 각 제품의 추론 생태계 내 관계

![](../images/inference_ecosystem.png)

## 단계별 흐름

### 1. 모델 준비
- **PaddlePaddle**
  - 개발 + 훈련 → 모델
- **Tensorflow / ONNX 등**
  - 모델 →  
- **X2Paddle**
  - 변환 →


### 2. 모델 최적화
- **PaddleSlim**  
  압축 / 양자화 / 증류


### 3. 추론 배포
- **Paddle Inference**  
  - 서버
- **Paddle Serving**  
  - 서비스화 배포
- **Paddle Lite**  
  - 모바일 단말 / 엣지 단말
- **Paddle.js**  
  - 웹 프론트엔드


## 하단
- 설치 및 환경 호환
---

## 사용자가 플라잉페이 추론 제품을 사용하는 워크플로우

1. 플라잉페이 추론 모델 획득 (방법 2가지)

    1. 플라잉페이로 직접 훈련하여 추론 모델 생성
    2. X2Paddle 도구를 사용해 TensorFlow, Caffe 등 타사 프레임워크에서 생성된 모델 변환

2. (선택 사항) PaddleSlim 도구를 이용해 모델 최적화 진행  
   - 모델 압축, 양자화, 가지치기 등 수행  
   - 실행 속도 향상 및 자원 소모 절감

3. 최종적으로 모델을 특정 추론 제품에 배포
