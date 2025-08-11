# 축구 시뮬레이션 프로젝트

축구 전술 및 물리 시뮬레이션을 위한 파이썬 프로젝트입니다. 이 프로젝트는 선수 이동, 공 물리, 스태미너 시스템, 전술 로직 및 다양한 시각화 시나리오를 포함합니다.
본래 의도는 풋볼매니저와 유사하지만 전술 부분에 특화된 축구 전략 게임을 만드는데 필요한 엔진을 만드는 것이 목표였습니다.
+ 패스, 슛, 선수 이동 및 충돌, 스태미너까지 구현된 상태이고 현재 중단된 상태입니다.

## 주요 기능

  - **물리 엔진**: `player`, `ball`, `engine`을 중심으로 선수의 이동, 충돌, 속도 업데이트를 처리합니다.
  - **전술/스태미너**: `tactics` 및 `stamina_system`을 기반으로 선수의 행동과 체력 변화를 제어합니다.
  - **시뮬레이션 환경**: `simulation/environment.py`를 통해 시뮬레이션의 틱(tick) 및 프레임(frame) 업데이트를 관리합니다.
  - **시각화**: 패스, 슛, 마킹, 오버랩 등 다양한 상황을 시각적으로 보여주는 프리셋 스크립트를 제공합니다.
  - **버전 관리**: `version/` 폴더 내에 `player`, `ball` 등 핵심 요소의 버전별 구현을 관리합니다.

## 폴더 구조

```
.
├── main.py                # 간단 실행 예제 (선수 생성/행동/업데이트)
├── config.py              # 설정 파일 (현재 비어 있음)
├── requirements.txt       # 프로젝트 의존성 목록
├── physics/               # 핵심 도메인 로직 (player.py, ball.py, engine.py 등)
├── simulation/            # 시뮬레이션 환경 (environment.py)
├── visualize/             # 다양한 시각화 시나리오 모듈
├── version/               # 핵심 요소의 버전별 구현
├── utils/                 # 보조 유틸리티 (atomic_actions.py 등)
└── test/                  # 테스트 코드 (현재는 스텁 파일)
```

## 요구 사항

  - Python 3.x
  - 의존성: `numpy`, `pandas`, `pygame`, `matplotlib`, `scipy`

## 설치

1.  가상 환경을 생성하고 활성화합니다.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    # Windows의 경우: .venv\Scripts\activate
    ```
2.  필요한 패키지를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

## 빠른 시작

아래 명령어로 기본 시뮬레이션을 실행할 수 있습니다.

```bash
python main.py
```

> **참고**: 현재 `main.py`는 `physics.player_version2`에서 `Player`를 임포트합니다. 다른 버전의 구현은 `version/` 폴더에 있으니, 필요시 `import` 경로를 확인하고 수정하세요.

## 시각화 사용법

`visualize/` 폴더에는 패스, 슛, 오버랩/언더랩 등 다양한 시나리오 모듈이 포함되어 있습니다.

각 예제 스크립트를 참고하여 환경, 플레이어, 공을 구성한 뒤, 시각화 클래스를 호출하여 애니메이션을 렌더링할 수 있습니다.

포함된 예제: 패스, 드리블, 슛, 오버랩/언더랩, 맨마킹, 커버섀도우 마킹, 커브드런, 선수간 폭 넓히기/좁히기
