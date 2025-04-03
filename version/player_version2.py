import numpy as np
from simulation.environment import Field


class Player:
    ACTIONS = {
        "MOVE": [
            "MOVE_LEFT", "MOVE_RIGHT", "MOVE_TOP", "MOVE_BOTTOM",
            "MOVE_TOP_LEFT", "MOVE_TOP_RIGHT", "MOVE_BOTTOM_LEFT", "MOVE_BOTTOM_RIGHT"
        ],
        "BALL_CONTROL": ["SHORT_PASS", "LONG_PASS", "HIGH_PASS", "SHOT", "CROSS"],
        "DRIBBLING": ["SPRINT", "STOP_MOVING", "DRIBBLE",
                      "RELEASE_DIRECTION", "RELEASE_SPRINT", "RELEASE_DRIBBLE"],
        "DEFENSE": ["SLIDE", "PRESSURE", "TEAM_PRESSURE", "SWITCH",
                    "RELEASE_SLIDE", "RELEASE_PRESSURE", "RELEASE_TEAM_PRESSURE", "RELEASE_SWITCH"],
        "GOALKEEPER": ["GK_PICKUP", "KEEPER_RUSH", "RELEASE_KEEPER_RUSH"],
        "OTHER": ["IDLE", "BUILTIN_AI"]
    }

    def __init__(self, player_id, position, base_speed=10.0, max_speed=30.0,
                 acceleration=5.0, stamina=100.0):
        """
        :param player_id: 선수 식별자
        :param position: 초기 위치 (리스트 또는 튜플, [x, y])
        :param base_speed: 기본 이동 속력
        :param max_speed: 최대 이동 속력
        :param acceleration: 가속도 계수 (목표 속도에 도달하는 변화율)
        :param stamina: 선수의 체력
        """
        self.player_id = player_id
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.base_speed = base_speed / 3.6
        self.max_speed = max_speed / 3.6
        self.acceleration = acceleration / 3.6
        self.stamina = stamina
        self.current_action = "IDLE"
        self.sprinting = False
        self.dribbling = False

    def _get_direction_vector(self, direction):
        """
        주어진 방향 문자열에 해당하는 정규화된 2D 벡터 반환
        """
        direction_map = {
            "MOVE_LEFT": [-1, 0], "MOVE_RIGHT": [1, 0],
            "MOVE_TOP": [0, -1], "MOVE_BOTTOM": [0, 1],
            "MOVE_TOP_LEFT": [-1, -1], "MOVE_TOP_RIGHT": [1, -1],
            "MOVE_BOTTOM_LEFT": [-1, 1], "MOVE_BOTTOM_RIGHT": [1, 1]
        }
        vector = np.array(direction_map.get(direction, [0, 0]), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def update_velocity(self, target_direction):
        """
        속도 업데이트 과정을 명확하게 분리한 함수 (수도코드 기반)

        1. 스프린트 여부에 따라 effective_speed 결정
        2. 목표 속도 벡터(target_velocity) 계산
        3. 현재 속도와 목표 속도의 차이(velocity_delta) 계산
        4. 가속도 계수를 적용하여 현재 속도를 점진적으로 변화
        5. 업데이트된 속도가 max_speed를 초과하면 제한 적용
        """
        # 1. 목표 속력 결정
        effective_speed = self.base_speed * (1.5 if self.sprinting else 1.0)

        # 2. 목표 속도 벡터 계산 (target_direction은 이미 정규화된 상태)
        target_velocity = target_direction * effective_speed

        # 3. 현재 속도와 목표 속도 차이 계산
        velocity_delta = target_velocity - self.velocity

        # 4. 가속도 적용하여 업데이트된 속도 계산
        updated_velocity = self.velocity + (velocity_delta * self.acceleration)

        # 5. 최대 속도 제한 적용
        current_speed = np.linalg.norm(updated_velocity)
        if current_speed > self.max_speed:
            updated_velocity = (updated_velocity / current_speed) * self.max_speed

        self.velocity = updated_velocity

    def move(self, direction):
        """
        이동 명령 처리: 주어진 방향에 따라 update_velocity() 호출
        """
        direction_vector = self._get_direction_vector(direction)
        self.update_velocity(direction_vector)

    def perform_action(self, action):
        """
        전달된 행동 문자열에 따라 상태 변화 및 이동 처리
        """
        all_actions = sum(self.ACTIONS.values(), [])
        if action not in all_actions:
            print(f"⚠️ Invalid action: {action}")
            return

        self.current_action = action

        if action == "SPRINT":
            self.sprinting = True
        elif action == "RELEASE_SPRINT":
            self.sprinting = False
        elif action == "DRIBBLE":
            self.dribbling = True
        elif action == "RELEASE_DRIBBLE":
            self.dribbling = False
        elif action == "STOP_MOVING":
            self.velocity = np.zeros(2, dtype=np.float32)
        elif action.startswith("MOVE_"):
            self.move(action)
        # 추가적인 행동 로직(패스, 슛 등)은 여기서 처리

    def _apply_friction(self, friction=0.5):
        """
        이동 입력이 없을 때, 마찰로 인한 감속 처리
        """
        self.velocity *= friction
        if np.linalg.norm(self.velocity) < 0.01:
            self.velocity = np.zeros(2, dtype=np.float32)

    def update(self, delta_time):
        """
        매 프레임 호출: 위치 업데이트, 스태미너 감소 및 감속 처리
        """
        self.position += self.velocity * delta_time
        
        # 최대 최소 처리
        self.position[0] = min(max(self.position[0], 0), Field.REAL_WIDTH)
        self.position[1] = min(max(self.position[1], 0), Field.REAL_HEIGHT)

        if self.sprinting:
            self.stamina = max(self.stamina - 0.5 * delta_time, 0)

        # 이동 입력이 없으면 감속 처리
        if not self.current_action.startswith("MOVE_"):
            self._apply_friction()

    def __str__(self):
        return (f"Player {self.player_id}: Position={self.position}, "
                f"Velocity={self.velocity}, Action={self.current_action}, "
                f"Stamina={self.stamina:.2f}")


# 예제 실행
if __name__ == "__main__":
    # 선수 생성: player_id, 초기위치, 기본속력, 최대속력, 가속도, 스태미너
    player = Player(player_id=1, position=[50, 50], base_speed=10, max_speed=30, acceleration=5, stamina=100)

    # 행동 수행: 스프린트 후 대각선 이동
    player.perform_action("SPRINT")
    player.perform_action("MOVE_TOP_RIGHT")

    # 10 프레임 동안 업데이트 (각 프레임 0.1초)
    for i in range(5):
        player.update(1)
        print(player)
