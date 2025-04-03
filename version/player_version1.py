import numpy as np
# from config import delta_time


class Player:
    ACTIONS = {
        "MOVE": [
            "MOVE_LEFT", "MOVE_RIGHT", "MOVE_TOP", "MOVE_BOTTOM",  # 기본 방향 이동
            "MOVE_TOP_LEFT", "MOVE_TOP_RIGHT", "MOVE_BOTTOM_LEFT", "MOVE_BOTTOM_RIGHT"  # 대각선 이동 추가
        ],
        "BALL_CONTROL": [
            "SHORT_PASS", "LONG_PASS", "HIGH_PASS", "SHOT", "CROSS"  # 크로스 추가
        ],
        "DRIBBLING": [
            "SPRINT", "STOP_MOVING", "DRIBBLE",
            "RELEASE_DIRECTION", "RELEASE_SPRINT", "RELEASE_DRIBBLE"  # 버튼 해제 액션 추가
        ],
        "DEFENSE": [
            "SLIDE", "PRESSURE", "TEAM_PRESSURE", "SWITCH",
            "RELEASE_SLIDE", "RELEASE_PRESSURE", "RELEASE_TEAM_PRESSURE", "RELEASE_SWITCH"
        ],
        "GOALKEEPER": [
            "GK_PICKUP", "KEEPER_RUSH", "RELEASE_KEEPER_RUSH"
        ],
        "OTHER": [
            "IDLE", "BUILTIN_AI"  # 기본 정지 & AI 모드
        ]
    }

    def __init__(self, player_id, position, speed=1.0, acceleration=0.1, max_speed=2.0):
        """
        선수 객체 초기화
        :param player_id: 선수 ID
        :param position: 초기 위치 (numpy array)
        :param speed: 선수 기본 이동 속도
        """
        self.player_id = player_id
        self.position = np.array(position, dtype=np.float32)  # 2D 위치 (x, y)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # 이동 속도(벡터)
        self.acceleration = acceleration
        self.speed = speed  # 기본 이동 속도(스칼라)
        self.max_speed = max_speed
        self.current_action = "IDLE"  # 현재 행동
        self.sprinting = False  # 스프린트 여부
        self.dribbling = False  # 드리블 여부
        self.stamina = 100.0
        self.stamina_decay_rate = 0.05  # 기본 체력 감소율
        self.sprint_decay_multiplier = 2.0  # 스프린트 시 체력 감소 배율
        self.accumulated_distance = 0.0

    def move(self, direction):
        """
        선수 이동 (대각선 이동 포함)
        :param direction: 이동 방향 (ex: "MOVE_LEFT", "MOVE_TOP_RIGHT" 등)
        """
        direction_map = {
            "MOVE_LEFT": [-1, 0], "MOVE_RIGHT": [1, 0],
            "MOVE_TOP": [0, -1], "MOVE_BOTTOM": [0, 1],
            "MOVE_TOP_LEFT": [-1, -1], "MOVE_TOP_RIGHT": [1, -1],
            "MOVE_BOTTOM_LEFT": [-1, 1], "MOVE_BOTTOM_RIGHT": [1, 1]
        }

        if direction in direction_map:
            move_vector = np.array(direction_map[direction], dtype=np.float32)
            move_vector = move_vector / np.linalg.norm(move_vector)  # 정규화

            speed_multiplier = 2.5 if self.sprinting else 1.0  # 스프린트 적용
            target_velocity = move_vector * self.speed * speed_multiplier  # 목표 속도

            # 가속도 적용: 현재 속도에서 목표 속도로 점진적으로 증가
            velocity_change = (target_velocity - self.velocity) * self.acceleration
            self.velocity += velocity_change

            # 최대 속도 제한
            velocity_magnitude = np.linalg.norm(self.velocity)
            if velocity_magnitude > self.max_speed:
                self.velocity = (self.velocity / velocity_magnitude) * self.max_speed

                # 이동 거리 누적
                distance_moved = np.linalg.norm(self.velocity)
                self.accumulated_distance += distance_moved

                # 체력 감소 계산
                stamina_loss = self.stamina_decay_rate * distance_moved
                if self.sprinting:
                    stamina_loss *= self.sprint_decay_multiplier  # 스프린트 시 추가 소모
                self.stamina = max(0, self.stamina - stamina_loss)  # 체력 감소 (최소 0)

                # 체력에 따른 속도 조정 (체력이 낮아질수록 속도 감소)
                self.update_speed()
        else:
            # 방향이 없으면 감속
            self.velocity *= 0.6  # 점진적으로 감속 (마찰 효과)

            # 너무 느려지면 정지
            if np.linalg.norm(self.velocity) < 0.01:
                self.velocity = np.array([0.0, 0.0], dtype=np.float32)

    def update_speed(self):
        """체력에 따라 속도를 조정"""
        if self.stamina > 75:
            self.speed = 5.0  # 정상 속도
        elif self.stamina > 50:
            self.speed = 4.5  # 약간 느려짐
        elif self.stamina > 25:
            self.speed = 4.0  # 피로함
        else:
            self.speed = 3.0  # 거의 걷는 수준

    def recover_stamina(self, amount=5.0):
        """체력 회복 (예: 일정 시간 동안 멈춰 있으면 회복)"""
        self.stamina = min(100, self.stamina + amount)

    def perform_action(self, action):
        """
        선수 행동 수행
        :param action: 수행할 행동 (ex: "SPRINT", "SHORT_PASS", "TACKLE" 등)
        """
        all_actions = sum(self.ACTIONS.values(), [])
        if action in all_actions:
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
                self.velocity = np.array([0.0, 0.0], dtype=np.float32)
            elif action.startswith("MOVE_"):  # 이동 관련 액션
                self.move(action)
        else:
            print(f"⚠️ 잘못된 행동 입력: {action}")

    def update(self, delta_time):
        """
        선수의 위치 업데이트 (delta_time을 활용하여 물리적 반영)
        """
        self.position += self.velocity * delta_time

    def __str__(self):
        return f"Player {self.player_id}: Pos={self.position}, Vel={self.velocity}, Action={self.current_action}"


# 예제 실행
if __name__ == "__main__":
    player = Player(player_id=1, position=[50, 50])
    player.perform_action("MOVE_TOP_RIGHT")
    player.update(0.1)  # 0.1초 경과
    print(player.position)
