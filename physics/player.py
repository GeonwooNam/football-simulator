import random
import numpy as np
from simulation.environment import Field
from physics.stamina_system import StaminaSystem


ACTIONS = {
    "MOVE": [
        "MOVE_LEFT", "MOVE_RIGHT", "MOVE_TOP", "MOVE_BOTTOM",
        "MOVE_TOP_LEFT", "MOVE_TOP_RIGHT", "MOVE_BOTTOM_LEFT", "MOVE_BOTTOM_RIGHT"
    ],
    "BALL_CONTROL": ["SHORT_PASS", "LONG_PASS", "HIGH_PASS", "SHOT", "CROSS"],
    "DRIBBLING": ["SPRINT", "STOP_MOVING", "DRIBBLE",
                  "RELEASE_DIRECTION", "RELEASE_SPRINT", "RELEASE_DRIBBLE", "DASH_DRIBBLE"],
    "DEFENSE": ["SLIDE", "PRESSURE", "TEAM_PRESSURE", "SWITCH",
                "RELEASE_SLIDE", "RELEASE_PRESSURE", "RELEASE_TEAM_PRESSURE", "RELEASE_SWITCH"],
    "GOALKEEPER": ["GK_PICKUP", "KEEPER_RUSH", "RELEASE_KEEPER_RUSH", "SHORT_PASS", "LONG_PASS", "HIGH_PASS"],
    "OTHER": ["IDLE"]
}


class Player:
    TURN_SPEED = 0.6  # 방향 전환 속도 (0~1, 1이면 즉각 전환)
    DECELERATION_FACTOR = 0.6  # 감속 계수 (0~1)
    collision_threshold = 2.5
    shoot_distance = 3.0
    mark_front_check_time = 1.8

    MOVE_INTENSITY = ["IDLE", "WALK", "JOGGING", "LOW_INTENSITY_RUNNING", "HIGH_INTENSITY_RUNNING", "SPRINT"]

    player_list = []

    def __init__(self, player_id, position, role, team_id=0, max_speed=30.0,
                 acceleration=5.0, stamina=100.0, stat_dict={}):
        """
        :param player_id: 선수 식별자
        :param position: 초기 위치 (리스트 또는 튜플, [x, y])
        :param team_id: 팀 id
        :param max_speed: 최대 이동 속력
        :param acceleration: 가속도 계수 (목표 속도에 도달하는 변화율)
        :param stamina: 선수의 체력
        """
        
        # 선수 및 팀 정보
        self.player_id = player_id
        self.team_id = team_id
        self.is_home_team = None
        self.role = role

        # 운동 상태
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.move_intensity = "IDLE"
        self.move_direction = None
        self.target_speed = 0.0
        self.speed = 0.0
        self.max_speed = max_speed / 3.6
        self.acceleration = acceleration
        self.current_action = "IDLE"

        # 드리블
        self.sprinting = False
        self.dribbling = False
        self.dash_dribble = False
        self.skill_move = False
        self.ball = None

        # 능력치
        self.ball_control_quality = stat_dict.get("ball_control", 80.0)
        self.skill_rating = stat_dict.get("skill", 5.0)
        self.pass_quality = stat_dict.get("pass", 80.0)
        self.shoot_quality = stat_dict.get("shoot", 80.0)
        self.max_power = stat_dict.get("power", 80.0)

        # 스태미너
        self.stamina = stamina
        self.stamina_system = StaminaSystem(max_stamina=stamina)

        # 커브드런 관련
        self.move_vector = None
        self.target_pos = None
        self.curve_motion = {}

        # 마킹 관련
        self.view_ball = None
        self.marked_target = None
        self.last_marked_target_position = np.zeros(2, dtype=np.float32)  # 커버섀도우 시 알고 있는 위치 바탕으로 수비
        self.marking_offset = (0, 0)  # 앞이면 +값, 뒤면 -값
        self.is_marking = False
        self.mark_front = False
        self.mark_front_timer = 0.0
        self.cover_shadow = False

        # 뒤돌기
        self.is_rotating = False
        self.rotate_timer = 0.0

        # 포지셔닝
        self.position = None
        self.offense_mode = True
        self.offense_line = 4  # TODO. 값 조정
        self.offense_idx = 0
        self.defense_line = 3
        self.defense_idx = 0

        self.player_list.append(self)

    def update_move_intensity(self, intensity_):
        self.move_intensity = intensity_
        if intensity_ in ["WALK", "JOGGING", "LOW_INTENSITY_RUNNING", "HIGH_INTENSITY_RUNNING"]:
            self.current_action = "MOVE"
        elif intensity_ == "SPRINT":
            if self.stamina < 10.0:  # 스프린트 불가
                self.move_intensity = "HIGH_INTENSITY_RUNNING"
            else:
                self.current_action = "SPRINT"
        elif intensity_ == "IDLE":
            self.current_action = "IDLE"
            self.velocity = np.zeros(2, dtype=np.float32)
            self.speed = 0.0
        self.get_target_speed()  # 목표 속도 업데이트

    def update_running_info(self, move_direction=None, move_intensity=None):
        self.dribbling = False
        self.dash_dribble = False
        self.move_direction = move_direction
        self.update_move_intensity(move_intensity)

    def update_dribble_info(self, is_dash_dribble=False, move_direction=None, move_intensity=None):
        self.dribbling = True
        self.dash_dribble = is_dash_dribble
        if move_direction:
            self.move_direction = move_direction
        if move_intensity:
            self.update_move_intensity(move_intensity)

    def _get_direction_vector(self):
        """
        주어진 방향 문자열에 해당하는 정규화된 2D 벡터 반환
        """
        if self.move_vector is not None:
            vector = np.array(self.move_vector, dtype=np.float32)
        elif self.target_pos is not None:
            self.move_vector = self.target_pos[:2] - self.position[:2]
            vector = np.array(self.move_vector, dtype=np.float32)
        else:
            direction_map = {
                "MOVE_LEFT": [-1, 0], "MOVE_RIGHT": [1, 0],
                "MOVE_TOP": [0, -1], "MOVE_BOTTOM": [0, 1],
                "MOVE_TOP_LEFT": [-1, -1], "MOVE_TOP_RIGHT": [1, -1],
                "MOVE_BOTTOM_LEFT": [-1, 1], "MOVE_BOTTOM_RIGHT": [1, 1]
            }
            vector = np.array(direction_map.get(self.move_direction, [0, 0]), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def get_target_speed(self):
        """
        이동 강도에 따라 목표 속도를 설정
        """
        intensity_speeds = {
            "IDLE": 0,
            "WALK": 5 / 3.6,
            "JOGGING": 7 / 3.6,
            "LOW_INTENSITY_RUNNING": 15 / 3.6,
            "HIGH_INTENSITY_RUNNING": max(20 / 3.6, 0.7 * self.max_speed * self.stamina_speed_modifier()),
            "SPRINT": self.max_speed * self.stamina_speed_modifier()
        }
        speed = intensity_speeds.get(self.move_intensity, 0)
        if self.dribbling and not self.dash_dribble:
            speed *= 0.9  # 드리블 중일 때 약간 감속
        self.target_speed = speed

    def stamina_speed_modifier(self):
        """
        스태미너에 따라 속도 보정 계수 반환 (0.5 ~ 1.0 사이)
        """
        stamina_ratio = self.stamina / 100.0
        return 0.5 + 0.5 * stamina_ratio  # stamina 100 → 1.0, stamina 0 → 0.5

    def get_dribble_touch_length(self):
        base_touch = {
            "IDLE": 0.0,
            "WALK": 0.5,
            "JOGGING": 1.0,
            "LOW_INTENSITY_RUNNING": 1.5,
            "HIGH_INTENSITY_RUNNING": 2.0,
            "SPRINT": 2.5
        }
        length = base_touch.get(self.move_intensity, 1.0)
        control_ratio = 1.0 - (self.ball_control_quality - 70.0) / 300.0
        control_modifier = np.clip(control_ratio, 0.85, 1.20)
        if self.stamina < 30:
            control_modifier += (1.0 - self.stamina / 30) * 0.5
        return length * control_modifier

    def perform_dribble_touch(self):
        if not self.dribbling or not self.ball:
            return
        direction = self._get_direction_vector()
        if np.linalg.norm(direction) == 0:
            return
        touch_length = self.get_dribble_touch_length()
        min_ball_distance = 2.0
        actual_touch = max(min_ball_distance, touch_length)
        self.ball.position[:2] = self.position[:2] + direction * actual_touch

    def perform_dash_dribble(self, touch_length=5.0):
        if not self.dribbling or not self.ball:
            return
        direction = self._get_direction_vector()
        if np.linalg.norm(direction) == 0:
            return
        touch_length = max(touch_length, 5.0)  # 강제 긴 터치
        if np.linalg.norm(self.position[:2] - self.ball.position[:2]) < self.collision_threshold:
            self.ball.position[:2] = self.position[:2] + direction * touch_length

    def try_skill_move(self):  # TODO. 고도화 필요
        if not self.dribbling:
            return False
        skill_chance = self.skill_rating / 5.0
        self.skill_move = True  # stamina에 넣을거면 필요
        # TODO. 스태미너와 수비수 민첩성, 태클 정확도에 따른 성공률 조정
        self.ball.skill_move = True
        if random.random() < skill_chance:
            direction = self._get_direction_vector()
            self.velocity = direction * self.target_speed * 1.2  # 순간 가속
            return True
        return False

    def able_to_shoot(self, ball):
        return self.distance_to_ball(ball) < self.shoot_distance

    def distance_to_ball(self, ball):
        return np.linalg.norm(self.position[:2] - ball.position[:2])

    def get_stamina_modifier(self, action=None):
        return self.stamina_system.inaccuracy_modifier(action=action)

    def get_stamina_percentage(self):
        return self.stamina / 100.0

    def _smooth_turn(self, current_dir, target_dir):
        """
        방향 전환을 부드럽게 하기 위한 보간 함수
        """
        if np.linalg.norm(current_dir) == 0:
            return target_dir  # 정지 상태면 바로 목표 방향 사용

        # 현재 방향 벡터 정규화
        current_dir = current_dir / np.linalg.norm(current_dir)

        # 목표 방향 벡터 정규화
        target_dir = target_dir / np.linalg.norm(target_dir)

        # 보간된 방향 계산 (LERP 사용)
        new_direction = (1 - self.TURN_SPEED) * current_dir + self.TURN_SPEED * target_dir
        return new_direction / np.linalg.norm(new_direction)  # 정규화된 벡터 반환

    def _apply_inertia(self, target_direction):
        """
        현재 속도 방향과 목표 방향의 차이에 따라 감속을 적용
        """
        if np.linalg.norm(self.velocity) == 0:
            return target_direction  # 정지 상태면 바로 목표 방향 사용

        # 현재 속도 방향 정규화
        current_dir = self.velocity / np.linalg.norm(self.velocity)

        # 목표 방향 정규화
        target_dir = target_direction / np.linalg.norm(target_direction)

        # 두 벡터의 내적 계산 (cosine similarity)
        similarity = np.dot(current_dir, target_dir)

        # 방향 전환이 클수록 속도를 줄이도록 설정
        if similarity < 0.5:  # 60도 이상 차이나면 속도를 감속
            self.velocity *= 0.8
        elif similarity < 0:  # 정반대 방향(180도 회전)이라면 속도를 더 감속
            self.velocity *= 0.5

        return target_dir

    def update_velocity(self):
        """
        방향 전환을 부드럽게 적용하여 속도를 업데이트하는 함수
        """
        if self.move_intensity == "IDLE":
            self.velocity[:] = 0
            return

        target_direction = self._get_direction_vector()
        if np.linalg.norm(target_direction) > 0:
            if np.linalg.norm(self.velocity) == 0:
                self.velocity = target_direction * self.speed
            else:
                # 관성 적용 (급격한 방향 전환 시 감속)
                adjusted_target_direction = self._apply_inertia(target_direction)

                # 부드러운 방향 전환 적용
                smooth_direction = self._smooth_turn(self.velocity, adjusted_target_direction)

                # 최종 속도 적용
                self.velocity = smooth_direction * min(self.target_speed, np.linalg.norm(self.velocity) + self.acceleration * self.target_speed)

    def clear_target_pos(self):
        self.target_pos = None
        self.move_vector = None

    def move_towards(self, target_pos, move_intensity=None):
        if move_intensity and self.move_intensity != move_intensity:
            self.update_move_intensity(move_intensity)
        self.target_pos = target_pos[:2]

    def hold_position(self):
        self.velocity *= 0.2  # 점차 멈추게
        if np.linalg.norm(self.velocity) < 0.1:
            self.velocity *= 0

    # 수비 위치 복귀 (필요 없을수도 있음)
    def reposition_to(self, target_pos, stop_tolerance=0.3, move_intensity="LOW_INTENSITY_RUNNING"):
        delta = target_pos[:2] - self.position[:2]
        distance = np.linalg.norm(delta)
        if distance > stop_tolerance:
            self.move_towards(target_pos, move_intensity=move_intensity)
        else:
            self.hold_position()

    # 거리 유지하며 따라가기
    def adjust_distance_from(self, target, desired_distance=1.5):
        delta = target.position[:2] - self.position[:2]
        distance = np.linalg.norm(delta)
        if distance < 1e-4:
            return
        direction = delta / distance
        move_needed = distance - desired_distance
        if abs(move_needed) > 0.1:  # 허용 오차
            self.position[:2] += self.velocity * DT
        else:
            self.velocity *= 0  # 멈추기
            self.speed = 0.0

    def is_ball_angle_attack_side(self):
        ball_angle = self.ball_angle()
        if self.is_home_team:
            return not 90 < ball_angle % 360 < 270
        else:
            return 90 < ball_angle % 360 < 270

    def ball_angle(self):
        if self.ball is not None:
            relative_pos = self.ball.position[:2] - self.position[:2]
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            angle_deg = np.degrees(angle)
            return angle_deg

    @staticmethod
    def calculate_angle_to_move(angle1, angle2):
        return (angle2 - angle1 + 180) % 360 - 180

    def rotate_towards(self, dt, degree=0.0):  # degree는 이동할 각도 (델타값 아님)
        if self.ball is not None:
            current_angle = self.ball_angle()
            degree_to_move = self.calculate_angle_to_move(current_angle, degree)

            if not self.is_rotating:
                self.is_rotating = True
                self.rotate_timer = abs(degree_to_move) / 180.0  # TODO. 값 조정
            else:
                self.rotate_timer -= dt

            if self.is_rotating and self.rotate_timer <= 0:
                # 현재 위치 기준, 공과의 벡터
                relative_pos = self.ball.position[:2] - self.position[:2]

                # 회전 벡터 계산
                rad = degree_to_move * np.pi / 180.0
                rotation_matrix = np.array([
                    [np.cos(rad), -np.sin(rad)],
                    [np.sin(rad), np.cos(rad)]
                ])
                rotated_vec = rotation_matrix @ relative_pos

                # 회전 후 위치 적용
                self.ball.position[:2] = self.position[:2] + rotated_vec
                self.is_rotating = False
                self.rotate_timer = 0.0

    def start_marking(self, target_player, offset=np.array([5, 0]), mark_front=False, cover_shadow=True, update_interval=1.8):
        """마킹 시작 - 대상과 오프셋 설정"""
        # 무조건 뒤(또는 옆)에서 마킹하도록 (그래야 상시 체크 가능)
        multiplier = 1
        if self.is_home_team:
            multiplier *= -1
        if mark_front or cover_shadow:
            multiplier *= -1
            self.mark_front = True
            self.mark_front_check_time = update_interval
            self.mark_front_timer = self.mark_front_check_time
            if cover_shadow:
                self.cover_shadow = True
                self.last_marked_target_position = np.array(target_player.position[:2].copy(), dtype=np.float32)
        else:
            self.mark_front = False

        offset[0] = multiplier * abs(offset[0])

        self.marked_target = target_player
        self.marking_offset = offset
        self.is_marking = True

    def stop_marking(self):
        """마킹 중지"""
        self.marked_target = None
        self.last_marked_target_position = np.zeros(2, dtype=np.float32)
        self.is_marking = False
        self.mark_front = False
        self.cover_shadow = False

    def update_marking(self, dt, move_intensity="LOW_INTENSITY_RUNNING"):
        """마킹 중일 때 위치 업데이트"""
        update = False
        if not self.mark_front:  # 뒤에서 마킹할 경우 상시 업데이트
            update = True
        elif not self.cover_shadow:  # 앞에서 마킹할 경우 1.5초마다 업데이트 (커버섀도우는 별도)
            self.mark_front_timer -= dt
            if self.mark_front_timer <= 0:
                update = True
                self.mark_front_timer = self.mark_front_check_time

        if self.is_marking and self.marked_target:  # 타겟의 러닝 강도가 LOW_INTENSITY_RUNNING보다 빠를 경우 그값대로 업데이트
            target_move_intensity = self.marked_target.move_intensity
            target_move_intensity_idx = [i for (i, move_intensity_) in enumerate(self.MOVE_INTENSITY) if move_intensity_ == target_move_intensity][0]
            move_intensity_idx = [i for (i, move_intensity_) in enumerate(self.MOVE_INTENSITY) if move_intensity_ == move_intensity][0]
            if target_move_intensity_idx > move_intensity_idx:
                move_intensity = target_move_intensity

        if update and self.is_marking and self.marked_target:  # 일반 마킹 업데이트
            # 목표 마킹 위치 계산
            target_pos = self.marked_target.position[:2] + self.marking_offset
            direction = target_pos - self.position[:2]
            length = np.linalg.norm(direction)

            if length > 15.0:
                move_intensity = "SPRINT"
                self.update_move_intensity(move_intensity)
            if length > 0.05:  # 너무 가까우면 안 움직이게
                # print(self.speed, self.move_intensity)
                if self.move_intensity == "IDLE":
                    self.update_move_intensity(move_intensity)
                move = direction / length * self.speed * dt
                self.position[:2] += move
            else:
                self.update_move_intensity("IDLE")
            return

        # 마킹 대상이 공을 소유한 경우 (수비 뚫린 경우) → 커버섀도우 해제, 뒤에서 다시 마킹
        if self.view_ball.owner_player_id == self.marked_target.player_id:
            self.start_marking(
                self.marked_target,
                offset=np.array([6, 0]),
                mark_front=False,
                cover_shadow=False
            )
            return

        elif self.is_marking and self.marked_target and self.cover_shadow:  # 커버섀도우 마킹 업데이트
            self.mark_front_timer -= dt
            if self.mark_front_timer <= 0:
                self.mark_front_timer = self.mark_front_check_time
                self.last_marked_target_position = np.array(self.marked_target.position[:2].copy(), dtype=np.float32)

            # 공의 위치와 마지막으로 업데이트된 공격수 위치 사이를 수비
            distance = np.linalg.norm(self.marking_offset)
            ball_pos = self.view_ball.position[:2]
            mark_target_pos = self.last_marked_target_position

            direction = ball_pos - mark_target_pos
            norm = np.linalg.norm(direction)

            if norm > 0.01:  # 0으로 나누지 않도록 방어
                direction_unit = direction / norm
                cover_shadow_pos = mark_target_pos + direction_unit * distance
                target_pos = cover_shadow_pos
            else:
                # 공과 타겟이 너무 가까우면 그냥 타겟 위치로
                target_pos = mark_target_pos

            # 위치 이동
            direction = target_pos - self.position[:2]
            length = np.linalg.norm(direction)

            if length > 15.0:
                move_intensity = "SPRINT"
                self.update_move_intensity(move_intensity)
            if length > 0.05:  # 너무 가까우면 안 움직이게
                # print(self.speed, self.move_intensity)
                if self.move_intensity == "IDLE":
                    self.update_move_intensity(move_intensity)
                move = direction / length * self.speed * dt
                self.position[:2] += move
            else:
                self.update_move_intensity("IDLE")
            return

    def update_speed(self, dt):
        """
        속도 조절 로직 (가속 및 감속 적용)
        """
        if self.speed < self.target_speed:
            self.speed += self.acceleration * self.target_speed * dt  # 가속 적용
            if self.speed > self.target_speed:
                self.speed = self.target_speed  # 초과하지 않도록 제한
        elif self.speed > self.target_speed:
            self.speed *= self.DECELERATION_FACTOR  # 감속 적용
            if self.speed < self.target_speed:
                self.speed = self.target_speed  # 초과 감속 방지

    def update(self, dt, dash_touch_length=5.0):
        """
        매 프레임 호출: 속도 및 방향 업데이트, 위치 조정
        """
        self.stamina_system.update(self, dt)
        self.stamina = self.stamina_system.stamina

        self.update_speed(dt)  # 속도 조절
        self.update_velocity()

        # 위치 업데이트
        if self.curve_motion:
            self.update_curve_motion(dt)
        else:
            self.position[:2] += self.velocity * dt

        if self.target_pos is not None:
            to_target = self.target_pos - self.position[:2]
            if np.linalg.norm(to_target) < 0.5:
                self.speed = 0
                self.velocity = np.zeros(2, dtype=np.float32)
                self.target_pos = None
                self.move_vector = None
                self.move_direction = None

        # 경기장 경계 내에서 위치 제한
        self.position[0] = min(max(self.position[0], 0), Field.REAL_WIDTH)
        self.position[1] = min(max(self.position[1], 0), Field.REAL_HEIGHT)

        if self.dribbling:
            if self.dash_dribble:
                self.perform_dash_dribble(dash_touch_length)
            else:
                self.perform_dribble_touch()

    def __str__(self):  # TODO. 수정
        # Player {self.player_id}:
        return (f"Position={self.position}, "
                f"Velocity={self.velocity*3.6}, Speed={self.speed*3.6:.2f}, "
                f"Target Speed={self.target_speed*3.6:.2f}, "
                f"Intensity={self.move_intensity}, Move Direction={self.move_direction}, Stamina={self.stamina:.2f}")

    def compute_overlap_curve(self, winger, outside_offset=5.0, shorten_ratio=1.0, move_intensity="SPRINT"):
        self.update_move_intensity(move_intensity)

        target_pos = winger.position[:2].copy()
        target_pos[1] += outside_offset if target_pos[1] > Field.REAL_HEIGHT / 2 else (-outside_offset)  # 바깥쪽으로 얼마나 돌지

        block_pos = target_pos.copy()
        target_pos[0] += 5.0 if target_pos[0] > Field.REAL_WIDTH / 2 else -5.0  # 곡선으로 돌아서 앞쪽까지 침투
        block_pos[0] += 10.0 if target_pos[0] < Field.REAL_WIDTH / 2 else -10.0  # 인위적으로 만든 수직의 막는 선수 위치 (값은 의미없음)

        self.compute_circle_curve(target_pos, block_pos, shorten_ratio=shorten_ratio)

    def compute_underlap_run(self, defender, inside_offset=3.0, move_intensity="SPRINT"):
        target_pos = defender.position[:2].copy()
        target_pos[1] += inside_offset if target_pos[1] < Field.REAL_HEIGHT / 2 else (-inside_offset)
        target_pos[0] += 7.0 if target_pos[0] > Field.REAL_WIDTH / 2 else -7.0  # 앞쪽까지 침투
        self.move_towards(target_pos, move_intensity=move_intensity)

    def depth_providing(self, target_y=None, x_offset=0.0, move_intensity="HIGH_INTENSITY_RUNNING"):
        """
        :param target_y: 좌우로 어떤 좌표에 위치할지 (선택값)
        :param x_offset: 양수면 더 깊이 위치, 음수면 더 얕게(낮게 위치)
        :param move_intensity: 운동강도
        :return:
        """
        opponent_players = [p for p in self.player_list if self.team_id != p.team_id]
        if self.is_home_team:
            target_x = max(p.position[0] for p in opponent_players) + x_offset
        else:
            target_x = min(p.position[0] for p in opponent_players) - x_offset

        target_pos = np.array([target_x, target_y if target_y is not None else self.position[1]], dtype=np.float32)
        self.move_towards(target_pos, move_intensity)
        self.offense_line = 1

    def compute_circle_curve(self, target_pos, block_pos, shorten_ratio=1.0):
        """
        target_pos: 도달해야 할 위치
        block_pos: 패스길을 막아야 하는 상대편 위치
        shorten_ratio: 1.0이면 기존 원,
                       0.5이면 완만한 곡선,
                       1.5이면 날카로운 곡선. (수직으로 떨어지는 거리 기준)
        """
        # 시작점(A)와 목표점(B) 좌표
        A = np.array(self.position[:2], dtype=np.float32)
        B = np.array(target_pos[:2], dtype=np.float32)

        # chord_mid: A와 B의 중간점
        chord_mid = (A + B) / 2.0
        chord_length = np.linalg.norm(B - A)

        # 기존 방식으로 block_player를 이용해 원의 중심(C_orig) 계산
        dir_AB = B - A
        C_block = np.array(block_pos[:2], dtype=np.float32)
        dir_TC = C_block - B
        perp_AB = np.array([dir_AB[1], -dir_AB[0]])
        perp_TC = np.array([dir_TC[1], -dir_TC[0]])
        M_matrix = np.column_stack((perp_TC, -perp_AB))
        # 여기서는 chord_mid를 기준으로 C_orig를 구해보자.
        rhs = chord_mid - B
        try:
            st = np.linalg.solve(M_matrix, rhs)
        except np.linalg.LinAlgError:
            return None
        s, _ = st
        C_orig = B + s * perp_TC

        # 기존 반지름 r_orig: A와 C_orig 사이의 거리
        r_orig = np.linalg.norm(A - C_orig)
        # h_orig: chord_mid와 C_orig 사이의 거리
        h_orig = np.linalg.norm(C_orig - chord_mid)
        # L_orig: 원 경계(시작점 기준 r_orig)와 chord_mid 사이의 거리
        L_orig = r_orig - h_orig
        # L_target: shorten_ratio에 따라 조절된 L_orig
        L_target = shorten_ratio * L_orig

        # 새로운 반지름를 구하는 식:
        # new_radius^2 = (new_radius - L_target)^2 + (chord_length/2)^2
        # 이를 풀면 new_radius = (L_target^2 + (chord_length/2)^2) / (2 * L_target)
        # 단, L_target가 0이면 기존 반지름을 그대로 사용.
        if L_target == 0:
            new_radius = r_orig
        else:
            new_radius = (L_target ** 2 + (chord_length / 2) ** 2) / (2 * L_target)

        # new_center: chord_mid에서, C_orig와 chord_mid를 잇는 선의 방향으로,
        # new_center는 chord_mid로부터 (new_radius - L_target)만큼 떨어진 곳에 위치
        u = C_orig - chord_mid
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            u = np.array([0, 1], dtype=np.float32)
        else:
            u = u / norm_u
        new_center = chord_mid + (new_radius - L_target) * u

        # 새로운 시작각과 도착각: new_center를 기준으로
        new_angle_start = np.arctan2(A[1] - new_center[1], A[0] - new_center[0])
        new_angle_end = np.arctan2(B[1] - new_center[1], B[0] - new_center[0])

        # 진행 방향 결정: 만약 cross(B-new_center, A-new_center) < 0이면 시계 방향
        clockwise = np.cross(B - new_center, A - new_center) < 0

        if self.target_speed == 0.0:
            self.move_intensity = "LOW_INTENSITY_RUNNING"
            self.get_target_speed()
        angular_speed = self.target_speed / new_radius if new_radius != 0 else 0.0

        # curve_motion에 저장 (update_curve_motion()에서 사용)
        self.curve_motion = {
            "center": new_center,
            "radius": new_radius,
            "angle_start": new_angle_start,
            "angle_end": new_angle_end,
            "clockwise": clockwise,
            "angular_speed": angular_speed,
            "current_angle": new_angle_start,
            "target": B
        }

    def update_curve_motion(self, dt):
        cm = self.curve_motion
        if not cm:
            return

        # 각속도만큼 angle 누적
        delta_angle = cm["angular_speed"] if cm["clockwise"] else -cm["angular_speed"]
        cm["current_angle"] += delta_angle * dt

        # 현재 각도 계산 (전체 궤적은 angle_start → angle_end)
        angle = cm["current_angle"]

        # 새로운 위치 계산
        x = cm["center"][0] + cm["radius"] * np.cos(angle)
        y = cm["center"][1] + cm["radius"] * np.sin(angle)
        self.position[:2] = np.array([x, y], dtype=np.float32)

        # 속도 방향 업데이트 (선택)
        self.velocity = np.array([
            -np.sin(angle), np.cos(angle)  # 원 궤도에서 접선 방향
        ]) * self.speed  # 또는 self.target_speed

        # 종료 조건 (도착 또는 전체 각도 도달)
        angle_diff = cm["angle_end"] - cm["angle_start"]
        progress = (cm["current_angle"] - cm["angle_start"]) / angle_diff if angle_diff != 0 else 1.0

        # 시계/반시계 방향별 종료 조건 달리 적용
        done = (progress >= 1.0) if not cm["clockwise"] else (progress <= -1.0)

        if done or np.linalg.norm(self.position[:2] - cm["target"][:2]) < 0.5:
            self.clear_curve_motion()

    def clear_curve_motion(self):
        self.curve_motion = {}
        self.move_intensity = "IDLE"
        self.update_velocity()

    @classmethod
    def add_home_away_and_view_ball(cls, ball):
        for p in cls.player_list:
            if p.team_id == ball.home_team_id:
                p.is_home_team = True
            elif p.team_id == ball.away_team_id:
                p.is_home_team = False
            else:
                p.is_home_team = None
            p.view_ball = ball


# 테스트 실행
if __name__ == "__main__":
    cb = Player(player_id=1, position=np.array([80.0, 34.0, 0.0]), team_id=1)
    fb = Player(player_id=2, position=np.array([79.0, 24.0, 0.0]), team_id=1)
    st = Player(player_id=9, position=np.array([0.0, 40.0, 0.0]), team_id=2)
    players = [cb, fb, st]

    st.compute_circle_curve(fb, cb)


    """
    player = Player(player_id=1, position=[50, 30], max_speed=30, acceleration=0.2, stamina=100)

    # 이동 강도 변경 테스트
    intensity_levels = ["IDLE", "WALK", "JOGGING", "LOW_INTENSITY_RUNNING", "HIGH_INTENSITY_RUNNING", "SPRINT"]
    move_direction = [
        "MOVE_LEFT", "MOVE_RIGHT", "MOVE_TOP", "MOVE_BOTTOM",
        "MOVE_TOP_LEFT", "MOVE_TOP_RIGHT", "MOVE_BOTTOM_LEFT", "MOVE_BOTTOM_RIGHT"
    ]

    change_percentage = 0.0
    for i in range(50):
        if random.random() < change_percentage:
            player.update_move_intensity(random.choice(intensity_levels))
            player.move_direction = random.choice(move_direction)
            change_percentage = 0.0
        else:
            change_percentage += 0.1
        player.update(1)
        print(player)
    """
