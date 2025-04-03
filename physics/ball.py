import numpy as np
from simulation.environment import Field
from physics.stamina_system import StaminaSystem


class Ball:
    gravity = 30.0
    shoot_gravity = 2.0
    deflection_threshold = 70.0  # 맞고 굴절되는 공 최소 속도 (이값 이하면 소유권 획득)
    friction = 0.65  # 마찰력 정도
    min_speed = 1.0  # 일정 이하 속도는 멈춘 것으로 처리
    bound = 0.6
    visual_factor = 2.0  # 슈팅 능력치는 100이 기본값인데 시각화 시에 실제 슈팅처럼 빠르도록 2배를 곱함
    max_receive_height = 2.0
    collision_threshold = 2.5

    def __init__(self, position=(0, 0, 0), home_team_id=0, away_team_id=1):
        self.position = np.array(position, dtype=np.float32)
        self.prev_position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.use_gravity = True

        self.owner_team_id = None      # 공 소유 팀 (None: 소유 없음)
        self.owner_player_id = None    # 공 소유 선수 (None: 소유 없음)
        self.pass_player_id = None     # 마지막으로 패스한 선수 (트래핑 관련)
        self.last_touch_team = None

        self.collision_ignore_id = None
        self.is_controlled = True
        self.control_timer = 0.0

        self.possession_state = "held"
        self.motion_state = "on_ground"

        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.restart_event = None
        self.restart_timer = 0

        self.skill_move = False  # 개인기 성공률 조정

    def update(self, dt=1.0):
        self.prev_position = self.position.copy()

        if self.restart_event:
            self.restart_timer -= dt
            if self.restart_timer <= 0:
                self.reset_for_restart(self.restart_event)
                self.restart_event = None
                self.restart_timer = 0
            return

        if not self.is_controlled and self.owner_player_id:
            self.control_timer -= dt
            if self.control_timer < 0:
                self.is_controlled = True

        if self.owner_player_id is None:
            self.position += self.velocity * dt

            if self.position[2] == 0.0:
                self.velocity[:2] *= self.friction ** dt
                if np.linalg.norm(self.velocity[:2]) < self.min_speed * self.visual_factor:
                    self.velocity[:2] = np.zeros(2, dtype=np.float32)

            if self.use_gravity and self.position[2] > 0.0:
                self.velocity[2] -= self.gravity * dt

            elif np.linalg.norm(self.velocity) < 0.1:
                self.use_gravity = True

            # 바닥에 닿으면 바운스
            if self.position[2] < 0.0:
                self.position[2] = 0.0
                if self.velocity[2] < 0.0:
                    self.velocity[2] *= -self.bound

            if np.linalg.norm(self.velocity[:2]) < self.min_speed:
                self.velocity[:2] = np.zeros(2, dtype=np.float32)
        else:
            self.use_gravity = True

        # possession_state
        self.possession_state = "held" if self.owner_player_id is not None else "free"

        # motion_state
        if self.position[2] > 0.01:
            self.motion_state = "in_air"
        elif np.linalg.norm(self.velocity[:2]) > self.min_speed:
            self.motion_state = "rolling"
        else:
            self.motion_state = "on_ground"

        # 재개 상황 체크
        event = self.check_restart_event()
        if event and not self.restart_event:
            print("재개 상황:", event)
            self.restart_event = event
            self.restart_timer = 2.0

    def kick(self, player, power, direction, shoot_max_power=1000.0):  # direction: 정규화된 2D 벡터
        self.collision_ignore_id = player.player_id
        self.pass_player_id = player.player_id
        self.is_controlled = False

        direction = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        self.velocity = min(power, shoot_max_power) * direction * self.visual_factor  # 적절히 조정 (슛파워 기본값 100)
        self.owner_team_id = None
        self.owner_player_id = None
        if player.team_id is not None:
            self.last_touch_team = player.team_id

        player.dribbling = False
        player.dash_dribble = False
        player.ball = None

    def control_inaccuracy_factor(self, base=1.3):
        """
        control_timer: 트래핑 후 경과 시간 (초 단위)
        base: 감소 비율
        return: 0.0 ~ 1.0 사이의 control 부정확도 계수
        """
        # control timer
        inaccuracy_value = np.clip(base ** self.control_timer, 0.0, 3.0)

        # height
        if self.position[2] < 0.01:
            inaccuracy_value *= 0.8
        elif self.position[2] < 0.35:
            inaccuracy_value *= 1.0
        else:
            inaccuracy_value *= 1.5

        return inaccuracy_value

    def pass_to(self, player_from, player_to, target_offset=(0, 0), power=100.0, is_lob=False, lob_vector=0.3):
        """
        player_from: 패스하는 선수
        player_to: 패스받는 선수
        target_offset: 받는 선수 위치로부터 변동값 (앞에 스루패스를 줄지, 뒤로 안전하게 줄지)
        power: 패스 세기
        is_lob: True면 z 방향도 포함된 방향 계산
        lob_vector: 로빙 패스 시 목표 z 방향 값
        """
        target_xy = np.array(player_to.position[:2]) + np.array(target_offset)
        direction_xy = target_xy - self.position[:2]
        norm_xy = np.linalg.norm(direction_xy)
        if norm_xy == 0:
            return
        direction_xy /= norm_xy

        noise_strength = self.calculate_pass_noise(player_from, power, norm_xy)
        raw_noise = np.random.normal(0, noise_strength)
        angle_noise = np.clip(raw_noise, -noise_strength, noise_strength)
        cos_a = np.cos(angle_noise)
        sin_a = np.sin(angle_noise)

        dx, dy = direction_xy[0], direction_xy[1]
        new_xy = np.array([
            dx * cos_a - dy * sin_a,
            dx * sin_a + dy * cos_a
        ])

        if is_lob:
            self.use_gravity = True
        z_dir = lob_vector if is_lob else 0.0
        xy_size = 1 - z_dir if z_dir < 1 else 0.1
        direction_3d = np.array([new_xy[0] * xy_size, new_xy[1] * xy_size, z_dir])
        self.kick(player_from, power, direction_3d)

    def shoot(self, player=None, goal_pos=(0, 0, 0), power=100.0):
        """
        :param goal_pos:(x,y,z)
        :param power: 슈팅 세기
        """
        direction = np.array(goal_pos) - self.position
        norm = np.linalg.norm(direction)

        noise_strength = self.calculate_shoot_noise(player, power, norm)
        raw_noise = np.random.normal(0, noise_strength)
        angle_noise = np.clip(raw_noise, -noise_strength, noise_strength)
        cos_a = np.cos(angle_noise)
        sin_a = np.sin(angle_noise)

        if norm == 0:
            return
        direction /= norm

        dx, dy = direction[0], direction[1]
        new_dir = np.array([
            dx * cos_a - dy * sin_a,
            dx * sin_a + dy * cos_a,
            direction[2]
        ])

        self.use_gravity = False
        self.kick(player, power, new_dir, shoot_max_power=player.max_power)

    def set_owner(self, player):
        self.owner_team_id = player.team_id
        self.owner_player_id = player.player_id
        self.last_touch_team = player.team_id
        player.ball = self

    def release(self):
        self.owner_team_id = None
        self.owner_player_id = None

    def is_free(self):
        return self.owner_player_id is None

    def check_collision_with_players(self, players, dt):
        """
        player_positions: list of (team_id, player_id, position)
        """
        if self.owner_player_id is not None:
            return

        if self.control_timer > 0:
            self.control_timer -= dt
            return

        for receive_player in players:
            team_id, player_id, pos = receive_player.team_id, receive_player.player_id, receive_player.position
            if player_id != self.collision_ignore_id:
                player_pos = np.array(pos[:2])
                prev = self.prev_position[:2]
                curr = self.position[:2]

                line_vec = curr - prev
                point_vec = player_pos - prev
                line_len = np.linalg.norm(line_vec)

                if line_len == 0:
                    closest_point = curr
                else:
                    proj = np.dot(point_vec, line_vec) / (line_len ** 2)
                    proj = np.clip(proj, 0, 1)
                    closest_point = prev + proj * line_vec

                distance = np.linalg.norm(player_pos - closest_point)

                if distance <= self.collision_threshold and self.position[2] < self.max_receive_height:
                    self.use_gravity = True
                    # 튕겨나오는 경우
                    if np.linalg.norm(self.velocity) > self.deflection_threshold * self.visual_factor:
                        self.deflect_off_player(receive_player)
                        self.collision_ignore_id = player_id
                    # 소유하는 경우
                    else:
                        self.set_owner(receive_player)
                        self.collision_ignore_id = None
                        receive_player.update_move_intensity("IDLE")

                        pass_player_id = self.pass_player_id if self.pass_player_id else self.owner_player_id
                        pass_player = [p_ for p_ in players if p_.player_id == pass_player_id][0]
                        self.compute_trapping_delay(
                            pass_player.pass_quality, receive_player.ball_control_quality, receive_player.stamina
                        )
                        self.velocity = np.zeros(3, dtype=np.float32)  # 소유 상태면 정지
                    break

    def deflect_off_player(self, player, randomness=0.05):
        self.last_touch_team = player.team_id

        ball_pos = self.position[:2]
        ball_vel = self.velocity[:2]
        player_pos = np.array(player.position[:2])

        incoming_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-6)
        normal = ball_pos - player_pos
        normal = normal / (np.linalg.norm(normal) + 1e-6)

        # 반사 벡터 계산 (R = D - 2(D · N)N)
        reflect_dir = incoming_dir - 2 * np.dot(incoming_dir, normal) * normal

        # 빗겨 맞을수록 반사 유지, 정면일수록 튕겨남
        angle = np.dot(incoming_dir, normal)  # -1 = 정면, 0 = 직각

        # noise 스케일: 정면이면 반사 유지, 빗겨갈수록 흔들기
        scaled_randomness = randomness * (1 - abs(angle))  # angle=-1 → 0, angle=0 → max randomness
        noise = np.random.normal(0, scaled_randomness, size=2)

        new_dir = reflect_dir + noise
        new_dir = new_dir / np.linalg.norm(new_dir)

        speed = np.linalg.norm(self.velocity)
        self.velocity = np.array([new_dir[0], new_dir[1], self.velocity[2] * 0.3]) * speed * 0.5

    def compute_trapping_delay(self, pass_quality=100.0, ball_control_quality=100.0, stamina=100.0, max_delay=3.0, min_delay=0.2):
        """
        ball_control_quality: 해당 선수의 볼컨트롤 능력치 (예: 1~100)
        """
        velocity_magnitude = np.linalg.norm(self.velocity) / 2.0
        velocity_factor = velocity_magnitude / 100.0
        control_factor = ball_control_quality / 100.0  # 0~1 스케일
        pass_factor = pass_quality / 100.0  # 0~1 스케일
        stamina_factor = stamina / 100.0

        # 속도 대비 제어력 → 큰 속도 + 낮은 제어력 + 낮은 패스 퀄리티 = 긴 딜레이
        raw_delay = velocity_factor / (control_factor + 1e-4) / (pass_factor + 1e-4) / (stamina_factor + 1e-4)  # 0 divide 방지

        if self.position[2] < 0.1:
            z_factor = 0.5
        elif self.position[2] < 0.3:
            z_factor = 1.3
        elif self.position[2] < 1.0:
            z_factor = 1.7
        else:
            z_factor = 2.4

        # 스케일 조정 후 클리핑
        delay = np.clip((raw_delay * 10 * z_factor), min_delay, max_delay)
        self.control_timer = delay
        return

    def check_goal(self, side="left"):
        """
        ball_pos: np.array([x, y, z])
        side: 'left' or 'right'
        """
        goal_y_center = Field.REAL_HEIGHT / 2
        goal_y_min = goal_y_center - Field.GOAL_WIDTH / 2
        goal_y_max = goal_y_center + Field.GOAL_WIDTH / 2

        goal_z_max = Field.GOAL_HEIGHT

        if side == 'left':
            is_x_in = self.position[0] <= 0.0
        else:
            is_x_in = self.position[0] >= Field.REAL_WIDTH

        is_y_in = goal_y_min <= self.position[1] <= goal_y_max
        is_z_in = 0.0 <= self.position[2] <= goal_z_max

        return is_x_in and is_y_in and is_z_in

    def check_restart_event(self):
        x, y, _ = self.position

        # 골라인을 벗어났는지 확인
        if x < 0 or x > Field.REAL_WIDTH:
            side = "left" if x < 0 else "right"
            if self.check_goal(side=side):
                return "goal"
            elif (side == "left" and self.last_touch_team == self.away_team_id) or (side == "right" and self.last_touch_team == self.home_team_id):
                return f"goal_kick_{side}"
            else:
                return f"corner_kick_{side}"

        # 사이드라인을 벗어났는지 확인
        if y < 0 or y > Field.REAL_HEIGHT:
            return "throw_in"

        return None

    def reset_for_restart(self, event):
        """
        event: 'goal_kick_left', 'goal_kick_right', 'goal',
               'corner_kick_left', 'corner_kick_right', 'throw_in'
        """
        self.restart_event = event
        if event.startswith("goal_kick"):
            side = "left" if event.endswith("left") else "right"
            x = 5.5 if side == "left" else Field.REAL_WIDTH - 5.5
            y = Field.REAL_HEIGHT / 2 - 3 if self.position[1] < Field.REAL_HEIGHT / 2 else Field.REAL_HEIGHT / 2 + 3
        elif event.startswith("corner_kick"):
            side = "left" if event.endswith("left") else "right"
            x = 0.0 if side == "left" else Field.REAL_WIDTH
            y = 0.0 if self.position[1] < Field.REAL_HEIGHT / 2 else Field.REAL_HEIGHT
        elif event == "throw_in":
            # 사이드라인 기준 가장 가까운 y 경계에 위치
            x = np.clip(self.position[0], 0, Field.REAL_WIDTH)
            y = 0.1 if self.position[1] < 0 else Field.REAL_HEIGHT - 0.1
        elif event == "goal":
            x = Field.REAL_WIDTH / 2
            y = Field.REAL_HEIGHT / 2
        else:
            return

        self.position = np.array([x, y, 0], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.owner_team_id = None
        self.owner_player_id = None
        self.use_gravity = True

    def calculate_pass_noise(self, player, power, norm_xy):
        base_noise = 1.3

        accuracy = np.clip(player.pass_quality / 100, 0.01, 1.0)
        adjusted_accuracy_factor = (1.0 - accuracy) ** 3
        power_factor = np.clip(power / 100, 0.0, 1.0)
        adjusted_power_factor = power_factor ** 2
        distance_factor = np.clip(norm_xy / 10, 0, 20)
        control_factor = self.control_inaccuracy_factor()
        stamina_factor = player.stamina_system.inaccuracy_modifier("pass")

        noise_strength = base_noise * adjusted_accuracy_factor * adjusted_power_factor * distance_factor * control_factor * stamina_factor
        return noise_strength

    def calculate_shoot_noise(self, player, power, norm):
        base_noise = 0.45

        accuracy = np.clip(player.shoot_quality / 100, 0.01, 1.0)
        adjusted_accuracy_factor = (1.0 - accuracy) ** 2
        power_factor = np.clip(power / 100, 0.0, 1.0)
        adjusted_power_factor = power_factor ** 2
        distance_factor = np.clip(norm / 10, 0, 20)
        adjusted_distance_factor = distance_factor ** 2
        stamina_factor = player.stamina_system.inaccuracy_modifier("shoot")

        noise_strength = base_noise * adjusted_accuracy_factor * adjusted_power_factor * adjusted_distance_factor * stamina_factor
        return noise_strength
