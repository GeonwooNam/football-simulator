import numpy as np
from simulation.environment import Field


class Ball:
    gravity = 9.8  # TODO. 값 조정
    shoot_gravity = 2.0
    deflection_threshold = 70.0  # 맞고 굴절되는 공 최소 속도
    visual_factor = 2.0  # 슈팅 능력치는 100이 기본값인데 시각화 시에 실제 슈팅처럼 빠르도록 2배를 곱함

    def __init__(self, position=(0, 0, 0), friction=0.96, min_speed=0.01):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.friction = friction
        self.min_speed = min_speed  # 일정 이하 속도는 멈춘 것으로 처리
        self.use_gravity = True

        self.owner_team = None      # 공 소유 팀 (None: 소유 없음)
        self.owner_player = None    # 공 소유 선수 (None: 소유 없음)
        self.collision_cooldown = 0

        self.possession_state = "held"
        self.motion_state = "on_ground"

    def update(self, dt=1.0):
        if self.owner_player is None:
            self.position += self.velocity * dt
            self.velocity[:2] *= self.friction ** dt

            if self.use_gravity:
                self.velocity[2] -= self.gravity * dt

            elif np.linalg.norm(self.velocity) < 0.1:
                self.use_gravity = True

            # 바닥에 닿으면 바운스
            if self.position[2] < 0.0:
                self.position[2] = 0.0
                if self.velocity[2] < 0:
                    self.velocity[2] *= -0.6

            if np.linalg.norm(self.velocity[:2]) < self.min_speed:
                self.velocity[:2] = np.zeros(2, dtype=np.float32)
        else:
            self.use_gravity = True

        # possession_state
        self.possession_state = "held" if self.owner_player is not None else "free"

        # motion_state
        if self.position[2] > 0.01:
            self.motion_state = "in_air"
        elif np.linalg.norm(self.velocity[:2]) > self.min_speed:
            self.motion_state = "rolling"
        else:
            self.motion_state = "on_ground"

    def kick(self, power, direction, shoot_max_power=1000.0):  # direction: 정규화된 2D 벡터
        direction = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        self.velocity = min(power, shoot_max_power) * direction * self.visual_factor  # 적절히 조정 (슛파워 기본값 100)
        self.owner_team = None
        self.owner_player = None

    def pass_to(self, target_pos, power=100.0, pass_accuracy_rating=100.0, is_lob=False, lob_vector=0.3, max_noise=0.45):
        """
        target_pos: (x, y)
        power: 패스 세기
        pass_accuracy_rating: 패스 정확도 능력치
        is_lob: True면 z 방향도 포함된 방향 계산
        lob_vector: 로빙 패스 시 목표 z 방향 값
        max_noise: 난수 최대값
        """
        target_xy = np.array(target_pos[:2], dtype=np.float32)
        direction_xy = target_xy - self.position[:2]
        norm_xy = np.linalg.norm(direction_xy)
        if norm_xy == 0:
            return
        direction_xy /= norm_xy

        accuracy = np.clip(pass_accuracy_rating / 100, 0.01, 1.0)
        adjusted_accuracy_factor = (1.0 - accuracy) ** 2
        power_factor = np.clip(power / 100, 0.0, 1.0)
        adjusted_power_factor = power_factor ** 2
        distance_factor = np.clip(norm_xy / 10, 0, 20)
        adjusted_distance_factor = distance_factor ** 2

        noise_strength = adjusted_accuracy_factor * adjusted_power_factor * adjusted_distance_factor * max_noise
        raw_noise = np.random.normal(0, noise_strength)
        angle_noise = np.clip(raw_noise, -noise_strength, noise_strength)
        cos_a = np.cos(angle_noise)
        sin_a = np.sin(angle_noise)

        dx, dy = direction_xy[0], direction_xy[1]
        new_xy = np.array([
            dx * cos_a - dy * sin_a,
            dx * sin_a + dy * cos_a
        ])

        z_dir = lob_vector if is_lob else 0.0
        direction_3d = np.array([new_xy[0], new_xy[1], z_dir])
        self.kick(power, direction_3d)

    def shoot(self, goal_pos, power=100.0, shoot_accuracy_rating=100.0, player_max_power=100.0, max_noise=0.45):
        """
        :param goal_pos:(x,y,z)
        :param power: 슈팅 세기
        :param shoot_accuracy_rating: 슈팅 정확도 (난수 고려시 적용)
        :param player_max_power: 최대 슈팅 세기
        :param max_noise: 난수 최대값
        """
        direction = np.array(goal_pos) - self.position
        norm = np.linalg.norm(direction)

        accuracy = np.clip(shoot_accuracy_rating / 100, 0.01, 1.0)
        adjusted_accuracy_factor = (1.0 - accuracy) ** 2
        power_factor = np.clip(power / 100, 0.0, 1.0)
        adjusted_power_factor = power_factor ** 2
        distance_factor = np.clip(norm / 10, 0, 20)
        adjusted_distance_factor = distance_factor ** 2

        noise_strength = adjusted_accuracy_factor * adjusted_power_factor * adjusted_distance_factor * max_noise
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
        self.kick(power, new_dir, shoot_max_power=player_max_power)

    def set_owner(self, team_id, player_id):
        self.owner_team = team_id
        self.owner_player = player_id
        self.velocity = np.zeros(3, dtype=np.float32)  # 소유 상태면 정지

    def release(self):
        self.owner_team = None
        self.owner_player = None

    def is_free(self):
        return self.owner_player is None

    def check_collision_with_players(self, player_positions, dt, pass_dict=None, threshold=3.0):
        """
        player_positions: list of (team_id, player_id, position)
        threshold: 충돌 인식 반경
        """
        if pass_dict is None:
            pass_dict = {}
        if self.owner_player is not None:
            return

        if self.collision_cooldown > 0:
            self.collision_cooldown -= dt
            return

        for team_id, player_id, pos in player_positions:
            dist = np.linalg.norm(np.array(pos) - self.position)
            if dist < threshold:
                ball_to_player = np.array(pos[:2]) - self.position[:2]
                velocity_dir = self.velocity[:2]
                if player_id == 2:
                    print(f"  {ball_to_player}, {velocity_dir}")

                # 방향이 수비수를 향하고 있는지 확인 (내적 > 0이면 같은 방향)
                approaching = np.dot(ball_to_player, velocity_dir) > 0
                if player_id == 2:
                    print(np.dot(ball_to_player, velocity_dir), dist)
                if approaching:
                    print(dist, self.velocity)

                    if np.linalg.norm(self.velocity) > self.deflection_threshold * self.visual_factor:
                        self.deflect_off_player(pos)
                    else:
                        self.set_owner(team_id, player_id)
                        if pass_dict is None:
                            self.collision_cooldown = 1.0
                        elif "pass_quality" in pass_dict and "receiver_control" in pass_dict:
                            self.collision_cooldown = self.compute_collision_delay(pass_dict["pass_quality"], pass_dict["receiver_control"])
                    break

    def deflect_off_player(self, player_pos, randomness=0.05):
        ball_pos = self.position[:2]
        ball_vel = self.velocity[:2]
        player_pos = np.array(player_pos[:2])

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

    def compute_collision_delay(self, pass_quality, receiver_control, max_delay=3.0, min_delay=0.2):
        """
        receiver_control: 해당 선수의 볼컨트롤 능력치 (예: 1~100)
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        control_factor = receiver_control / 100.0  # 0~1 스케일
        pass_factor = pass_quality / 100.0  # 0~1 스케일

        # 속도 대비 제어력 → 큰 속도 + 낮은 제어력 + 낮은 패스 퀄리티 = 긴 딜레이
        raw_delay = velocity_magnitude / (control_factor + 1e-4) / (pass_factor + 1e-4)  # 0 divide 방지

        # 스케일 조정 후 클리핑
        delay = np.clip((raw_delay * 0.5), min_delay, max_delay)  # TODO. 0.5 대신 적절값 조정
        return delay

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

