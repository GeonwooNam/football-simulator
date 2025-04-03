import numpy as np

DT = 1 / 60  # 고정 시간 간격


# 1. 움직임 및 위치 제어
# 1-1. 목표 위치로 이동



# 1-2. 대상과 일정 거리 유지
def adjust_distance_from(player, target, desired_distance=1.5):
    delta = target.position[:2] - player.position[:2]
    distance = np.linalg.norm(delta)
    if distance < 1e-4:
        return
    direction = delta / distance
    move_needed = distance - desired_distance
    if abs(move_needed) > 0.1:  # 허용 오차
        player.position[:2] += player.velocity * DT
    else:
        player.velocity *= 0  # 멈추기
        player.speed = 0.0


# 1-3. 대상 주위를 곡선으로 돌며 접근
# def circle_around(player, target, block_pos, radius_adjust=1.0, move_intensity="LOW_INTENSITY_RUNNING"):
def calculate_curve(player, target, block_pos, ry_adjust=1.0):
    player.curve_motion = {
        "center": (50, 50),
        "rx": 20,
        "ry": 10,
        "angle_offset": np.radians(30),  # 30도 회전된 타원
        "clockwise": False,
        "angular_speed": 0.04,
        "angle": 0.0
    }


def update_curved_run(player, DT, move_intensity="LOW_INTENSITY_RUNNING"):
    cm = player.curve_motion
    if cm is None:
        return  # 움직일 정보가 없으면 종료

    # 각도 업데이트
    delta_theta = (-cm["angular_speed"] if cm["clockwise"] else cm["angular_speed"]) * DT
    cm["angle"] += delta_theta

    # 타원 기준 좌표 계산
    raw_x = cm["rx"] * np.cos(cm["angle"])
    raw_y = cm["ry"] * np.sin(cm["angle"])

    # 타원 방향 회전 (angle_offset 만큼 회전)
    cos_a = np.cos(cm["angle_offset"])
    sin_a = np.sin(cm["angle_offset"])

    rotated_x = cos_a * raw_x - sin_a * raw_y
    rotated_y = sin_a * raw_x + cos_a * raw_y

    # 최종 위치 = 중심점 + 회전 좌표
    final_x = cm["center"][0] + rotated_x
    final_y = cm["center"][1] + rotated_y
    # print(final_x, final_y)

    move_towards(player, (final_x, final_y), move_intensity=move_intensity)


"""
def angle_between_vectors(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product)


def circle_around(player, target, block_pos, radius_adjust=1.0, move_intensity="LOW_INTENSITY_RUNNING"):
    '''
    상대를 곡선으로 돌며 접근하되, block_pos(패스 차단 위치)를 막는 방향으로 회전 경로 생성
    radius_adjust: 곡률 조절 (1.0 = 기본)
    '''
    # 1. 대상에서 수비자 방향
    to_target = target.position[:2] - player.position[:2]
    base_angle = np.arctan2(to_target[1], to_target[0])

    # 2. 대상에서 block_pos로 향하는 벡터
    to_block = block_pos[:2] - target.position[:2]
    block_angle = np.arctan2(to_block[1], to_block[0])

    # 3. 두 각도의 차이 → 차단하려는 방향과의 중간 각도 생성
    angle_diff = angle_between_vectors(to_target, to_block)
    direction = np.sign(np.cross(to_target, to_block))  # +1: counter-clockwise, -1: clockwise
    curvature_angle = direction * angle_diff * 0.5  # 중간 회전 각도

    # 4. 반지름 = block_pos와 target 간 거리 × 조절 인자
    radius = np.linalg.norm(to_block) * radius_adjust
    tangent_angle = base_angle + curvature_angle

    # 5. 회전 목표 위치 계산
    offset_pos = target.position[:2] + radius * np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

    # 6. 이동
    move_towards(player, offset_pos, move_intensity=move_intensity)
"""


# 1-4. 멈추기
def hold_position(player):
    player.velocity *= 0.2  # 점차 멈추게


# 1-5. 전술적으로 지정된 위치로 복귀
def reposition_to(player, target_pos, stop_tolerance=0.3, move_intensity="LOW_INTENSITY_RUNNING"):
    delta = target_pos[:2] - player.position[:2]
    distance = np.linalg.norm(delta)
    if distance > stop_tolerance:
        move_towards(player, target_pos, move_intensity=move_intensity)
    else:
        hold_position(player)
