import pygame
import sys
import random
import numpy as np
from physics.ball import Ball
from physics.player import Player
from simulation.environment import Field

# 색상 정의
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

team_color_by_id = {}

# 픽셀 필드 크기
PIXEL_FIELD_WIDTH = 1000
PIXEL_FIELD_HEIGHT = 700
BASE_OUTSIDE_DEPTH = 50
PIXEL_GOAL_WIDTH = int(Field.GOAL_WIDTH / Field.REAL_HEIGHT * PIXEL_FIELD_HEIGHT)
DT = 1 / 60

# 변환 함수: 미터 → 픽셀


def meters_to_pixels(x_m, y_m):
    px = int((x_m / Field.REAL_WIDTH) * (PIXEL_FIELD_WIDTH - 2 * BASE_OUTSIDE_DEPTH)) + BASE_OUTSIDE_DEPTH
    py = int((y_m / Field.REAL_HEIGHT) * (PIXEL_FIELD_HEIGHT - 2 * BASE_OUTSIDE_DEPTH)) + BASE_OUTSIDE_DEPTH
    return px, py


def draw_field(screen):
    screen.fill(GREEN)
    # 1. 외곽
    pygame.draw.rect(screen, WHITE, (BASE_OUTSIDE_DEPTH, BASE_OUTSIDE_DEPTH, PIXEL_FIELD_WIDTH - 2 * BASE_OUTSIDE_DEPTH, PIXEL_FIELD_HEIGHT - 2 * BASE_OUTSIDE_DEPTH), 5)
    # 2. 중앙선
    pygame.draw.line(screen, WHITE, (PIXEL_FIELD_WIDTH // 2, BASE_OUTSIDE_DEPTH), (PIXEL_FIELD_WIDTH // 2, PIXEL_FIELD_HEIGHT - BASE_OUTSIDE_DEPTH), 5)
    # 3. 센터 서클
    pygame.draw.circle(screen, WHITE, (PIXEL_FIELD_WIDTH // 2, PIXEL_FIELD_HEIGHT // 2), 70, 5)
    # 4. 페널티 박스
    unit_size = 80
    pygame.draw.rect(screen, WHITE, (BASE_OUTSIDE_DEPTH, PIXEL_FIELD_HEIGHT // 2 - unit_size, unit_size, unit_size*2), 5)
    pygame.draw.rect(screen, WHITE,
                     (PIXEL_FIELD_WIDTH - BASE_OUTSIDE_DEPTH - unit_size, PIXEL_FIELD_HEIGHT // 2 - unit_size, unit_size, unit_size*2), 5)
    # 5. 골대 (정확한 크기 적용)
    goal_y = PIXEL_FIELD_HEIGHT // 2 - PIXEL_GOAL_WIDTH // 2
    pygame.draw.rect(screen, WHITE, (50, goal_y, 40, PIXEL_GOAL_WIDTH), 5)
    pygame.draw.rect(screen, WHITE, (PIXEL_FIELD_WIDTH - 50 - 40, goal_y, 40, PIXEL_GOAL_WIDTH), 5)


def draw_ball(screen, ball):
    x, y = meters_to_pixels(ball.position[0], ball.position[1])
    z = ball.position[2]
    radius = 8 + min(z, 2.0)/2.0 * (11.9 - 8)
    pygame.draw.circle(screen, WHITE, (x, y), int(radius))


def draw_players(screen, players):
    for player in players:
        if player.team_id in team_color_by_id:
            team_color = team_color_by_id[player.team_id]
        else:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            team_color_by_id[player.team_id] = color
            team_color = color
        x, y = meters_to_pixels(player.position[0], player.position[1])
        pygame.draw.circle(screen, team_color, (x, y), 12)


def main():
    pygame.init()
    screen = pygame.display.set_mode((PIXEL_FIELD_WIDTH, PIXEL_FIELD_HEIGHT))
    pygame.display.set_caption('축구장 시각화')
    clock = pygame.time.Clock()

    ball = Ball(position=(40, 34, 0), home_team_id=1, away_team_id=2)
    ball.velocity = np.array([0, 0, 0], dtype=np.float32)

    player_1 = Player(
        position=(10, 34, 0), team_id=1, player_id=1, max_speed=50.0, stamina=80.0,
        stat_dict={"pass": 80.0, "shoot": 80.0, "power": 200.0}
    )
    players = [player_1]

    goal_position = np.array([105, 34, 0], dtype=np.float32)
    frame = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if player_1.position[0] < 40 and np.linalg.norm(player_1.position - ball.position) > 2.0:
            player_1.update_running_info(move_direction="MOVE_RIGHT", move_intensity="SPRINT")
        elif player_1.position[0] < 90:
            player_1.update_dribble_info(move_direction="MOVE_RIGHT")

            if 30.0 < player_1.position[0] < 70.0:
                player_1.update_dribble_info(is_dash_dribble=False, move_intensity="HIGH_INTENSITY_RUNNING")
            else:
                player_1.update_dribble_info(is_dash_dribble=True, move_intensity="HIGH_INTENSITY_RUNNING")

        if player_1.position[0] >= 90.0 and player_1.able_to_shoot(ball):
            ball.shoot(player_1, goal_position, power=80.0)
            ball.release()
            player_1.update_running_info(move_intensity="IDLE")

        ball.update(DT)
        ball.check_collision_with_players(players, dt=DT)

        player_1.update(DT)

        draw_field(screen)
        draw_players(screen, players)
        draw_ball(screen, ball)
        pygame.display.flip()
        clock.tick(60)
        frame += 1


if __name__ == '__main__':
    main()
