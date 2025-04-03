import pygame
import sys
import numpy as np
from physics.ball_version2 import Ball
from physics.player import Player
from simulation.environment import Field

# 색상 정의
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Constants for real soccer field dimensions in meters
REAL_FIELD_LENGTH = 105  # meters
REAL_FIELD_WIDTH = 68  # meters

# Constants for pixel dimensions of the field
PIXEL_FIELD_WIDTH = 1000  # pixels
PIXEL_FIELD_HEIGHT = 700  # pixels
BASE_OUTSIDE_DEPTH = 50
DT = 1 / 60

# Conversion factors
meter_to_pixel_length = PIXEL_FIELD_WIDTH / REAL_FIELD_LENGTH
meter_to_pixel_width = PIXEL_FIELD_HEIGHT / REAL_FIELD_WIDTH

# 공 속성
BALL_RADIUS = 10
ball_pos = [PIXEL_FIELD_WIDTH // 2, PIXEL_FIELD_HEIGHT // 2]
ball_vel = [0, 0]  # x, y 방향 속도


def draw_field(screen):
    # 필드 배경
    screen.fill(GREEN)

    # 필드 경계선
    pygame.draw.rect(screen, WHITE, (50, 50, PIXEL_FIELD_WIDTH - 100, PIXEL_FIELD_HEIGHT - 100), 5)

    # 중앙선
    pygame.draw.line(screen, WHITE, (PIXEL_FIELD_WIDTH // 2, 50), (PIXEL_FIELD_WIDTH // 2, PIXEL_FIELD_HEIGHT - 50), 5)

    # 센터 서클
    pygame.draw.circle(screen, WHITE, (PIXEL_FIELD_WIDTH // 2, PIXEL_FIELD_HEIGHT // 2), 70, 5)

    # 페널티 박스
    pygame.draw.rect(screen, WHITE, (50, PIXEL_FIELD_HEIGHT // 2 - 100, 100, 200), 5)
    pygame.draw.rect(screen, WHITE, (PIXEL_FIELD_WIDTH - 150, PIXEL_FIELD_HEIGHT // 2 - 100, 100, 200), 5)

    # 골대
    pygame.draw.rect(screen, WHITE, (50, PIXEL_FIELD_HEIGHT // 2 - 50, 40, 100), 5)
    pygame.draw.rect(screen, WHITE, (PIXEL_FIELD_WIDTH - 50 - 40, PIXEL_FIELD_HEIGHT // 2 - 50, 40, 100), 5)


def draw_ball(screen, ball):
    x, y = int(ball.position[0]), int(ball.position[1])
    pygame.draw.circle(screen, WHITE, (x, y), BALL_RADIUS)


def draw_players(screen, players):
    for player in players:
        team_color = (0, 0, 255) if player.team_id == 0 else (255, 0, 0)

        x = int((player.position[0] / Field.REAL_WIDTH) * PIXEL_FIELD_WIDTH) + BASE_OUTSIDE_DEPTH
        y = int((player.position[1] / Field.REAL_HEIGHT) * PIXEL_FIELD_HEIGHT) + BASE_OUTSIDE_DEPTH
        pygame.draw.circle(screen, team_color, (x, y), 12)


def generate_formation(team_id, is_left=True):
    formation = []

    # 기준 좌표 (미터 단위)
    x_base = 30 if is_left else Field.REAL_WIDTH - 30
    direction = 1 if is_left else -1

    # 수비수 (4명)
    for i in [-12, -4, 4, 12]:
        x = x_base - 10 * direction
        y = Field.REAL_HEIGHT / 2 + i
        formation.append((x, y))

    # 미드필더 (4명)
    for i in [-12, -4, 4, 12]:
        x = x_base
        y = Field.REAL_HEIGHT / 2 + i
        formation.append((x, y))

    # 공격수 (2명)
    for i in [-4, 4]:
        x = x_base + 10 * direction
        y = Field.REAL_HEIGHT / 2 + i
        formation.append((x, y))

    # 골키퍼
    x = x_base - 18 * direction
    y = Field.REAL_HEIGHT / 2
    formation.append((x, y))

    players = []
    for pid, pos in enumerate(formation):
        players.append(Player(position=pos, team_id=team_id, player_id=pid))
    return players


def main():
    pygame.init()
    screen = pygame.display.set_mode((PIXEL_FIELD_WIDTH, PIXEL_FIELD_HEIGHT))
    pygame.display.set_caption('축구장 시각화')
    clock = pygame.time.Clock()

    ball = Ball(position=(400, 300, 0))
    ball.velocity = np.array([0, 0, 0], dtype=np.float32)  # 초기 속도 (pixel/sec)

    team_a_players = generate_formation(team_id=0, is_left=True)
    team_b_players = generate_formation(team_id=1, is_left=False)
    players = team_a_players + team_b_players

    print(players[0].position)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        ball.update(DT)
        for player in players:
            player.update(DT)

        draw_field(screen)
        draw_players(screen, players)
        draw_ball(screen, ball)
        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()
