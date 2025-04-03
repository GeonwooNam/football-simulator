import numpy as np
from physics.ball import Ball
from physics.player import Player
from simulation.environment import Field


class GameEngine:
    def __init__(self):
        self.home_team_id = 0
        self.away_team_id = 1
        self.ball = Ball(position=(Field.REAL_WIDTH / 2, Field.REAL_HEIGHT / 2, 0))
        self.players = self._create_players()
        self.score = {self.home_team_id: 0, self.away_team_id: 0}
        self.game_time = 0.0
        self.dt = 1 / 60

    def _create_players(self):
        team_a = self.generate_formation(0, is_left=True)
        team_b = self.generate_formation(0, is_left=False)
        return team_a + team_b

    def update(self):
        for player in self.players:
            player.update(self.dt)

        player_positions = [(p.team_id, p.player_id, p.position) for p in self.players]
        self.ball.update(self.dt)
        self.ball.check_collision_with_players(player_positions, dt=self.dt)
        self.check_goal()
        self.game_time += self.dt

    def check_goal(self):
        if self.ball.check_goal("left"):
            self.score[self.away_team_id] += 1
            print("\u26bd️ GOAL! 팀 1 득점")
            self.reset_ball()
        elif self.ball.check_goal("right"):
            self.score[self.home_team_id] += 1
            print("\u26bd️ GOAL! 팀 0 득점")
            self.reset_ball()

    def reset_ball(self):
        self.ball = Ball(position=(Field.REAL_WIDTH / 2, Field.REAL_HEIGHT / 2, 0))

    def get_state(self):
        return {
            "ball_position": self.ball.position,
            "ball_velocity": self.ball.velocity,
            "possession": (self.ball.owner_team, self.ball.owner_player),
            "score": self.score,
            "time": self.game_time,
        }

    @staticmethod
    def generate_formation(self, team_id, is_left=True):
        # TODO. 일단은 4-4-2만 구현했는데 추후 확장 가능성?

        formation = []

        unit_x = 13
        unit_y = 13

        x_base = unit_x * 0.2 if is_left else Field.REAL_WIDTH - unit_x * 0.2
        direction = 1 if is_left else -1

        for i in [-unit_x * 1.5, -unit_x * 0.5, unit_x * 0.5, unit_x * 1.5]:
            x = x_base + unit_y * direction
            y = Field.REAL_HEIGHT / 2 + i
            formation.append((x, y))

        for i in [-unit_x * 1.5, -unit_x * 0.5, unit_x * 0.5, unit_x * 1.5]:
            x = x_base + unit_y * direction * 2
            y = Field.REAL_HEIGHT / 2 + i
            formation.append((x, y))

        for i in [-unit_x * 0.5, unit_x * 0.5]:
            x = x_base + unit_y * direction * 3
            y = Field.REAL_HEIGHT / 2 + i
            formation.append((x, y))

        x = x_base
        y = Field.REAL_HEIGHT / 2
        formation.append((x, y))

        players = [Player(position=pos, team_id=team_id, player_id=i) for i, pos in enumerate(formation)]
        return players
