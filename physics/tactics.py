import math
import numpy as np
from physics.ball import Ball
from physics.player import Player
from simulation.environment import Field

class TacticsManager:
    # 포지션
    role_list = ['ST', 'LW', 'RW', 'CAM', 'LM', 'RM', 'CM', 'CDM', 'CB', 'LB', 'RB', 'GK']
    
    # 라인 폭 조절
    width_mode_list = ["TOUCH_LINE", "WIDE", "SEMI_WIDE", "BASE", "NARROW"]
    WIDTH_MODE_RATIO = {
        "WIDE": 0.25,
        "SEMI_WIDE": 0.7,
        "NARROW": 1.5
    }

    # 라인 높이 조절
    LINE_HEIGHT_MODE = {
        "VERY_HIGH": 85,
        "HIGH": 70,  # 수비라인 기준 x좌표
        "MEDIUM": 55,
        "LOW": 40,
        "VERY_LOW": 25,
    }

    # 선수 라인 이동 (drop/advance)
    horizontal_mode_list = ["LEFT", "MIDDLE", "MIDDLE_RIGHT", "RIGHT", "BY_IDX"]

    def __init__(self, team_id, players, is_home=True, base_formation=(4, 4, 2), attack_formation=None, defense_formation=None):
        if len(players) != 11 or \
                sum(base_formation) != 10 or \
                (attack_formation and sum(attack_formation) != 10) or \
                (defense_formation and sum(defense_formation) != 10):
            raise ValueError

        self.team_id = team_id
        self.is_home = is_home
        self.players = players
        self.base_formation = base_formation
        self.is_attack = True
        self.attack_formation = attack_formation
        self.defense_formation = defense_formation
        self.current_formation = base_formation

        self.defensive_line_pid = [p.player_id for p in players if p.role in ["CB", "LB", "RB"]]
        self.holding_line_pid = [p.player_id for p in players if p.role in ["CDM", "CM"]]
        self.attacking_line_pid = [p.player_id for p in players if p.role in ["LW", "RW", "LM", "RM", "CAM"]]
        self.forward_line_pid = [p.player_id for p in players if p.role in ["ST"]]

        self.line_pid_map = {
            "defensive": self.defensive_line_pid,
            "holding": self.holding_line_pid,
            "attacking": self.attacking_line_pid,
            "forward": self.forward_line_pid
        }

        self.drop_pid = []

    # 포지셔닝
    def positioning_by_current_formation(self, starting=False):
        self.players.sort(key=lambda p: p.position[0], reverse=not self.is_home)
        accum_v = 0
        line_count = len(self.current_formation)
        for idx, line_n in enumerate(self.current_formation):
            target_players = self.players[accum_v:accum_v+line_n]
            accum_v += line_n
            if starting:
                target_x = Field.REAL_WIDTH / 2 / (line_count+1) * (idx+1)
                if not self.is_home:
                    target_x = Field.REAL_WIDTH - target_x
            else:
                target_x = None
            self.line_width_control(target_players, mode="BASE", target_x=target_x, move_intensity="SPRINT")

    # 폭 조절
    def line_width_control(self, target_players, mode="BASE", exclude_pid=None, target_x=None, move_intensity="LOW_INTENSITY_RUNNING"):
        if mode not in self.width_mode_list:
            return
        n_line = len(target_players)
        if target_x is None:
            x_list = [p.position[0] for p in target_players if p.player_id != exclude_pid]
            target_x = sum(x_list) / max(len(x_list), 1)
        target_players.sort(key=lambda p: p.position[1])
        if n_line <= 3:
            unit_width = Field.REAL_HEIGHT / 5
            start_y = Field.REAL_HEIGHT / 2 - unit_width * (n_line - 1) / 2
        else:
            unit_width = Field.REAL_HEIGHT / (n_line + 1)
            start_y = unit_width

        start_y, unit_width = self.width_multiplier(start_y, unit_width, mode, n_line)

        for i, p in enumerate(target_players):
            target_y = start_y + unit_width * i
            p.move_towards((target_x, target_y), move_intensity)

    def width_multiplier(self, start_y, unit_width, mode, n_line):
        if mode == "TOUCH_LINE":
            start_y = 0
            unit_width = Field.REAL_HEIGHT / max(n_line - 1, 1)
        elif mode != "BASE":
            start_y *= self.WIDTH_MODE_RATIO[mode]
            unit_width = (Field.REAL_HEIGHT - 2 * start_y) / max(n_line - 1, 1)

        return start_y, unit_width

    # 로테이션
    def position_rotate_by_pid(self, pid_list, move_intensity="LOW_INTENSITY_RUNNING"):
        if len(pid_list) < 2:
            return

        players = [p for pid in pid_list for p in self.players if p.player_id == pid]
        positions = [p.position[:2] for p in players]

        rotated = positions[1:] + [positions[0]]

        for player, new_pos in zip(players, rotated):
            player.move_towards(new_pos, move_intensity)

    # 선수 라인 상하 이동 (drop / advance)
    def drop_player(self, player, mode=None):
        self._move_player_line(player, direction="DOWN", mode=mode)

    def advance_player(self, player, mode=None):
        self._move_player_line(player, direction="UP", mode=mode)

    def _move_player_line(self, player, direction="DOWN", mode=None):
        mode_list = ["LEFT", "MIDDLE", "MIDDLE_RIGHT", "RIGHT", "BY_IDX"]

        # 1. 선수가 속한 라인 찾기
        line = player.offense_line if self.is_attack else player.defense_line
        formation = self.attack_formation if self.is_attack else self.defense_formation

        # 2. 라인 이동 및 포메이션 업데이트
        if self.is_attack:
            max_line = len(self.attack_formation)
        else:
            max_line = len(self.defense_formation)

        if (direction == "DOWN" and line < max_line - 1) or (direction == "UP" and line > 1):
            dir_value = 1 if direction == "DOWN" else -1
            formation = list(formation)  # 튜플 → 리스트 (수정 가능하게)
            formation[max_line - line] -= 1  # 기존 라인에서 한 명 빼고
            formation[max_line - (line + dir_value)] += 1  # 다음 라인으로 한 명 추가
            formation = tuple(formation)  # 다시 튜플로 저장
            line += dir_value
        else:
            return

        # 3. 위치 변경 및 폭 조절 (폭 조절 관리 인자 필요?)
        original_line_n = line - dir_value
        original_line = [
            p for p in self.players if (p.offense_line if self.is_attack else p.defense_line) == original_line_n
        ]
        new_line = [
            p for p in self.players if (p.offense_line if self.is_attack else p.defense_line) == line
        ]

        # TODO. 바뀐 라인에서 좌우 위치 조절

        self.line_width_control(original_line)
        self.line_width_control(new_line, exclude_pid=player.player_id)

    # 라인 좌우 이동
    def shift_line_horizontal(self, target_line_num, delta_y):
        target_line = [
            p for p in self.players if (p.offense_line if self.is_attack else p.defense_line) == target_line_num
        ]
        for player in target_line:
            player.move_towards((player.position[0], player.position[1] + delta_y))

    # TODO
    # 라인 상하 폭 조절
    def compress_lines(self, spacing_delta, backline_delta):
        # 라인별 중심 위치 조정
        # spacing_delta > 0: 넓히기 / < 0: 압축

        pass

    # 라인 높이 조절
    def control_line_height_by_mode(self, mode="BASE"):
        if self.is_attack or mode not in self.LINE_HEIGHT_MODE.keys():
            return

        defense_line_x = [p.position[0] for p in self.players if p.defense_line == len(self.defense_formation)]
        average_x = sum(defense_line_x) / len(defense_line_x)
        delta_x = self.LINE_HEIGHT_MODE[mode] - average_x
        self.control_line_height_by_delta(delta_x=delta_x)

    def control_line_height_by_delta(self, delta_x=0.0):
        if self.is_attack:
            return

        for player in self.players:
            player.move_towards((player.position[0] + delta_x, player.position[1]))

    def control_single_line_height_by_delta(self, line_num, delta_x=0.0):
        if self.is_attack:
            return

        target_line = [p for p in self.players if p.defense_line == line_num]
        for player in target_line:
            player.move_towards((player.position[0] + delta_x, player.position[1]))

    # 포메이션 변경
    def switch_formation(self):
        pass

