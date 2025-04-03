class StaminaSystem:
    unit = 0.042  # 이걸로 증감량 크기 조절

    def __init__(self, max_stamina=100.0):
        self.stamina = max_stamina
        self.max_stamina = max_stamina

    def update(self, player, delta_time):
        update_value = 0.0
        # 회복 조건
        if player.move_intensity == "IDLE":
            update_value += 1.0
        elif player.move_intensity == "WALK":
            update_value += 0.5

        else:  # 소모 조건

            # 러닝 강도
            if player.move_intensity == "LOW_INTENSITY_RUNNING":
                update_value -= 1.0
            elif player.move_intensity == "HIGH_INTENSITY_RUNNING":
                update_value -= 2.0
            elif player.move_intensity == "SPRINT":
                update_value -= 5.0

            # 드리블
            if player.move_intensity != "SPRINT":
                if player.dribbling:
                    update_value -= 1.0
                elif player.dash_dribble:
                    update_value -= 2.0

            # 개인기
            if player.skill_move:
                update_value -= 1.0

        if update_value == 0:
            pass
        elif update_value > 0:
            self.recover(update_value * self.unit * delta_time)
        else:
            self.consume(update_value * self.unit * delta_time)

    def consume(self, amount):
        self.stamina = max(0.0, self.stamina - abs(amount))

    def recover(self, amount):
        self.stamina = min(self.max_stamina, self.stamina + abs(amount))

    def inaccuracy_modifier(self, action=None):  # stamina에 따른 정확도 보정 계수
        action = action.lower()
        action_list = ["pass", "shoot", "dribble", "skill_move", "tackle", "intercept"]
        if action not in action_list:
            return

        base_max_stamina = 100.0
        ratio = min(self.stamina / base_max_stamina, 1.0)
        inaccuracy_factor = (1.0 - ratio) ** 2

        # TODO. 가중치 조정 필요
        action_weights = {
            "pass": 0.7,
            "shoot": 1.4,
            "dribble": 1.0,
            "skill_move": 1.5,
            "tackle": 1.0,
            "intercept": 0.9,
        }

        weight = action_weights.get(action, 1.0)
        return inaccuracy_factor * weight
