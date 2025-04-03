from physics.player_version2 import Player


# 예제 실행
if __name__ == "__main__":
    # 선수 생성: player_id, 초기위치, 기본속력, 최대속력, 가속도, 스태미너
    player = Player(player_id=1, position=[50, 50], base_speed=10, max_speed=30, acceleration=5, stamina=100)

    # 행동 수행: 스프린트 후 대각선 이동
    player.perform_action("SPRINT")
    player.perform_action("MOVE_TOP_RIGHT")

    # 10 프레임 동안 업데이트 (각 프레임 0.1초)
    for i in range(5):
        player.update(1)
        print(player)
