class Field:
    REAL_WIDTH = 105.0  # 실제 경기장 너비 (미터)
    REAL_HEIGHT = 68.0  # 실제 경기장 높이 (미터)

    NORMALIZED_WIDTH = 1.0  # 정규화된 너비
    NORMALIZED_HEIGHT = 1.0  # 정규화된 높이

    GOAL_WIDTH = 7.32  # 골대 가로 (y 방향)
    GOAL_HEIGHT = 2.44  # 골대 높이 (z 방향)

    GOAL_DEPTH = 2.0

    @staticmethod
    def to_normalized(position):
        """
        실제 좌표를 정규화된 좌표 (0~1)로 변환
        """
        return [position[0] / Field.REAL_WIDTH, position[1] / Field.REAL_HEIGHT]

    @staticmethod
    def to_real(position):
        """
        정규화된 좌표를 실제 좌표 (미터 단위)로 변환
        """
        return [position[0] * Field.REAL_WIDTH, position[1] * Field.REAL_HEIGHT]


# 테스트
"""
real_pos = [52.5, 34.0]  # 경기장 중앙
norm_pos = Field.to_normalized(real_pos)
print("정규화된 좌표:", norm_pos)

real_pos_converted = Field.to_real(norm_pos)
print("다시 실제 크기로 변환:", real_pos_converted)
"""
