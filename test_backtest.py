from backtest import trades

# 0주차 신호 → 1주차 시가 진입, 3주차 종가가 MA 아래 → 4주차 시가 청산
c = [10, 11, 12, 8, 9, 9]
ma = [9, 9, 9, 9, 9, 9]
assert trades(c, c, ma, [1, 0, 0, 0, 0, 0]) == [(1, 4, False)]
# 끝까지 MA 위면 마지막 주에서 미청산 처리
assert trades(c, [10, 11, 12, 12, 12, 12], ma, [1, 0, 0, 0, 0, 0]) == [(1, 5, True)]
# 마지막 주 신호는 진입 불가
assert trades(c, c, ma, [0, 0, 0, 0, 0, 1]) == []
print("ok")
