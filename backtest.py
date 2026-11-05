"""
40주선 돌파 전략 백테스트 (S&P 500 현재 구성종목, 2005~)
  진입: 신호 주 다음 주 시가
  청산: 주봉 종가가 40주선 아래로 마감하면 다음 주 시가
  포트폴리오: 보유 종목 동일비중, 매주 리밸런싱, 보유 종목 없으면 현금(0%)

사용법:
  python backtest.py            # 첫 실행 시 가격 다운로드 후 prices.pkl 캐시
  python backtest.py --refresh  # 캐시 무시하고 다시 다운로드
"""
import argparse
import os
import pandas as pd
import yfinance as yf
from ma40_screener import load_tickers, MA_LEN, SLOPE_WEEKS, VOL_AVG_WEEKS, VOL_MULT, MIN_DOLLAR_VOL

START = "2005-01-01"
COST = 0.001         # 편도 거래비용 0.1%
CACHE = "prices.pkl"


def get_prices(tickers, refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return pd.read_pickle(CACHE)
    data = yf.download(tickers + ["SPY"], start=START, auto_adjust=True,
                       group_by="ticker", threads=True, progress=False)
    data.to_pickle(CACHE)
    return data


def weekly(d):
    w = d.resample("W-FRI").agg({"Open": "first", "Close": "last", "Volume": "sum"}).dropna()
    w = w[w.index <= pd.Timestamp.today()]  # 진행 중인 주 제외
    w["MA"] = w["Close"].rolling(MA_LEN).mean()
    return w


def signals(w, mkt):
    above = w["Close"] > w["MA"]
    cross = above & ~above.shift(1, fill_value=False) & w["MA"].shift(1).notna()
    vol_avg = w["Volume"].shift(1).rolling(VOL_AVG_WEEKS).mean()
    ext = w["Close"] / w["MA"] - 1
    daily_dv = (w["Close"] * w["Volume"]).rolling(4).mean() / 5  # 일평균 거래대금 근사
    up = (cross
          & (w["MA"] >= w["MA"].shift(SLOPE_WEEKS))
          & (w["Volume"] >= VOL_MULT * vol_avg)
          & ext.between(0.02, 0.15)
          & (daily_dv >= MIN_DOLLAR_VOL))
    m = mkt.reindex(w.index, fill_value=False)
    return {"기본": cross, "기본+시장": cross & m, "업그레이드": up, "업그레이드+시장": up & m}


def trades(o, c, ma, sig):
    """(진입 idx, 청산 idx, 미청산 여부) 목록. 진입/청산 모두 해당 주 시가."""
    out, i, n = [], 0, len(o)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = j = i + 1
        while j < n and c[j] >= ma[j]:
            j += 1
        x = min(j + 1, n - 1)
        if x > e:
            out.append((e, x, j + 1 > n - 1))
        i = x
    return out


def trade_stats(r, held):
    win, loss = r[r > 0], r[r <= 0]
    return {
        "거래수": len(r),
        "승률%": round(len(win) / len(r) * 100, 1),
        "평균수익%": round(r.mean() * 100, 2),
        "평균이익%": round(win.mean() * 100, 1),
        "평균손실%": round(loss.mean() * 100, 1),
        "손익비": round(win.mean() / -loss.mean(), 2),
        "PF": round(win.sum() / -loss.sum(), 2),
        "평균보유주": round(held.mean(), 1),
    }


def port_stats(p, exposure):
    eq = (1 + p).cumprod()
    return {
        "CAGR%": round((eq.iloc[-1] ** (52 / len(p)) - 1) * 100, 1),
        "MDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
        "변동성%": round(p.std() * 52 ** 0.5 * 100, 1),
        "투자비중%": round(exposure * 100, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    tickers = load_tickers()["Symbol"].tolist()
    data = get_prices(tickers, args.refresh)

    spy = weekly(data["SPY"].dropna())
    mkt = (spy["Close"] > spy["MA"]) & (spy["MA"] >= spy["MA"].shift(SLOPE_WEEKS))
    idx = spy.index[spy["MA"].notna()]  # 평가 구간: SPY 40주선 생긴 뒤부터

    variants = ["기본", "기본+시장", "업그레이드", "업그레이드+시장"]
    rets = {v: [] for v in variants}
    held = {v: [] for v in variants}
    psum = {v: pd.Series(0.0, idx) for v in variants}
    pcnt = {v: pd.Series(0.0, idx) for v in variants}

    for t in tickers:
        if t not in data.columns.get_level_values(0):
            continue
        d = data[t].dropna()
        if len(d) < 300:
            continue
        w = weekly(d)
        oo = (w["Open"].shift(-1) / w["Open"] - 1).fillna(0)  # 이번 주 시가 → 다음 주 시가
        o, c, ma = w["Open"].values, w["Close"].values, w["MA"].values
        for v, s in signals(w, mkt).items():
            for e, x, _ in trades(o, c, ma, s.values):
                rets[v].append(o[x] / o[e] - 1 - 2 * COST)
                held[v].append(x - e)
                r = oo.iloc[e:x].copy()
                r.iloc[0] -= COST
                r.iloc[-1] -= COST
                psum[v] = psum[v].add(r.reindex(idx, fill_value=0), fill_value=0)
                pcnt[v] = pcnt[v].add(pd.Series(1.0, r.index).reindex(idx, fill_value=0), fill_value=0)

    # ponytail: 보유 종목 수 제한 없음 + 매주 동일비중 리밸런싱 비용 무시. 실전 근사하려면 최대 N종목 + RS 순위 추가
    spy_oo = (spy["Open"].shift(-1) / spy["Open"] - 1).reindex(idx).fillna(0)
    ports = {v: (psum[v] / pcnt[v]).where(pcnt[v] > 0, 0) for v in variants}

    print(f"기간: {idx[0].date()} ~ {idx[-1].date()}, 종목 {len(tickers)}개, 편도비용 {COST:.1%}\n")
    print("===== 거래 단위 =====")
    print(pd.DataFrame({v: trade_stats(pd.Series(rets[v]), pd.Series(held[v])) for v in variants}).T.to_string())

    print("\n===== 포트폴리오 (동일비중) =====")
    pt = {v: port_stats(ports[v], (pcnt[v] > 0).mean()) for v in variants}
    pt["SPY 보유"] = port_stats(spy_oo, 1.0)
    print(pd.DataFrame(pt).T.to_string())

    print("\n===== 연도별 수익률% =====")
    yearly = pd.DataFrame({**ports, "SPY": spy_oo}).groupby(lambda d: d.year).apply(lambda x: (1 + x).prod() - 1)
    print((yearly * 100).round(1).to_string())


if __name__ == "__main__":
    main()
