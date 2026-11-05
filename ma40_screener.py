"""
40주 이동평균선 돌파 스크리너 (S&P 500)
  - 기본 버전  : 주봉 종가가 40주 SMA를 신규 상향 돌파
  - 업그레이드 : 기본 + 시장 필터 + MA 기울기 + 거래량 + 상대강도(RS) 정렬

사용법:
  pip install yfinance pandas
  python ma40_screener.py            # 두 버전 모두 실행
  python ma40_screener.py --lookback 3   # 최근 3주 내 돌파까지 포함
"""
import argparse
import pandas as pd
import yfinance as yf

TICKER_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
MA_LEN = 40          # 이동평균 기간(주)
SLOPE_WEEKS = 5      # 기울기 비교 기간
VOL_AVG_WEEKS = 10   # 거래량 평균 기간
VOL_MULT = 1.5       # 돌파 주 거래량 배수
RS_WEEKS = 26        # 상대강도 기간(약 6개월)
MIN_DOLLAR_VOL = 20e6  # 최소 일평균 거래대금($)


def load_tickers():
    df = pd.read_csv(TICKER_CSV)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    return df[["Symbol", "Security", "GICS Sector"]]


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 주봉(금요일 기준). 진행 중인 주는 마지막 거래일 종가 사용."""
    return daily.resample("W-FRI").agg(
        {"Close": "last", "Volume": "sum"}
    ).dropna()


def download(tickers, years=3):
    data = yf.download(tickers, period=f"{years}y", interval="1d",
                       auto_adjust=True, group_by="ticker",
                       threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            d = data[t][["Close", "Volume"]].dropna()
            if len(d) > 250:
                out[t] = d
        except KeyError:
            pass
    return out


def analyze(t, daily, bench_w, lookback):
    w = to_weekly(daily)
    if len(w) < MA_LEN + SLOPE_WEEKS + 2:
        return None
    w["MA"] = w["Close"].rolling(MA_LEN).mean()
    w["VolAvg"] = w["Volume"].shift(1).rolling(VOL_AVG_WEEKS).mean()
    above = w["Close"] > w["MA"]
    cross = above & ~above.shift(1, fill_value=False)

    recent = cross.iloc[-lookback:]
    if not recent.any() or not above.iloc[-1]:
        return None
    cross_date = recent[recent].index[-1]
    cw = w.loc[cross_date]
    last = w.iloc[-1]

    # 상대강도: 종목 수익률 / 벤치마크 수익률 (RS_WEEKS)
    b = bench_w.reindex(w.index).ffill()
    rs = (last["Close"] / w["Close"].iloc[-RS_WEEKS - 1]) / \
         (b.iloc[-1] / b.iloc[-RS_WEEKS - 1])

    return {
        "Ticker": t,
        "돌파주": cross_date.date(),
        "종가": round(last["Close"], 2),
        "MA40": round(last["MA"], 2),
        "MA대비%": round((last["Close"] / last["MA"] - 1) * 100, 1),
        "MA기울기%": round((last["MA"] / w["MA"].iloc[-1 - SLOPE_WEEKS] - 1) * 100, 2),
        "거래량배수": round(cw["Volume"] / cw["VolAvg"], 2) if cw["VolAvg"] else None,
        "RS": round(rs, 3),
        "일평균거래대금($M)": round((daily["Close"] * daily["Volume"]).tail(20).mean() / 1e6, 1),
    }


def market_ok(bench_w):
    ma = bench_w.rolling(MA_LEN).mean()
    return bench_w.iloc[-1] > ma.iloc[-1] and ma.iloc[-1] >= ma.iloc[-1 - SLOPE_WEEKS], \
        bench_w.iloc[-1], ma.iloc[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=1, help="최근 N주 내 돌파 포함 (기본 1 = 이번 주)")
    args = ap.parse_args()

    meta = load_tickers()
    tickers = meta["Symbol"].tolist()
    print(f"S&P 500 {len(tickers)}개 종목 다운로드 중...")
    daily = download(tickers)

    spy = yf.download("SPY", period="3y", auto_adjust=True, progress=False)
    spy_close = spy["Close"].squeeze()
    bench_w = spy_close.resample("W-FRI").last().dropna()

    rows = [r for t, d in daily.items() if (r := analyze(t, d, bench_w, args.lookback))]
    df = pd.DataFrame(rows)
    if df.empty:
        print("돌파 종목 없음")
        return
    df = df.merge(meta, left_on="Ticker", right_on="Symbol").drop(columns="Symbol")

    # ---------- 기본 버전 ----------
    basic = df.sort_values("MA대비%")
    print(f"\n===== [기본] 40주선 신규 돌파: {len(basic)}개 =====")
    print(basic[["Ticker", "Security", "GICS Sector", "돌파주", "종가", "MA40", "MA대비%"]]
          .to_string(index=False))
    basic.to_csv("ma40_basic.csv", index=False, encoding="utf-8-sig")

    # ---------- 업그레이드 버전 ----------
    ok, spx, spx_ma = market_ok(bench_w)
    print(f"\n시장 필터 (SPY {spx:.2f} vs 40주선 {spx_ma:.2f}): {'통과 ✅' if ok else '미통과 ⚠️ (신호 신뢰도 낮음)'}")
    up = df[
        (df["MA기울기%"] >= 0) &
        (df["거래량배수"] >= VOL_MULT) &
        (df["MA대비%"].between(2, 15)) &
        (df["일평균거래대금($M)"] >= MIN_DOLLAR_VOL / 1e6)
    ].sort_values("RS", ascending=False)
    print(f"\n===== [업그레이드] 필터 통과: {len(up)}개 (RS 순) =====")
    print(up[["Ticker", "Security", "GICS Sector", "돌파주", "MA대비%", "MA기울기%", "거래량배수", "RS"]]
          .to_string(index=False))
    up.to_csv("ma40_upgraded.csv", index=False, encoding="utf-8-sig")
    print("\nCSV 저장: ma40_basic.csv, ma40_upgraded.csv")


if __name__ == "__main__":
    main()
