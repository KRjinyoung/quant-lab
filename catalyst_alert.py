"""
단일 호재 과대상승 → 호재 다각화 하락 경보 백테스트 (S&P 500 현재 구성종목, 2005~)

가설: 하나의 호재로 급등한 종목이 이후 '새로운 호재'를 연달아 내놓기 시작하면
      (스토리 확장 = 주가 방어용 내러티브) 고점 신호 → 이후 초과수익 음(-)

정의
  과대상승  : 60거래일 수익률 >= SURGE, SPY 대비 초과 >= SURGE_EX. 종목당 250일 쿨다운
  호재일    : 일수익률 >= +7% & 거래량 >= 50일 평균 x2.5 (3일 이내는 한 묶음)
  단일 호재 : 급등 구간 내 호재 묶음 1개 & 그 묶음이 상승폭(log)의 40% 이상
  호재 다각화 경보 : 급등 확인 후 120거래일 내, 최근 30거래일
                     '내러티브 8-K'(Item 1.01/2.01/7.01/8.01) >= 3건 & 서로 다른 Item >= 2종
                     & 직전 1년 평균 빈도의 2배 이상 & 주가가 급등 전 대비 +30% 이상 유지
  성과      : 경보 다음날 시가 진입 기준 20/60/120일 SPY 대비 초과수익
  대조군    : (a) 단일 호재인데 경보 없음 — 경보 평균 지연일에 측정
              (b) 다중/점진 상승인데 경보 발생
"""
import numpy as np
import pandas as pd

SURGE, SURGE_EX, WIN = 0.50, 0.40, 60
CAT_RET, CAT_VOL = 0.07, 2.5
SINGLE_SHARE = 0.40
WATCH, ROLL = 120, 30
NARR = {"1.01", "2.01", "7.01", "8.01"}
HOLD = [20, 60, 120]


def surge_events(c, spy_c):
    r = c / c.shift(WIN) - 1
    ex = r - (spy_c / spy_c.shift(WIN) - 1)
    hit = (r >= SURGE) & (ex >= SURGE_EX)
    out, last = [], -10**9
    for i in np.flatnonzero(hit.values):
        if i - last >= 250:
            out.append(i)
            last = i
    return out


def classify(d, i):
    """급등 구간 [i-WIN, i] 의 호재 묶음 수와 최대 묶음 기여도"""
    c, v = d["Close"].values, d["Volume"].values
    ret = np.r_[np.nan, c[1:] / c[:-1] - 1]
    vavg = pd.Series(v).shift(1).rolling(50).mean().values
    days = [k for k in range(i - WIN + 1, i + 1) if ret[k] >= CAT_RET and v[k] >= CAT_VOL * vavg[k]]
    clusters = []
    for k in days:
        if clusters and k - clusters[-1][-1] <= 3:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    total = np.log(c[i] / c[i - WIN])
    share = max((sum(np.log1p(ret[k]) for k in cl) for cl in clusters), default=0) / total
    return len(clusters), share


def alert_day(d, i, f8k):
    """급등 확인일 i 이후 WATCH일 내 첫 다각화 경보 인덱스 (없으면 None)"""
    idx, c = d.index, d["Close"].values
    base = c[i - WIN]
    narr = f8k[f8k["narr"]]
    for k in range(i, min(i + WATCH, len(d) - 1)):
        t0, t1 = idx[max(k - ROLL, 0)], idx[k]
        w = narr[(narr.date > t0) & (narr.date <= t1)]
        if len(w) < 3 or len(set().union(*w["nitems"])) < 2 or c[k] < base * 1.3:
            continue
        prior = narr[(narr.date > t1 - pd.Timedelta(days=365 + 45)) & (narr.date <= t1 - pd.Timedelta(days=45))]
        rate = len(prior) / 365 * 45  # 30거래일 ≈ 45일
        if len(w) >= 2 * max(rate, 0.5):
            return k
    return None


def fwd(d, spy, k):
    """k 다음날 시가 진입 → h일 후 시가, SPY 대비 초과"""
    o, so = d["Open"], spy["Open"].reindex(d.index)
    out = {}
    for h in HOLD:
        if k + 1 + h < len(d):
            out[f"ex{h}"] = (o.iloc[k + 1 + h] / o.iloc[k + 1]) - (so.iloc[k + 1 + h] / so.iloc[k + 1])
    return out


def summarize(df, name):
    row = {"그룹": name, "N": len(df)}
    for h in HOLD:
        s = df[f"ex{h}"].dropna() if f"ex{h}" in df else pd.Series(dtype=float)
        if len(s) < 3:
            continue
        row[f"{h}d 평균%"] = round(s.mean() * 100, 1)
        row[f"{h}d 중앙%"] = round(s.median() * 100, 1)
        row[f"{h}d 하락%"] = round((s < 0).mean() * 100, 0)
        row[f"{h}d t"] = round(s.mean() / (s.std(ddof=1) / len(s) ** 0.5), 2)
    return row


def run(data, filings, verbose=True):
    spy = data["SPY"].dropna()
    filings = filings.copy()
    filings["nitems"] = filings["items"].fillna("").map(lambda s: frozenset(x.strip() for x in s.split(",")) & NARR)
    filings["narr"] = filings["nitems"].map(bool)
    by_t = dict(tuple(filings.groupby("ticker")))

    ev = []
    for t in data.columns.get_level_values(0).unique():
        if t == "SPY" or t not in by_t:
            continue
        d = data[t].dropna()
        if len(d) < 300:
            continue
        sc = spy["Close"].reindex(d.index).ffill()
        for i in surge_events(d["Close"], sc):
            if i < 60:
                continue
            n, share = classify(d, i)
            single = n == 1 and share >= SINGLE_SHARE
            a = alert_day(d, i, by_t[t])
            ev.append({"ticker": t, "surge": d.index[i], "single": single, "n_cat": n, "share": share,
                       "alert": d.index[a] if a is not None else pd.NaT,
                       "lag": (a - i) if a is not None else np.nan, "i": i, "a": a, "_d": d})
    ev = pd.DataFrame(ev)
    lag = int(ev.loc[ev.single & ev.alert.notna(), "lag"].median()) if (ev.single & ev.alert.notna()).any() else 30

    rows = []
    for _, e in ev.iterrows():
        k = e.a if pd.notna(e.alert) else min(e.i + lag, len(e._d) - 2)
        rows.append({**{c: e[c] for c in ["ticker", "surge", "single", "n_cat", "share", "alert", "lag"]},
                     **fwd(e._d, spy, int(k))})
    res = pd.DataFrame(rows)
    groups = {
        "단일호재+다각화경보 (신호)": res[res.single & res.alert.notna()],
        "단일호재+경보없음 (대조a)": res[res.single & res.alert.isna()],
        "다중/점진+다각화경보 (대조b)": res[~res.single & res.alert.notna()],
        "다중/점진+경보없음": res[~res.single & res.alert.isna()],
        "전체 급등": res,
    }
    table = pd.DataFrame([summarize(g, k) for k, g in groups.items()]).set_index("그룹")
    if verbose:
        print(f"급등 이벤트 {len(res)}건 (단일호재 {res.single.sum()}), 경보없음 그룹 측정 시점 = 급등 +{lag}일\n")
        print(table.to_string())
    return res, table


if __name__ == "__main__":
    import sys
    data = pd.read_pickle("prices.pkl")
    filings = pd.read_pickle("edgar_8k.pkl")
    res, table = run(data, filings)
    sig = res[res.single & res.alert.notna()].sort_values("alert")
    print("\n===== 신호 목록 =====")
    print(sig[["ticker", "surge", "alert", "lag", "share", "ex20", "ex60", "ex120"]]
          .assign(**{c: lambda x, c=c: (x[c] * 100).round(1) for c in ["share", "ex20", "ex60", "ex120"]})
          .to_string(index=False))

    if "--grid" in sys.argv:
        print("\n===== 민감도 (신호 그룹 60d 평균초과% / N) =====")
        import catalyst_alert as m
        for s, se in [(0.4, 0.3), (0.5, 0.4), (0.7, 0.5)]:
            for sh in [0.3, 0.4, 0.5]:
                m.SURGE, m.SURGE_EX, m.SINGLE_SHARE = s, se, sh
                _, tb = m.run(data, filings, verbose=False)
                r = tb.loc["단일호재+다각화경보 (신호)"]
                c = tb.loc["단일호재+경보없음 (대조a)"]
                print(f"급등>={s:.0%} 기여>={sh:.0%}: 신호 {r.get('60d 평균%')}% (N={r.N}, t={r.get('60d t')})"
                      f" | 대조a {c.get('60d 평균%')}% (N={c.N})")
