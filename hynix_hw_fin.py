"""
SK하이닉스: 호재를 2개 테마(하드웨어 / 재무)로 묶었을 때
'하드웨어 단일 호재 과대상승 → 재무 호재로 다각화 → 하락' 가설 검증

  하드웨어: HBM/AI, 범용메모리 업황, 낸드/SSD, 빅테크 계약, 신사업/차세대, 정책/인프라(팹·증설)
  재무    : 주주환원, 자본시장/상장, M&A/지분
  지표    : 최근 60일 테마 태깅 헤드라인 중 재무 비중(fin)
  경보    : 과대상승(6M >= S) & 상승 초반(4개월 전 시점) 하드웨어 비중 >= H
            & 현재 재무 비중 - 초반 재무 비중 >= D
"""
import numpy as np
import pandas as pd
import yfinance as yf
from hynix_diversify import tag, zigzag

HW = ["HBM/AI", "범용메모리 업황", "낸드/SSD", "빅테크 계약", "신사업/차세대", "정책/인프라"]
FIN = ["주주환원", "자본시장/상장", "M&A/지분"]


def fin_share(news, end, days=60):
    w = news[(news.date > end - pd.Timedelta(days=days)) & (news.date <= end)]
    hw, fin = w[HW].any(axis=1).sum(), w[FIN].any(axis=1).sum()
    return (fin / (hw + fin), int(hw + fin)) if hw + fin >= 8 else (np.nan, int(hw + fin))


def build(px, rel, news, freq, lag_back, horizons):
    p = px.resample(freq).last()
    r = rel.resample(freq).last()
    rec = []
    for i, t in enumerate(p.index):
        if t < pd.Timestamp("2016-07-01"):
            continue
        back = p[:t].iloc[-lag_back - 1] if len(p[:t]) > lag_back else np.nan
        e_fin, _ = fin_share(news, t - pd.DateOffset(months=4))
        fin, n = fin_share(news, t)
        o = {"t": t, "r6": p[t] / back - 1, "early_fin": e_fin, "fin": fin, "n": n}
        for name, h in horizons.items():
            j = i + h
            o[f"r{name}"] = p.iloc[j] / p[t] - 1 if j < len(p) else np.nan
            o[f"x{name}"] = r.iloc[j] / r[t] - 1 if j < len(r) else np.nan
        rec.append(o)
    m = pd.DataFrame(rec).set_index("t")
    m["d_fin"] = m.fin - m.early_fin
    return m


def stat(g, name, hz):
    o = {"그룹": name, "N": len(g)}
    for h in hz:
        s, x = g[f"r{h}"].dropna(), g[f"x{h}"].dropna()
        if len(s):
            o[f"{h} 평균%"] = round(s.mean() * 100, 1)
            o[f"{h} 중앙%"] = round(s.median() * 100, 1)
            o[f"{h} 하락확률%"] = round((s < 0).mean() * 100)
            o[f"{h} KOSPI대비%"] = round(x.mean() * 100, 1)
    return o


def test(m, S, H, D, hz):
    surge = m.r6 >= S
    single = (1 - m.early_fin) >= H
    div = m.d_fin >= D
    g = {"경보 (과대+HW단일+재무로 다각화)": m[surge & single & div],
         "과대+HW단일, 다각화 없음": m[surge & single & ~div],
         "과대상승 전체": m[surge],
         "전체": m}
    return pd.DataFrame([stat(v, k, hz) for k, v in g.items()]).set_index("그룹"), m[surge & single & div]


def main():
    px = pd.read_pickle("hynix.pkl")["Close"]
    ks = yf.download("^KS11", start="2015-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    rel = px / ks.reindex(px.index).ffill()
    news = tag(pd.read_pickle("hynix_news.pkl"))
    news["HW"], news["FIN"] = news[HW].any(axis=1), news[FIN].any(axis=1)

    print("연도별 하드웨어/재무 헤드라인 수와 재무 비중")
    y = news.groupby(news.date.dt.year)[["HW", "FIN"]].sum()
    y["재무%"] = (y.FIN / (y.HW + y.FIN) * 100).round(0)
    print(y.to_string(), "\n")

    print("===== 주요 상승 구간: 초반 vs 고점 직전 재무 비중 =====")
    rows = []
    piv = zigzag(px["2016":])
    for (t0, p0), (t1, p1), (t2, p2) in zip(piv, piv[1:], piv[2:]):
        if p1 / p0 - 1 < 0.40 or p2 > p1:
            continue
        e = fin_share(news, t0 + (t1 - t0) / 3, days=max(60, ((t1 - t0) / 3).days))[0]
        l = fin_share(news, t1)[0]
        rows.append({"상승시작": t0.date(), "고점": t1.date(), "상승%": round((p1 / p0 - 1) * 100),
                     "이후하락%": round((p2 / p1 - 1) * 100),
                     "초반 재무%": round(e * 100) if e == e else None,
                     "고점前 재무%": round(l * 100) if l == l else None,
                     "변화%p": round((l - e) * 100) if e == e and l == l else None})
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    mh = {"1M": 1, "3M": 3, "6M": 6}
    m = build(px, rel, news, "ME", 6, mh)
    tb, al = test(m, 0.5, 0.8, 0.15, mh)
    print("===== 월말 경보 (과대 6M>=50%, 초반 HW>=80%, 재무비중 +15%p) =====")
    print(tb.to_string())
    print("경보 월:", [f"{t:%y-%m}(재무 {a*100:.0f}%→{b*100:.0f}%, 3M {r*100:+.0f}%)" for t, a, b, r in
                      zip(al.index, al.early_fin, al.fin, al.r3M)], "\n")

    wh = {"4W": 4, "13W": 13, "26W": 26}
    w = build(px, rel, news, "W-FRI", 26, wh)
    print("===== 주간 민감도 그리드 (13주 후 수익률) =====")
    out = []
    for S in (0.3, 0.5, 1.0):
        for H in (0.7, 0.8, 0.9):
            for D in (0.05, 0.10, 0.15, 0.25):
                tb, al = test(w, S, H, D, wh)
                a, b = tb.loc["경보 (과대+HW단일+재무로 다각화)"], tb.loc["과대+HW단일, 다각화 없음"]
                out.append({"과대≥": f"{S:.0%}", "HW단일≥": f"{H:.0%}", "재무↑≥": f"{D*100:.0f}%p",
                            "경보주": int(a.N), "경보 13W평균%": a.get("13W 평균%"), "경보 13W하락%": a.get("13W 하락확률%"),
                            "비경보주": int(b.N), "비경보 13W평균%": b.get("13W 평균%"), "비경보 13W하락%": b.get("13W 하락확률%"),
                            "경보 시기": ",".join(sorted({f"{t:%y-%m}" for t in al.index}))[:60]})
    print(pd.DataFrame(out).to_string(index=False), "\n")

    s = w[w.r6 >= 0.5].dropna(subset=["d_fin", "r13W"])
    print(f"상관(재무비중 변화 vs 13W 수익률): 전체 {w.dropna(subset=['d_fin','r13W']).d_fin.corr(w.r13W):.3f}"
          f" | 과대상승 주만 {s.d_fin.corr(s.r13W):.3f} (N={len(s)})")
    print(f"상관(재무비중 수준 vs 13W 수익률): 과대상승 주만 {s.fin.corr(s.r13W):.3f}")
    w.to_csv("hynix_hw_fin_weekly.csv")


if __name__ == "__main__":
    main()
