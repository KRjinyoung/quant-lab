"""
SK하이닉스: '단일 호재 과대상승 → 호재 다각화 → 하락' 가설 검증

1) 뉴스 헤드라인을 호재 테마로 분류 (실적/일반 기사는 제외)
2) 60일 창마다 테마 집중도: 1위 테마 비중, 유효 테마 수(1/HHI)
3) 25% 지그재그로 주요 상승→하락 구간 추출 → 구간 초반 vs 고점 직전 테마 구성 비교
4) 경보 규칙(월말 평가) 백테스트:
     과대상승: 6개월 수익률 >= +50%
     단일 호재: 상승 시작 시점(6개월 전~4개월 전) 1위 테마 비중 >= 50%
     다각화   : 최근 60일 1위 테마 비중이 초반 대비 15%p 이상 하락 & 유효 테마 수 +0.8 이상
   → 경보 후 1/3/6개월 수익률(절대, KOSPI 대비)을 '과대상승인데 경보 없음'과 비교
"""
import re
import numpy as np
import pandas as pd
import yfinance as yf

THEMES = {
    "HBM/AI": r"HBM|고대역폭|AI ?메모리|엔비디아|NVIDIA|Nvidia|GPU|블랙웰|루빈|AI 반도체|AI반도체",
    "범용메모리 업황": r"D램 ?가격|디램|DRAM|메모리 ?가격|슈퍼 ?사이클|슈퍼호황|호황|업황|공급 ?부족|쇼티지|고정거래|가격 ?인상|DDR5|DDR4|LPDDR",
    "낸드/SSD": r"낸드|NAND|SSD|솔리다임|Solidigm|QLC",
    "M&A/지분": r"인수|도시바|키옥시아|Kioxia|인텔 ?낸드|지분|합병|M&A",
    "주주환원": r"배당|자사주|주주환원|밸류업|소각",
    "자본시장/상장": r"ADR|미국 ?상장|뉴욕증시 ?상장|MSCI|지수 ?편입|외국인 ?순매수|시가총액|시총",
    "빅테크 계약": r"오픈AI|OpenAI|스타게이트|Stargate|마이크로소프트|구글|TPU|아마존|AWS|메타|브로드컴|장기 ?계약|LTA|선수금|공급 ?계약|파트너십",
    "신사업/차세대": r"CXL|PIM|온디바이스|뉴로모픽|파운드리|패키징|인디애나|시스템반도체|이미지센서|CIS|로봇|양자|PIM|커스텀",
    "정책/인프라": r"용인|클러스터|보조금|칩스법|CHIPS|전력|발전소|세제|팹|공장|증설|청주|M15",
}
PAT = {k: re.compile(v) for k, v in THEMES.items()}


def tag(news):
    for k, p in PAT.items():
        news[k] = news["title"].str.contains(p)
    return news


def concentration(news, end, days=60):
    w = news[(news.date > end - pd.Timedelta(days=days)) & (news.date <= end)]
    cnt = w[list(THEMES)].sum()
    tot = cnt.sum()
    if tot < 8:
        return np.nan, np.nan, None, int(tot)
    s = cnt / tot
    return s.max(), 1 / (s ** 2).sum(), s.idxmax(), int(tot)


def zigzag(c, th=0.25):
    piv, mode, ei, ev = [(c.index[0], c.iloc[0])], "up", c.index[0], c.iloc[0]
    for t, p in c.items():
        if mode == "up":
            if p > ev: ei, ev = t, p
            elif p <= ev * (1 - th): piv.append((ei, ev)); mode, ei, ev = "dn", t, p
        else:
            if p < ev: ei, ev = t, p
            elif p >= ev / (1 - th): piv.append((ei, ev)); mode, ei, ev = "up", t, p
    piv.append((ei, ev))
    return piv


def main():
    px = pd.read_pickle("hynix.pkl")["Close"]
    ks = yf.download("^KS11", start="2015-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    news = tag(pd.read_pickle("hynix_news.pkl"))
    print(f"헤드라인 {len(news)}건, 테마 태깅 {news[list(THEMES)].any(axis=1).sum()}건\n")
    print("연도별 테마 비중(%)")
    yr = news.groupby(news.date.dt.year)[list(THEMES)].sum()
    print((yr.div(yr.sum(axis=1), axis=0) * 100).round(0).astype(int).to_string(), "\n")

    # ---- 3) 주요 상승→하락 구간
    piv = zigzag(px["2016":])
    print("===== 주요 상승 구간: 초반 vs 고점 직전 호재 구성 =====")
    rows = []
    for (t0, p0), (t1, p1), (t2, p2) in zip(piv, piv[1:], piv[2:]):
        if p1 / p0 - 1 < 0.40 or p2 > p1:   # 상승 >= 40% 후 하락 구간
            continue
        early_end = t0 + (t1 - t0) / 3
        e = concentration(news, early_end, days=max(60, (early_end - t0).days))
        l = concentration(news, t1, 60)
        rows.append({"상승시작": t0.date(), "고점": t1.date(), "상승%": round((p1 / p0 - 1) * 100),
                     "이후하락%": round((p2 / p1 - 1) * 100),
                     "초반 1위테마": e[2], "초반 비중%": round(e[0] * 100) if e[0] == e[0] else None,
                     "초반 유효테마수": round(e[1], 1) if e[1] == e[1] else None,
                     "고점前 1위테마": l[2], "고점前 비중%": round(l[0] * 100) if l[0] == l[0] else None,
                     "고점前 유효테마수": round(l[1], 1) if l[1] == l[1] else None})
    ep = pd.DataFrame(rows)
    print(ep.to_string(index=False), "\n")

    # ---- 4) 월말 경보 백테스트
    me = px.resample("ME").last()
    rel = (px / ks.reindex(px.index).ffill()).resample("ME").last()
    rec = []
    for t in me.loc["2016-07":].index:
        r6 = me[t] / me[:t].iloc[-7] - 1
        e_top, e_eff, e_th, _ = concentration(news, t - pd.DateOffset(months=4), 60)
        l_top, l_eff, l_th, n = concentration(news, t, 60)
        f = {}
        for h in (1, 3, 6):
            nxt = me.loc[t:].iloc[h] if len(me.loc[t:]) > h else np.nan
            f[f"r{h}m"] = nxt / me[t] - 1
            nr = rel.loc[t:].iloc[h] if len(rel.loc[t:]) > h else np.nan
            f[f"x{h}m"] = nr / rel[t] - 1
        rec.append({"month": t, "r6": r6, "early_top": e_top, "early_eff": e_eff, "early_theme": e_th,
                    "top": l_top, "eff": l_eff, "theme": l_th, "n": n, **f})
    m = pd.DataFrame(rec).set_index("month")
    m["surge"] = m.r6 >= 0.50
    m["single"] = m.early_top >= 0.50
    m["divers"] = (m.early_top - m.top >= 0.15) & (m.eff - m.early_eff >= 0.8)
    m["alert"] = m.surge & m.single & m.divers

    def stat(g, name):
        o = {"그룹": name, "월수": len(g)}
        for h in (1, 3, 6):
            s = g[f"r{h}m"].dropna(); x = g[f"x{h}m"].dropna()
            if len(s):
                o[f"{h}M 평균%"] = round(s.mean() * 100, 1)
                o[f"{h}M 하락확률%"] = round((s < 0).mean() * 100)
                o[f"{h}M KOSPI대비%"] = round(x.mean() * 100, 1)
        return o

    groups = {
        "경보 (과대+단일+다각화)": m[m.alert],
        "과대+단일, 다각화 없음": m[m.surge & m.single & ~m.divers],
        "과대+다각화 (단일 아님)": m[m.surge & ~m.single & m.divers],
        "과대상승 전체": m[m.surge],
        "전체 월": m,
    }
    print("===== 월말 경보 백테스트 (6M 수익률 >= +50% 를 과대상승으로 정의) =====")
    print(pd.DataFrame([stat(g, k) for k, g in groups.items()]).set_index("그룹").to_string(), "\n")

    print("경보 발생 월:")
    a = m[m.alert][["r6", "early_theme", "early_top", "theme", "top", "eff", "r1m", "r3m", "r6m"]].copy()
    for c in ["r6", "early_top", "top", "r1m", "r3m", "r6m"]:
        a[c] = (a[c] * 100).round(0)
    print(a.round(1).to_string(), "\n")

    # 전 기간: 다각화(1위 비중 하락)와 향후 수익률 상관
    m["d_top"] = m.top - m.early_top
    sub = m.dropna(subset=["d_top", "r3m"])
    print("전 기간 상관: 1위테마비중 변화 vs 3M 수익률 =", round(sub.d_top.corr(sub.r3m), 3),
          "| 과대상승 월만 =", round(sub[sub.surge].d_top.corr(sub[sub.surge].r3m), 3))
    m.to_csv("hynix_monthly_signal.csv")


if __name__ == "__main__":
    main()
