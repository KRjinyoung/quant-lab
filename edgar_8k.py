"""
SEC EDGAR 8-K 공시 수집 (S&P 500 현재 구성종목)
  결과: edgar_8k.pkl  — DataFrame[ticker, date, items]
사용법:
  python edgar_8k.py
"""
import json
import time
import pandas as pd
import requests
from ma40_screener import load_tickers

UA = {"User-Agent": "quant-lab research admin@example.com"}
OUT = "edgar_8k.pkl"


def get(url):
    for _ in range(5):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2)
    return None


def rows(block, t):
    for f, d, it in zip(block["form"], block["filingDate"], block["items"]):
        if f == "8-K":
            yield t, d, it


def main():
    cik = {v["ticker"].replace(".", "-"): v["cik_str"]
           for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    out = []
    tickers = load_tickers()["Symbol"].tolist()
    for n, t in enumerate(tickers):
        if t not in cik:
            continue
        j = get(f"https://data.sec.gov/submissions/CIK{cik[t]:010d}.json")
        time.sleep(0.12)
        if not j:
            continue
        out += rows(j["filings"]["recent"], t)
        for f in j["filings"]["files"]:
            if f["filingTo"] < "2004-06-01":
                continue
            k = get("https://data.sec.gov/submissions/" + f["name"])
            time.sleep(0.12)
            if k:
                out += rows(k, t)
        if n % 50 == 0:
            print(n, t, len(out), flush=True)
    df = pd.DataFrame(out, columns=["ticker", "date", "items"]).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    df.to_pickle(OUT)
    print("saved", len(df), "8-K filings,", df.ticker.nunique(), "tickers")


if __name__ == "__main__":
    main()
