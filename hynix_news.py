"""
SK하이닉스 과거 뉴스 헤드라인 수집 (Google News RSS, 10일 단위, 2016~)
  결과: hynix_news.pkl — DataFrame[date, title, source]
"""
import time
import xml.etree.ElementTree as ET
import pandas as pd
import requests

QUERIES = ["SK하이닉스", "SK하이닉스 주가", "하이닉스 호재"]
OUT = "hynix_news.pkl"


def fetch(q, a, b):
    for _ in range(4):
        try:
            r = requests.get("https://news.google.com/rss/search",
                             params={"q": f"{q} after:{a} before:{b}", "hl": "ko", "gl": "KR", "ceid": "KR:ko"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if r.status_code == 200:
                return ET.fromstring(r.content).findall(".//item")
        except Exception:
            pass
        time.sleep(5)
    return []


def main():
    rows = []
    starts = pd.date_range("2016-01-01", pd.Timestamp.today(), freq="10D")
    for n, s in enumerate(starts):
        a, b = s.strftime("%Y-%m-%d"), (s + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        for q in QUERIES:
            for it in fetch(q, a, b):
                rows.append((it.findtext("pubDate"), it.findtext("title"), it.findtext("source")))
            time.sleep(0.4)
        if n % 20 == 0:
            print(a, len(rows), flush=True)
    df = pd.DataFrame(rows, columns=["date", "title", "source"])
    df["date"] = pd.to_datetime(df["date"], format="%a, %d %b %Y %H:%M:%S %Z", errors="coerce").dt.tz_localize(None)
    df = df.dropna().drop_duplicates("title")
    df.to_pickle(OUT)
    print("saved", len(df))


if __name__ == "__main__":
    main()
