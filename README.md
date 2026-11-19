# quant-lab
Personal quant lab for US equities - Jinyoung Park from South Korea

## 스크립트

| 파일 | 내용 |
|---|---|
| `ma40_screener.py` | S&P 500 40주 이동평균선 돌파 스크리너 (기본 / 업그레이드) |
| `backtest.py` | 40주선 돌파 전략 백테스트 (2005~, 주봉, 동일비중) |
| `edgar_8k.py` | SEC EDGAR 8-K 공시 이력 수집 → `edgar_8k.pkl` |
| `catalyst_alert.py` | 단일 호재 과대상승 → 호재 다각화(8-K Item 기준) 하락 경보 이벤트 스터디 |
| `hynix_news.py` | SK하이닉스 뉴스 헤드라인 수집 (Google News RSS, 2016~) → `hynix_news.pkl` |
| `hynix_diversify.py` | SK하이닉스 9개 호재 테마 집중도 vs 상승·하락 구간 검증 |
| `hynix_hw_fin.py` | 하드웨어 / 재무 2개 테마 다각화 경보 + 주간 민감도 그리드 |

## 실행

```bash
python -m venv .venv && .venv/bin/pip install yfinance pandas requests
.venv/bin/python backtest.py            # 첫 실행 시 prices.pkl 캐시 생성
.venv/bin/python edgar_8k.py && .venv/bin/python catalyst_alert.py --grid
.venv/bin/python hynix_news.py && .venv/bin/python hynix_hw_fin.py
```

`hynix_*.py`는 `hynix.pkl`(000660.KS 일봉)이 필요합니다:

```bash
.venv/bin/python -c "import yfinance as yf; d=yf.download('000660.KS',start='2010-01-01',auto_adjust=True); d.columns=d.columns.get_level_values(0); d.to_pickle('hynix.pkl')"
```

데이터 파일(`*.pkl`, `*.csv`)은 git에 포함되지 않습니다.

## 지금까지 결과 요약 (SK하이닉스, 2016~2026)

- 호재가 다각화된 뒤에도 주가는 대부분 계속 올랐습니다. 다각화를 하락 경보로 쓸 근거는 나오지 않았습니다.
- 하드웨어 / 재무 두 테마로 나눠도 재무 비중 변화와 13주 수익률 사이에는 상관이 거의 없었습니다 (0.03).
- 재무 호재가 25%p 넘게 급증한 경우만 하락 경향이 있었지만, 독립 사례가 2~3개뿐이고 조건을 조금만 바꿔도 결과가 뒤집혔습니다.
