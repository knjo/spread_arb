"""Calendar known before each decision session, separate from realized closures.

Coverage includes every remaining contract session in the May--September study.
The annual schedule was published before this study; the July typhoon closure
was announced on July 9, without a usable intraday publication timestamp.
"""
from datetime import datetime, timedelta

START, END = "20260504", "20260916"
ANNUAL_SOURCE = "https://www.taifex.com.tw/file/taifex/CHINESE/4/2026Calendar.pdf"
PLANNED_HOLIDAYS = ("20260619",)
EXTRA_CLOSURES = (("20260710", "20260709",
    "https://www.taifex.com.tw/enl/eng11/newsDetail.do?idx=10470&thetype=2"),)


def calendar_spec() -> dict:
    return dict(start=START, end=END, annual_source=ANNUAL_SOURCE,
                planned_holidays=list(PLANNED_HOLIDAYS),
                extra_closures=[dict(day=d, publication_day=p, source=s)
                                for d, p, s in EXTRA_CLOSURES],
                availability="publication_day strictly before decision day; no intraday timestamp assumed")


def known_calendar(day: str) -> dict[str, bool]:
    if not START <= day <= END:
        raise ValueError("decision outside verified forecast-calendar coverage")
    closed = set(PLANNED_HOLIDAYS)
    closed.update(d for d, published, _ in EXTRA_CLOSURES if published < day)
    current, end = datetime.strptime(START, "%Y%m%d"), datetime.strptime(END, "%Y%m%d")
    result = {}
    while current <= end:
        key = current.strftime("%Y%m%d")
        result[key] = current.weekday() < 5 and key not in closed
        current += timedelta(days=1)
    return result
