"""
Data loaders for

  * GSW  - Gurkaynak, Sack & Wright (2007) Fed zero-coupon Treasury yield curve
           (FEDS 2006-28), continuously compounded zero yields SVENY01..SVENY30, percent.
  * ACM  - Adrian, Crump & Moench (2013) NY Fed term premium (ACMTP01..ACMTP10), percent.
  * KW   - Kim & Wright (2005) Fed term premium, FRED series THREEFYTP1..THREEFYTP10, percent.

All loaders return month-end data in DECIMALS (0.045 = 4.5%).

The download helper uses only the standard library. Some sites reject requests
without a browser User-Agent; one is set. If a download fails, save the file
manually from the URL and pass the local path instead.
"""
from __future__ import annotations

import io
import os
import urllib.request
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
ACM_URL = ("https://www.newyorkfed.org/medialibrary/media/research/"
           "data_indicators/ACMTermPremium.xls")
KW_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=THREEFYTP{n}"

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) vasicek3f research script"}


def download(url: str, dest: str, timeout: int = 60) -> str:
    """Download url to dest (creating folders). Returns dest."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def download_all(dest_dir: str = "data", kw_maturities: Iterable[int] = (10,)) -> dict:
    """Fetch GSW, ACM and KW files into dest_dir. Failures are reported, not raised."""
    out = {}
    jobs = [("gsw", GSW_URL, os.path.join(dest_dir, "feds200628.csv")),
            ("acm", ACM_URL, os.path.join(dest_dir, "ACMTermPremium.xls"))]
    jobs += [(f"kw{n}", KW_URL.format(n=n), os.path.join(dest_dir, f"THREEFYTP{n}.csv"))
             for n in kw_maturities]
    for name, url, path in jobs:
        try:
            out[name] = download(url, path)
            print(f"  ok   {name:5s} -> {path}")
        except Exception as e:  # noqa: BLE001
            out[name] = None
            print(f"  FAIL {name:5s} {url}\n       {e}\n       (download manually and pass the path)")
    return out


def _month_end(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Last available observation in each calendar month, indexed at month end."""
    return df.sort_index().resample("ME").last()


# ---------------------------------------------------------------------------
# GSW
# ---------------------------------------------------------------------------
def load_gsw(path: str, maturities: Sequence[float] = tuple(range(1, 11)),
             start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Load GSW zero-coupon yields at integer-year maturities, month-end, decimals.

    The Fed CSV begins with several lines of notes; the header row is the first
    line whose first field is 'Date'.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("date"))
    df = pd.read_csv(io.StringIO("".join(lines[hdr:])), na_values=["NA", "", "."])
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    cols = {}
    for m in maturities:
        if abs(m - round(m)) > 1e-9:
            raise ValueError(f"GSW tables provide integer-year maturities only (got {m})")
        name = f"SVENY{int(round(m)):02d}"
        if name not in df.columns:
            raise KeyError(f"{name} not found in GSW file")
        cols[float(m)] = pd.to_numeric(df[name], errors="coerce") / 100.0
    out = _month_end(pd.DataFrame(cols))
    out.columns.name = "maturity"
    if start:
        out = out.loc[pd.Timestamp(start):]
    if end:
        out = out.loc[: pd.Timestamp(end) + pd.offsets.MonthEnd(0)]
    return out.dropna(how="all")


# ---------------------------------------------------------------------------
# ACM
# ---------------------------------------------------------------------------
def load_acm(path: str, maturity: int = 10) -> pd.Series:
    """
    Load the ACM term premium for one maturity (years), month-end, decimals.
    Accepts the NY Fed .xls/.xlsx workbook (any sheet containing ACMTPxx) or a CSV.
    """
    col = f"ACMTP{maturity:02d}"
    if path.lower().endswith((".xls", ".xlsx")):
        sheets = pd.read_excel(path, sheet_name=None)
        frame = None
        for _, sh in sheets.items():
            sh.columns = [str(c).strip().upper() for c in sh.columns]
            if col in sh.columns:
                frame = sh
                break
        if frame is None:
            raise KeyError(f"{col} not found in any sheet of {path}")
    else:
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().upper() for c in frame.columns]
    date_col = next(c for c in frame.columns if "DATE" in c)
    s = pd.Series(pd.to_numeric(frame[col], errors="coerce").values,
                  index=pd.to_datetime(frame[date_col], format="mixed", dayfirst=False),
                  name=f"ACM_{maturity}y") / 100.0
    return _month_end(s.dropna())


# ---------------------------------------------------------------------------
# Kim-Wright (FRED)
# ---------------------------------------------------------------------------
def load_kw(path: str) -> pd.Series:
    """Load a FRED THREEFYTPn CSV (Kim-Wright term premium), month-end, decimals."""
    df = pd.read_csv(path, na_values=["."])
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if c.lower() in ("observation_date", "date"))
    val_col = next(c for c in df.columns if c != date_col)
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]), name=f"KW_{val_col}") / 100.0
    return _month_end(s.dropna())
