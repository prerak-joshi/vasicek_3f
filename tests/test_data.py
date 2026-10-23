"""
Loader tests on small synthetic files that mimic the layouts of the real
GSW (Fed), Kim-Wright (FRED) and ACM (NY Fed) files.
"""
import numpy as np
import pandas as pd

from vasicek3f.data import load_gsw, load_kw, load_acm


def _fake_gsw(path):
    days = pd.bdate_range("2020-01-01", "2020-04-30")
    notes = ("Fed GSW zero-coupon yields\nSome notes line\n\"Series, with, commas\"\n"
             "Another note\n\n")
    cols = ["Date", "BETA0"] + [f"SVENY{m:02d}" for m in range(1, 31)]
    rows = []
    for i, d in enumerate(days):
        vals = [f"{2.0 + 0.05 * m + 0.001 * i:.4f}" for m in range(1, 31)]
        rows.append(",".join([d.strftime("%Y-%m-%d"), "3.0"] + vals))
    rows[3] = rows[3].rsplit(",", 1)[0] + ",NA"
    with open(path, "w") as f:
        f.write(notes + ",".join(cols) + "\n" + "\n".join(rows) + "\n")


def test_load_gsw(tmp_path):
    p = tmp_path / "feds200628.csv"
    _fake_gsw(p)
    y = load_gsw(str(p), [1, 2, 10])
    assert list(y.columns) == [1.0, 2.0, 10.0]
    assert len(y) == 4 and (y.index == pd.date_range("2020-01-31", periods=4, freq="ME")).all()
    assert 0.0 < y.values.max() < 0.05              # converted to decimals
    last_jan = pd.bdate_range("2020-01-01", "2020-01-31")[-1]
    i = list(pd.bdate_range("2020-01-01", "2020-04-30")).index(last_jan)
    assert np.isclose(y.iloc[0][1.0], (2.05 + 0.001 * i) / 100)


def test_load_kw(tmp_path):
    p = tmp_path / "THREEFYTP10.csv"
    pd.DataFrame({"observation_date": ["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"],
                  "THREEFYTP10": [0.5, ".", 0.4, 0.3]}).to_csv(p, index=False)
    s = load_kw(str(p))
    assert np.allclose(s.values, [0.005, 0.003])


def test_load_acm_xlsx_and_csv(tmp_path):
    df = pd.DataFrame({"DATE": ["02-Jan-2020", "31-Jan-2020", "28-Feb-2020"],
                       "ACMTP10": [0.1, 0.2, -0.3], "ACMY10": [1.8, 1.5, 1.1]})
    px = tmp_path / "ACMTermPremium.xlsx"
    with pd.ExcelWriter(px) as w:
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="Notes", index=False)
        df.to_excel(w, sheet_name="ACM Monthly", index=False)
    s = load_acm(str(px), 10)
    assert np.allclose(s.values, [0.002, -0.003])
    pc = tmp_path / "acm.csv"
    df.to_csv(pc, index=False)
    assert np.allclose(load_acm(str(pc), 10).values, [0.002, -0.003])
