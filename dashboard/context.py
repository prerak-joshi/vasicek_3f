"""Loading results bundles and packaging what every view needs into one context object."""

from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from vasicek3f.params import VasicekParams
from vasicek3f.pipeline import BUNDLE_FILE, LEGACY_BUNDLE_FILE, RunConfig, load_bundle, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saved runs, in order of preference for the default view
PREFERRED = ["gsw", "simulated"]

DEMO_CFG = RunConfig(
    source="simulated",
    sim_periods=480,
    seed=42,
    fast=True,
    out_dir=os.path.join(tempfile.gettempdir(), "vasicek3f_demo"),
    make_plots=False,
    verbose=False,
)


def saved_runs() -> dict[str, str]:
    """name -> folder for every results/<name>/ that contains a bundle."""
    found = {}
    for pattern in (BUNDLE_FILE, LEGACY_BUNDLE_FILE):
        for p in glob.glob(os.path.join(ROOT, "results", "*", pattern)):
            found.setdefault(os.path.basename(os.path.dirname(p)), os.path.dirname(p))
    order = [n for n in PREFERRED if n in found] + sorted(n for n in found if n not in PREFERRED)
    return {n: found[n] for n in order}


def run_label(name: str, folder: str) -> str:
    labels = {
        "gsw": "US Treasuries (GSW) — full estimation",
        "simulated": "Simulated data — known true parameters",
    }
    return labels.get(name, name)


@st.cache_resource(show_spinner=False)
def cached_load(folder: str, mtime: float) -> dict:
    return load_bundle(folder)


@st.cache_resource(show_spinner=False)
def cached_run(cfg_json: str) -> dict:
    return run(RunConfig(**json.loads(cfg_json)))


def load_saved(folder: str) -> dict:
    path = os.path.join(folder, BUNDLE_FILE)
    if not os.path.exists(path):
        path = os.path.join(folder, LEGACY_BUNDLE_FILE)
    return cached_load(folder, os.path.getmtime(path))


def demo_bundle() -> dict:
    return cached_run(json.dumps(asdict(DEMO_CFG)))


def save_upload(upload, name: str) -> Optional[str]:
    """Persist a Streamlit upload to a temp file; return its path (None if nothing uploaded)."""
    if upload is None:
        return None
    d = os.path.join(tempfile.gettempdir(), "vasicek3f_uploads")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + os.path.splitext(upload.name)[1].lower())
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path


@dataclass
class Ctx:
    """Everything the views need, derived once from a results bundle."""

    bundle: dict
    params: VasicekParams
    yields: pd.DataFrame
    tau: np.ndarray
    dec: dict[str, pd.DataFrame]  # decomposition shown (MLE or bias-corrected P)
    dec_mle: dict[str, pd.DataFrame]
    dec_bc: Optional[dict[str, pd.DataFrame]]
    use_bc: bool
    states: pd.DataFrame
    tp_m: float
    fast: bool
    is_sim: bool
    is_us: bool
    dt: float

    @property
    def est(self) -> dict:
        return self.bundle["estimation"]

    @property
    def cfg(self) -> dict:
        return self.bundle["config"]

    @property
    def tp(self) -> pd.Series:
        return self.dec["term_premium"][self.tp_m]

    @property
    def tp_label(self) -> str:
        return "TP (bias-corrected P)" if self.use_bc else "TP (MLE)"

    @property
    def bands(self) -> Optional[dict]:
        return self.bundle.get("tp_bands")

    @property
    def benchmarks(self) -> dict[str, pd.Series]:
        return {k: v for k, v in (self.bundle.get("benchmarks") or {}).items() if v is not None}


def make_ctx(bundle: dict, tp_m: float, use_bc: bool, fast: bool) -> Ctx:
    dec_bc = bundle.get("decomposition_bc")
    use_bc = bool(use_bc and dec_bc is not None)
    return Ctx(
        bundle=bundle,
        params=bundle["params"],
        yields=bundle["yields"],
        tau=np.asarray(bundle["maturities"], float),
        dec=dec_bc if use_bc else bundle["decomposition"],
        dec_mle=bundle["decomposition"],
        dec_bc=dec_bc,
        use_bc=use_bc,
        states=bundle["states"],
        tp_m=float(tp_m),
        fast=fast,
        is_sim=bundle["config"]["source"] == "simulated",
        is_us=bundle["config"]["source"] == "gsw",
        dt=bundle["config"].get("dt", 1 / 12),
    )


def pct(x):
    return x * 100.0


def explain(text: str) -> None:
    with st.expander("How this connects to the model"):
        st.markdown(text)


def csv_download(df, filename: str, label: str = "Download CSV") -> None:
    data = df.to_csv().encode("utf-8")
    st.download_button(label, data, file_name=filename, mime="text/csv", key=f"dl_{filename}")
