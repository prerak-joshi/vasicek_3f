"""
Version-independent serialisation of results bundles.

Pickle files are tied to the exact pandas/numpy versions that wrote them, which
breaks when a deployment (e.g. Streamlit Community Cloud) installs different
versions. Bundles are therefore stored as gzipped JSON with explicit encodings for
DataFrames, Series, arrays, timestamps and the model's parameter dataclasses.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .params import VasicekParams


def _index_encode(idx: pd.Index) -> dict:
    if isinstance(idx, pd.DatetimeIndex):
        return {"kind": "datetime", "values": [d.isoformat() for d in idx], "name": idx.name}
    return {"kind": "plain", "values": [_encode(v) for v in idx.tolist()], "name": _encode(idx.name)}


def _index_decode(d: dict) -> pd.Index:
    if d["kind"] == "datetime":
        return pd.DatetimeIndex(pd.to_datetime(d["values"]), name=d["name"])
    return pd.Index(d["values"], name=d["name"])


def _encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else {"__float__": repr(f)}
    if isinstance(obj, VasicekParams):
        return {"__params__": {k: _encode(v) for k, v in asdict(obj).items()}}
    if isinstance(obj, pd.DataFrame):
        return {
            "__df__": {
                "index": _index_encode(obj.index),
                "columns": _index_encode(obj.columns),
                "data": [[_encode(v) for v in row] for row in obj.to_numpy(dtype=object)],
                "attrs": {k: _encode(v) for k, v in obj.attrs.items()},
            }
        }
    if isinstance(obj, pd.Series):
        return {
            "__series__": {
                "index": _index_encode(obj.index),
                "name": _encode(obj.name),
                "data": [_encode(v) for v in obj.tolist()],
            }
        }
    if isinstance(obj, np.ndarray):
        return {"__nd__": _encode(obj.tolist()), "dtype": str(obj.dtype)}
    if isinstance(obj, (pd.Timestamp, datetime)):
        return {"__ts__": obj.isoformat()}
    if isinstance(obj, date):
        return {"__date__": obj.isoformat()}
    if is_dataclass(obj):
        return {"__dc__": type(obj).__name__, "fields": {k: _encode(v) for k, v in asdict(obj).items()}}
    if isinstance(obj, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__float__" in obj:
        return float(obj["__float__"])
    if "__params__" in obj:
        return VasicekParams(**{k: _decode(v) for k, v in obj["__params__"].items()})
    if "__df__" in obj:
        d = obj["__df__"]
        df = pd.DataFrame(
            [[_decode(v) for v in row] for row in d["data"]],
            index=_index_decode(d["index"]),
            columns=_index_decode(d["columns"]),
        )
        df = df.infer_objects()
        df.attrs.update({k: _decode(v) for k, v in d.get("attrs", {}).items()})
        return df
    if "__series__" in obj:
        d = obj["__series__"]
        return pd.Series(
            [_decode(v) for v in d["data"]], index=_index_decode(d["index"]), name=_decode(d["name"])
        ).infer_objects()
    if "__nd__" in obj:
        return np.asarray(_decode(obj["__nd__"]), dtype=obj.get("dtype", "float64"))
    if "__ts__" in obj:
        return pd.Timestamp(obj["__ts__"])
    if "__date__" in obj:
        return date.fromisoformat(obj["__date__"])
    if "__dc__" in obj:
        return {k: _decode(v) for k, v in obj["fields"].items()}
    if "__dict__" in obj:
        return {_decode(k): _decode(v) for k, v in obj["__dict__"]}
    return {k: _decode(v) for k, v in obj.items()}


def dump_bundle(bundle: dict, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(_encode(bundle), f, separators=(",", ":"))


def load_bundle_file(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return _decode(json.load(f))
