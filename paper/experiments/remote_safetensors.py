"""Read individual tensors out of a HuggingFace safetensors repo via HTTP range requests.

Downloads only the shard header plus the exact bytes of the tensors requested, so
inspecting a handful of small norm vectors costs kilobytes instead of gigabytes.

safetensors layout:  [u64 header_len][JSON header][tensor data]
Offsets in the JSON header are relative to the end of the header.
"""
from __future__ import annotations
import json, struct, urllib.request
import numpy as np

UA = {"User-Agent": "curl/8", "Accept-Encoding": "identity"}


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers=dict(UA))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class RemoteSafetensors:
    def __init__(self, repo: str, shard: str, revision: str = "main"):
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
        n = struct.unpack("<Q", _get(self.url, 0, 7))[0]
        self.header = json.loads(_get(self.url, 8, 8 + n - 1))
        self.data_start = 8 + n

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def info(self, name):
        return self.header[name]

    def tensor(self, name) -> np.ndarray:
        e = self.header[name]
        lo, hi = e["data_offsets"]
        raw = _get(self.url, self.data_start + lo, self.data_start + hi - 1)
        dt = e["dtype"]
        if dt in ("BF16", "F16"):
            a = np.frombuffer(raw, dtype=np.uint16)
            if dt == "BF16":
                a = (a.astype(np.uint32) << 16).view(np.float32)
            else:
                a = a.view(np.float16).astype(np.float32)
        elif dt == "F32":
            a = np.frombuffer(raw, dtype=np.float32)
        elif dt == "F8_E4M3":
            b = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
            sign = (b >> 7) & 1
            exp = (b >> 3) & 0xF
            man = b & 0x7
            val = np.where(exp == 0,
                           (man / 8.0) * 2.0 ** (-6),
                           (1 + man / 8.0) * 2.0 ** (exp.astype(np.int64) - 7))
            a = np.where(sign == 1, -val, val).astype(np.float32)
        else:
            raise NotImplementedError(dt)
        return a.reshape(e["shape"])


def load_index(repo: str, revision: str = "main") -> dict:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors.index.json"
    return json.loads(_get(url))
