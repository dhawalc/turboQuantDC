"""Minimal dependency-free GGUF reader: metadata, tensor index, F32 tensor data."""
from __future__ import annotations
import struct, numpy as np
from typing import Any, Dict, Tuple

# gguf metadata value types
U8,I8,U16,I16,U32,I32,F32,BOOL,STR,ARR,U64,I64,F64 = range(13)
_FMT = {U8:"<B",I8:"<b",U16:"<H",I16:"<h",U32:"<I",I32:"<i",F32:"<f",
        BOOL:"<?",U64:"<Q",I64:"<q",F64:"<d"}
_SZ  = {U8:1,I8:1,U16:2,I16:2,U32:4,I32:4,F32:4,BOOL:1,U64:8,I64:8,F64:8}

# ggml tensor types we need to interpret (biases/norms are F32 or F16)
GGML_F32, GGML_F16 = 0, 1


class GGUF:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {magic!r}")
        self.version = self._u32()
        self.n_tensors = self._u64()
        self.n_kv = self._u64()
        self.meta: Dict[str, Any] = {}
        for _ in range(self.n_kv):
            k = self._str()
            self.meta[k] = self._value()
        self.tensors: Dict[str, Tuple[tuple, int, int]] = {}  # name -> (dims, ggml_type, offset)
        for _ in range(self.n_tensors):
            name = self._str()
            nd = self._u32()
            dims = tuple(self._u64() for _ in range(nd))
            ttype = self._u32()
            off = self._u64()
            self.tensors[name] = (dims, ttype, off)
        align = int(self.meta.get("general.alignment", 32))
        pos = self.f.tell()
        self.data_start = pos + (-pos) % align

    # --- primitive readers ---
    def _u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def _u64(self): return struct.unpack("<Q", self.f.read(8))[0]

    def _str(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def _scalar(self, t):
        return struct.unpack(_FMT[t], self.f.read(_SZ[t]))[0]

    def _value(self):
        t = self._u32()
        if t == STR:
            return self._str()
        if t == ARR:
            et = self._u32()
            n = self._u64()
            if et == STR:
                return [self._str() for _ in range(n)]
            if et == ARR:
                return [self._value() for _ in range(n)]
            raw = self.f.read(_SZ[et] * n)
            return list(struct.unpack("<" + _FMT[et][1] * n, raw))
        return self._scalar(t)

    # --- tensor data (only for uncompressed types) ---
    def read_tensor(self, name: str) -> np.ndarray:
        dims, ttype, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        if ttype == GGML_F32:
            dt, itemsize = np.float32, 4
        elif ttype == GGML_F16:
            dt, itemsize = np.float16, 2
        else:
            raise NotImplementedError(
                f"{name}: ggml type {ttype} is quantized; this reader only "
                f"handles F32/F16 (sufficient for biases and norm weights)"
            )
        self.f.seek(self.data_start + off)
        buf = self.f.read(n * itemsize)
        # GGUF dims are in reverse (ne[0] fastest); numpy wants C order
        return np.frombuffer(buf, dtype=dt).reshape(tuple(reversed(dims))).astype(np.float32)

    def arch(self) -> str:
        return self.meta.get("general.architecture", "?")

    def a(self, key: str, default=None):
        """Architecture-scoped metadata lookup, e.g. a('attention.head_count')."""
        return self.meta.get(f"{self.arch()}.{key}", default)

    def close(self):
        self.f.close()
