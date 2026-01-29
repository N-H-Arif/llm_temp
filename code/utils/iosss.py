import json, time, os
from pathlib import Path
import pandas as pd


def now_ms():
    return int(time.time() * 1000)


def ensure_parent(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


class CSVLogger:
    def __init__(self, path, header):
        self.path = Path(path)
        ensure_parent(self.path)
        self.header = header
        if not self.path.exists():
            pd.DataFrame(columns=header).to_csv(self.path, index=False)

    def append(self, row_dict):
        df = pd.DataFrame([row_dict], columns=self.header)
        df.to_csv(self.path, mode='a', header=False, index=False)
