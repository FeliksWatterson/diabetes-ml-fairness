from pathlib import Path
import pandas as pd


def load_xpt(path, columns=None):
    df = pd.read_sas(Path(path), format="xport")
    return df if columns is None else df[columns].copy() 