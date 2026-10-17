from pathlib import Path
import pandas as pd


def load_xpt(path, columns=None):
    df = pd.read_sas(Path(path), format="xport")
    # The pandas XPORT reader can decode SAS zero as 2**-260.
    df = df.replace(2.0 ** -260, 0.0)
    return df if columns is None else df[columns].copy() 
