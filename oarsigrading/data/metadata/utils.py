import pandas as pd
from sas7bdat import SAS7BDAT


def read_sas7bdat(fpath):
    rows = []
    with SAS7BDAT(fpath) as f:
        for row in f:
            rows.append(row)
    return pd.DataFrame(rows[1:], columns=rows[0])
