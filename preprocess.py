# preprocess.py: validate and clean an arbitrary CSV before it reaches Spark.
#
# Runs on the driver with pandas so every problem surfaces as a readable
# DataError, instead of a stack trace from inside a Spark task.
import re
import numpy as np
import pandas as pd

SEPARATORS = [",", ";", "\t"]
INDEX_COL = re.compile(r"^Unnamed: \d+$")   # pandas' name for a blank header cell


class DataError(ValueError):
    """The dataset cannot be trained on. The message is meant for end users."""


def _short(v, n=30):
    s = repr(v) if isinstance(v, str) else f"{v:g}"
    return s if len(s) <= n else s[:n - 3] + "..."


def _examples(values, k=5):
    values = list(values)
    shown = ", ".join(_short(v) for v in values[:k])
    return shown + (", ..." if len(values) > k else "")


def _detect_sep(path):
    # The header line is the most reliable place to count separators.
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            first = f.readline()
    except UnicodeDecodeError:
        raise DataError("The file is not UTF-8 text. Save it as CSV (UTF-8) and retry.")
    counts = {s: first.count(s) for s in SEPARATORS}
    sep = max(counts, key=counts.get)
    return sep if counts[sep] > 0 else ","


def _read(path, nrows=None):
    """Read the CSV, auto-detecting the separator and dropping index columns."""
    sep = _detect_sep(path)
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", nrows=nrows)
    except pd.errors.EmptyDataError:
        raise DataError("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DataError(f"Could not read the file as CSV: {e}")
    ignored = [c for c in df.columns if INDEX_COL.match(str(c))]
    return df.drop(columns=ignored), ignored


def read_header(path):
    """Usable column names (index columns removed), for picking the label."""
    df, _ = _read(path, nrows=0)
    return list(df.columns)


def _raise(problems):
    if len(problems) == 1:
        raise DataError(problems[0])
    raise DataError(f"Found {len(problems)} problems:\n" +
                    "\n".join(f"- {p}" for p in problems))


def clean_csv(src, label_col, dst):
    """
    Validate `src`, write a cleaned all-float, comma-separated CSV (header
    kept) to `dst`.

    Before cleaning, every structural problem is collected and reported in
    one DataError:
      - `label_col` missing, non-numeric, or holding values other than 0/1
      - non-numeric feature columns (named in the message)
      - no feature columns
    Then rows with missing or infinite values are dropped, and the result
    must still have rows and both label classes.

    Separators , ; and tab are detected automatically. Columns pandas names
    "Unnamed: N" (a blank header cell, usually a saved row index) are ignored
    and reported. Feature standardization is NOT done here; it runs in Spark
    (train.standardize).

    Returns a summary dict: rows_in, rows_dropped, rows, n_features,
    features, ignored_columns.
    """
    df, ignored = _read(src)
    problems = []

    if label_col not in df.columns:
        problems.append(f"Label column {label_col!r} is not in the file's header.")
    else:
        label = df[label_col].dropna()
        if not pd.api.types.is_numeric_dtype(label):
            problems.append(
                f"Label column {label_col!r} must contain only 0 and 1. "
                f"Found non-numeric values: {_examples(label.unique())}.")
        else:
            bad = [v for v in np.unique(label) if v not in (0, 1)]
            if bad:
                problems.append(
                    f"Label column {label_col!r} must contain only 0 and 1. "
                    f"Found other values: {_examples(bad)}.")

    features = [c for c in df.columns if c != label_col]
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(
            f"These columns are not numeric: {_examples(non_numeric, k=10)}. "
            "Remove or encode them before uploading. A column with even one "
            "text value (for example '?' or 'N/A') counts as non-numeric.")
    if not features:
        problems.append("The file has no feature columns besides the label.")

    if problems:
        _raise(problems)

    rows_in = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    rows_dropped = rows_in - len(df)
    if df.empty:
        raise DataError(
            f"All {rows_in:,} rows have at least one missing value, "
            "so nothing is left to train on.")

    labels = np.unique(df[label_col])
    if len(labels) < 2:
        after = " after dropping rows with missing values" if rows_dropped else ""
        raise DataError(
            f"Label column {label_col!r} only contains {labels[0]:g}{after}. "
            "Both classes (0 and 1) are needed to train a classifier.")

    df.astype(np.float64).to_csv(dst, index=False)

    return {
        "rows_in": rows_in,
        "rows_dropped": rows_dropped,
        "rows": len(df),
        "n_features": len(features),
        "features": features,
        "ignored_columns": ignored,
    }
