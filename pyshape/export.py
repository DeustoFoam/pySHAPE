"""CSV output shared by Python and user-interface clients."""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional, Tuple


def make_export_path(input_path: str | Path, suffix: str, avoid_overwrite: bool) -> Path:
    """Choose a CSV path, optionally appending a progressive counter.

    This chooses a name only; it does not reserve the path against other writers.
    """
    input_path = Path(input_path)
    base_name = f"{input_path.stem}{suffix}"
    candidate = input_path.parent / f"{base_name}.csv"
    if not avoid_overwrite or not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = input_path.parent / f"{base_name}_{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def csv_text(
    rows: Iterable[Tuple[object, ...]], source_name: Optional[str] = None
) -> str:
    """Serialize parameter/value/unit rows with an optional source-file row.

    Rows are ``(parameter, value)`` or ``(parameter, value, unit)``; the unit
    column is left empty for dimensionless values and two-element rows.
    """
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["parameter", "value", "unit"])
    if source_name is not None:
        writer.writerow(["source_file", source_name, ""])
    for row in rows:
        row = tuple(row)
        writer.writerow(row if len(row) == 3 else (*row, ""))
    return stream.getvalue()


def write_csv(
    path: str | Path,
    rows: Iterable[Tuple[object, ...]],
    source_name: Optional[str] = None,
) -> None:
    """Write UTF-8 CSV; an existing destination is overwritten."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        stream.write(csv_text(rows, source_name))
