"""Batch analysis of every particle file in a folder; no GUI or server is needed.

By default each particle gets its own CSV next to the input file, with the
same base name (``particle_01.stl`` -> ``particle_01.csv``). A file that
fails to load or analyse is reported and skipped; the rest of the batch runs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Optional

from .analysis import AXES_LABELS, OUTPUT_UNITS, UNIT_SCALES, AnalysisResult, analyze_file, unit_label
from .export import make_export_path
from .io import SUPPORTED_EXTENSIONS

SUMMARY_NAME = "pyshape_summary.csv"


@dataclass
class BatchItem:
    """Outcome for one input file: a result and CSV path, or an error."""

    path: Path
    result: Optional[AnalysisResult] = None
    csv_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class BatchResult:
    """All batch outcomes, in processing order, plus the summary CSV path if written."""

    folder: Path
    items: list[BatchItem] = field(default_factory=list)
    summary_path: Optional[Path] = None

    @property
    def succeeded(self) -> list[BatchItem]:
        return [item for item in self.items if item.ok]

    @property
    def failed(self) -> list[BatchItem]:
        return [item for item in self.items if not item.ok]


def find_particle_files(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    """Supported mesh files in ``folder`` (and subfolders if ``recursive``), sorted."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: (p.relative_to(folder).parts[:-1], p.name.lower()),
    )


def summary_csv_text(
    results: Iterable[AnalysisResult],
    keys: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> str:
    """One row per particle and one column per metric, with units in the header.

    Columns are ``keys`` or, by default, every default-selected metric of the
    first result. Unavailable values are left empty. ``names`` replaces the
    ``source_file`` column (default: each result's ``source_name``).
    """
    results = list(results)
    names = [r.source_name or "" for r in results] if names is None else list(names)
    if len(names) != len(results):
        raise ValueError("names must match results")
    if not results:
        raise ValueError("No results to summarise")
    first = results[0]
    if keys is None:
        keys = [m.key for m in first.metrics if m.default_selected]
    keys = list(keys)
    known = {m.key for m in first.metrics}
    for key in keys:
        if key not in known:
            raise ValueError(f"Unknown metric: {key}")
    header = ["source_file"]
    for key in keys:
        unit = unit_label(key, first.input_units, ascii=True, output_units=first.output_units)
        header.append(f"{key} [{unit}]" if unit else key)
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    for name, result in zip(names, results):
        values = result.values
        writer.writerow(
            [name]
            + ["" if values.get(key) is None else values[key] for key in keys]
        )
    return stream.getvalue()


def analyze_folder(
    folder: str | Path,
    *,
    units: str = "as-is",
    axes: str = "pca",
    output_units: str = "m",
    keys: Optional[Iterable[str]] = None,
    suffix: str = "",
    write_csv: bool = True,
    avoid_overwrite: bool = False,
    recursive: bool = False,
    summary: bool = False,
    progress: Optional[Callable[[int, int, BatchItem], None]] = None,
) -> BatchResult:
    """Analyse every supported particle file in ``folder``.

    ``units``, ``axes`` and ``output_units`` are as in :func:`pyshape.analyze_file`
    and apply to every file. With ``write_csv`` (default) each result is
    written next to its input as ``<stem><suffix>.csv`` (``keys`` selects the
    metrics, as in :meth:`AnalysisResult.to_csv`). Existing CSVs are
    overwritten unless ``avoid_overwrite``, which appends ``_1``, ``_2``, ...
    Inputs that share a stem (``a.stl`` and ``a.obj``) never overwrite each
    other's CSV. ``summary`` also writes ``pyshape_summary.csv`` (one row per
    particle) in ``folder``. ``progress(done, total, item)`` is called after
    each file. Per-file errors are recorded in the returned items, not raised.
    """
    if units not in UNIT_SCALES:
        raise ValueError(f"Unknown units '{units}'. Choose one of: {', '.join(UNIT_SCALES)}")
    if axes not in AXES_LABELS:
        raise ValueError(f"Unknown axes method '{axes}'. Choose one of: {', '.join(AXES_LABELS)}")
    if output_units != "same" and output_units not in OUTPUT_UNITS:
        raise ValueError(
            f"Unknown output units '{output_units}'. Choose same or {', '.join(OUTPUT_UNITS)}"
        )
    keys = None if keys is None else list(keys)
    folder = Path(folder)
    files = find_particle_files(folder, recursive=recursive)
    batch = BatchResult(folder)
    written: set[Path] = set()
    for index, path in enumerate(files, 1):
        item = BatchItem(path)
        try:
            item.result = analyze_file(path, units=units, axes=axes, output_units=output_units)
            if write_csv:
                text = item.result.to_csv(keys)
                target = make_export_path(path, suffix, avoid_overwrite)
                n = 1
                while target in written or (avoid_overwrite and target.exists()):
                    target = path.parent / f"{path.stem}{suffix}_{n}.csv"
                    n += 1
                with target.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(text)
                written.add(target)
                item.csv_path = target
        except Exception as exc:  # one bad file must not stop the batch
            item.error = f"{type(exc).__name__}: {exc}"
        batch.items.append(item)
        if progress is not None:
            progress(index, len(files), item)
    if summary and batch.succeeded:
        batch.summary_path = folder / SUMMARY_NAME
        text = summary_csv_text(
            [item.result for item in batch.succeeded], keys,
            [item.path.relative_to(folder).as_posix() for item in batch.succeeded],
        )
        with batch.summary_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
    return batch


def main(argv: Optional[list[str]] = None) -> int:
    """Command line: ``pyshape-batch FOLDER [options]``; exit code 1 if any file failed."""
    parser = argparse.ArgumentParser(
        description="Analyse every particle file in a folder and write one CSV per particle."
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("--units", default="as-is", choices=list(UNIT_SCALES),
                        help="units of the coordinates in the files (default: as-is)")
    parser.add_argument("--output-units", default="m", choices=["same", *OUTPUT_UNITS],
                        help="length unit of the results (default: m; ignored for as-is)")
    parser.add_argument("--axes", default="pca", choices=list(AXES_LABELS),
                        help="principal dimensions S, I, L (default: pca)")
    parser.add_argument("--keys", help="comma-separated metric keys (default: standard selection)")
    parser.add_argument("--suffix", default="", help="appended to each CSV name (default: none)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="keep existing CSVs and number the new ones")
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    parser.add_argument("--summary", action="store_true",
                        help=f"also write {SUMMARY_NAME} with one row per particle")
    args = parser.parse_args(argv)

    def report(done: int, total: int, item: BatchItem) -> None:
        name = item.path.relative_to(args.folder) if item.path.is_relative_to(args.folder) else item.path
        outcome = f"-> {item.csv_path.name}" if item.ok and item.csv_path else f"FAILED: {item.error}"
        print(f"[{done}/{total}] {name} {outcome}", flush=True)

    try:
        batch = analyze_folder(
            args.folder, units=args.units, axes=args.axes, output_units=args.output_units,
            keys=None if args.keys is None else args.keys.split(","), suffix=args.suffix,
            avoid_overwrite=args.no_overwrite, recursive=args.recursive,
            summary=args.summary, progress=report,
        )
    except (ValueError, NotADirectoryError) as exc:
        parser.error(str(exc))
    if not batch.items:
        print(f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) in {args.folder}")
    else:
        print(f"Done: {len(batch.succeeded)} analysed, {len(batch.failed)} failed.")
    if batch.summary_path is not None:
        print(f"Summary: {batch.summary_path}")
    return 1 if batch.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
