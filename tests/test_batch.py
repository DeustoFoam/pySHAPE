import csv
import io
import shutil

import pytest

from pyshape import analyze_file, analyze_folder, find_particle_files
from pyshape.batch import SUMMARY_NAME, main


@pytest.fixture
def folder(tmp_path, stl_path):
    shutil.copy(stl_path, tmp_path / "b_particle.stl")
    shutil.copy(stl_path, tmp_path / "a_particle.STL")
    (tmp_path / "broken.stl").write_text("not STL")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "sub").mkdir()
    shutil.copy(stl_path, tmp_path / "sub" / "c_particle.stl")
    return tmp_path


def rows(path):
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))


def test_finds_supported_files_sorted(folder):
    assert [p.name for p in find_particle_files(folder)] == [
        "a_particle.STL", "b_particle.stl", "broken.stl",
    ]
    assert [p.name for p in find_particle_files(folder, recursive=True)][-1] == "c_particle.stl"
    with pytest.raises(NotADirectoryError):
        find_particle_files(folder / "missing")


def test_writes_one_csv_per_particle_with_same_name(folder, stl_path):
    progress = []
    batch = analyze_folder(folder, units="mm", progress=lambda d, t, i: progress.append((d, t)))
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [item.path.name for item in batch.succeeded] == ["a_particle.STL", "b_particle.stl"]
    assert [item.path.name for item in batch.failed] == ["broken.stl"]
    assert batch.failed[0].error and batch.failed[0].csv_path is None
    expected = analyze_file(stl_path, units="mm")
    for item in batch.succeeded:
        assert item.csv_path == folder / f"{item.path.stem}.csv"
        assert item.result.values == expected.values
        assert rows(item.csv_path)[2:] == rows_from_text(expected.to_csv())[2:]
        assert rows(item.csv_path)[1] == ["source_file", item.path.name, ""]
    assert not (folder / "sub" / "c_particle.csv").exists()
    assert batch.summary_path is None and not (folder / SUMMARY_NAME).exists()


def rows_from_text(text):
    return list(csv.reader(io.StringIO(text)))


def test_overwrite_suffix_and_recursive(folder):
    (folder / "b_particle_SHAPE.csv").write_text("keep me")
    batch = analyze_folder(folder, suffix="_SHAPE", avoid_overwrite=True, recursive=True)
    names = {item.path.name: item.csv_path.name for item in batch.succeeded}
    assert names == {
        "a_particle.STL": "a_particle_SHAPE.csv",
        "b_particle.stl": "b_particle_SHAPE_1.csv",
        "c_particle.stl": "c_particle_SHAPE.csv",
    }
    assert (folder / "b_particle_SHAPE.csv").read_text() == "keep me"
    analyze_folder(folder, suffix="_SHAPE")
    assert (folder / "b_particle_SHAPE.csv").read_text() != "keep me"


def test_same_stem_never_overwrites_within_a_run(tmp_path, stl_path):
    shutil.copy(stl_path, tmp_path / "p.stl")
    (tmp_path / "sub").mkdir()
    shutil.copy(stl_path, tmp_path / "sub" / "p.stl")
    trimesh = pytest.importorskip("trimesh")
    trimesh.load(stl_path).export(tmp_path / "p.obj")  # same stem, same folder
    batch = analyze_folder(tmp_path, recursive=True)
    paths = [item.csv_path for item in batch.succeeded]
    assert len(set(paths)) == len(paths) == 3


def test_summary_and_keys(folder):
    batch = analyze_folder(folder, units="mm", output_units="mm", keys=["volume", "n_faces"],
                           summary=True, recursive=True)
    table = rows(batch.summary_path)
    assert table[0] == ["source_file", "volume [mm^3]", "n_faces"]
    assert [row[0] for row in table[1:]] == ["a_particle.STL", "b_particle.stl", "sub/c_particle.stl"]
    assert float(table[1][1]) == pytest.approx(batch.succeeded[0].result.values["volume"])
    assert [row[0] for row in rows(batch.succeeded[0].csv_path)[2:]] == ["volume", "n_faces"]


def test_unknown_key_fails_each_file(folder):
    batch = analyze_folder(folder, keys=["invalid"])
    assert not batch.succeeded
    assert all("Unknown metric" in item.error for item in batch.items if item.path.name != "broken.stl")


@pytest.mark.parametrize("kwargs", [{"units": "km"}, {"axes": "feret"}, {"output_units": "km"}])
def test_invalid_settings_raise_before_any_file(folder, kwargs):
    with pytest.raises(ValueError):
        analyze_folder(folder, **kwargs)
    assert not list(folder.glob("*.csv"))


def test_command_line(folder, capsys):
    assert main([str(folder), "--units", "mm", "--summary"]) == 1  # broken.stl fails
    out = capsys.readouterr().out
    assert "[1/3] a_particle.STL -> a_particle.csv" in out
    assert "broken.stl FAILED" in out
    assert "Done: 2 analysed, 1 failed." in out
    assert (folder / SUMMARY_NAME).exists()
    (folder / "broken.stl").unlink()
    assert main([str(folder)]) == 0
