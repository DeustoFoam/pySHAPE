"""The audit baseline (tools/baseline/) still describes the code: see docs/audit.md."""
import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit.py"


def _audit():
    spec = importlib.util.spec_from_file_location("pyshape_audit", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    return module


def test_existing_behaviour_matches_audit_baseline(capsys):
    pytest.importorskip("scipy")
    status = _audit().run_check(rtol=1e-9, atol=1e-12, strict=False)
    report = capsys.readouterr().out
    assert status == 0, "python tools/audit.py check reports breaking changes:\n" + report
