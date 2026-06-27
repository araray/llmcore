"""The contract guard must pass against the live llmcore source, and must fail
loudly when a mapped symbol is wrong (proving it is not vacuously green)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "bindings" / "scripts" / "contract_guard.py"
MAP_PATH = REPO_ROOT / "bindings" / "contract_map.yaml"


def _load_guard():
    spec = importlib.util.spec_from_file_location("contract_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_contract_guard_passes_against_real_source():
    guard = _load_guard()
    errors = guard.run(str(MAP_PATH), str(REPO_ROOT))
    assert errors == [], "contract drift detected:\n" + "\n".join(errors)


def test_contract_guard_detects_missing_method(tmp_path):
    guard = _load_guard()
    bogus = tmp_path / "bogus_map.yaml"
    bogus.write_text(
        "symbols:\n"
        "  - id: bogus.method\n"
        "    file: src/llmcore/api.py\n"
        "    kind: method\n"
        "    class: LLMCore\n"
        "    name: this_method_does_not_exist_anywhere\n",
        encoding="utf-8",
    )
    errors = guard.run(str(bogus), str(REPO_ROOT))
    assert any("this_method_does_not_exist_anywhere" in e for e in errors)


def test_contract_guard_detects_missing_param(tmp_path):
    guard = _load_guard()
    bogus = tmp_path / "bogus_param.yaml"
    bogus.write_text(
        "symbols:\n"
        "  - id: bogus.param\n"
        "    file: src/llmcore/api.py\n"
        "    kind: method\n"
        "    class: LLMCore\n"
        "    name: chat\n"
        "    params_include: [definitely_not_a_real_param]\n",
        encoding="utf-8",
    )
    errors = guard.run(str(bogus), str(REPO_ROOT))
    assert any("definitely_not_a_real_param" in e for e in errors)
