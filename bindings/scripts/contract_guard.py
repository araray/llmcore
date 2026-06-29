#!/usr/bin/env python3
"""Contract-vs-reality guard (spec §19).

Statically verifies — via ``ast`` only, with **no import** of llmcore (so it
runs in any CI without confy or heavy deps) — that every symbol the bridge
depends on still exists in the llmcore source with the assumed shape:

* methods/functions expose the required parameter names (subset check),
* their return annotation contains the expected type token (when specified),
* model classes declare the required fields,
* enum classes declare the required members.

Exit code is non-zero on any drift, with a precise per-symbol diagnosis.

Usage::

    python bindings/scripts/contract_guard.py            # uses defaults
    python bindings/scripts/contract_guard.py --map M --repo-root R
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import yaml


def _load_symbols(map_path: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(map_path).read_text(encoding="utf-8"))
    return list(data.get("symbols", []))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_func(body: list[ast.stmt], name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _arg_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    names = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _member_names(cls: ast.ClassDef) -> list[str]:
    out: list[str] = []
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.append(target.id)
    return out


def _check_fn(sym: dict[str, Any], fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    errors: list[str] = []
    names = _arg_names(fn)
    for param in sym.get("params_include", []):
        if param not in names:
            errors.append(f"{sym['id']}: missing parameter '{param}' (have: {names})")
    expected = sym.get("returns_contains")
    if expected:
        ret = ast.unparse(fn.returns) if fn.returns is not None else ""
        if expected not in ret:
            errors.append(f"{sym['id']}: return annotation '{ret}' lacks '{expected}'")
    return errors


def check_symbol(sym: dict[str, Any], repo_root: str) -> list[str]:
    """Return a list of drift diagnostics for ``sym`` (empty == OK)."""
    path = Path(repo_root) / sym["file"]
    if not path.exists():
        return [f"{sym['id']}: source file not found: {sym['file']}"]
    tree = _parse(path)
    kind = sym["kind"]

    if kind == "function":
        fn = _find_func(tree.body, sym["name"])
        if fn is None:
            return [f"{sym['id']}: function '{sym['name']}' not found in {sym['file']}"]
        return _check_fn(sym, fn)

    if kind == "method":
        cls = _find_class(tree, sym["class"])
        if cls is None:
            return [f"{sym['id']}: class '{sym['class']}' not found in {sym['file']}"]
        fn = _find_func(cls.body, sym["name"])
        if fn is None:
            return [f"{sym['id']}: method '{sym['class']}.{sym['name']}' not found"]
        return _check_fn(sym, fn)

    if kind in ("class", "enum_class"):
        cls = _find_class(tree, sym["name"])
        if cls is None:
            return [f"{sym['id']}: class '{sym['name']}' not found in {sym['file']}"]
        members = _member_names(cls)
        errors: list[str] = []
        for field in sym.get("fields_include", []):
            if field not in members:
                errors.append(
                    f"{sym['id']}: missing field '{field}' (have: {sorted(members)})"
                )
        for member in sym.get("members_include", []):
            if member not in members:
                errors.append(
                    f"{sym['id']}: missing enum member '{member}' (have: {sorted(members)})"
                )
        return errors

    return [f"{sym['id']}: unknown kind '{kind}'"]


def run(map_path: str, repo_root: str) -> list[str]:
    """Check all mapped symbols; return the aggregated list of diagnostics."""
    errors: list[str] = []
    for sym in _load_symbols(map_path):
        errors.extend(check_symbol(sym, repo_root))
    return errors


def _default_map() -> str:
    return str(Path(__file__).resolve().parents[1] / "contract_map.yaml")


def _default_repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llmcore bridge contract guard")
    parser.add_argument("--map", default=_default_map(), help="path to contract_map.yaml")
    parser.add_argument("--repo-root", default=_default_repo_root(), help="llmcore repo root")
    args = parser.parse_args(argv)

    errors = run(args.map, args.repo_root)
    if errors:
        print(f"CONTRACT GUARD: FAIL ({len(errors)} issue(s))")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("CONTRACT GUARD: OK — all mapped symbols present and consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
