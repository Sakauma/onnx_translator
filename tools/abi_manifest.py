# /**
#   ******************************************************************************
#   * @file        abi_manifest.py
#   * @author      Egor Izmaylov
#   * @brief       Generates and checks the public C ABI manifest for tensor_ops.h.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADER = ROOT / "tensor_ops" / "tensor_ops.h"
DEFAULT_MANIFEST = ROOT / "docs" / "abi_manifest.json"


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//.*", "", source)
    return source


def _normalize_type(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.replace(" *", "*").replace("* ", "* ")
    value = value.replace("const ", "const ")
    return value.strip()


def _split_args(args: str) -> list[str]:
    args = args.strip()
    if not args or args == "void":
        return []
    return [_normalize_type(arg) for arg in args.split(",")]


def _parse_enum(clean_source: str, name: str) -> list[str]:
    match = re.search(r"typedef\s+enum\s*\{(?P<body>.*?)\}\s*" + re.escape(name) + r"\s*;", clean_source, re.DOTALL)
    if not match:
        raise ValueError(f"enum {name} not found")
    values = []
    for item in match.group("body").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item.split("=", 1)[0].strip())
    return values


def _parse_structs(clean_source: str) -> dict[str, list[str]]:
    structs: dict[str, list[str]] = {}
    for match in re.finditer(r"typedef\s+struct\s*\{(?P<body>.*?)\}\s*(?P<name>\w+)\s*;", clean_source, re.DOTALL):
        fields = []
        for raw_field in match.group("body").split(";"):
            field = _normalize_type(raw_field)
            if field:
                fields.append(field)
        structs[match.group("name")] = fields
    return dict(sorted(structs.items()))


def _parse_functions(clean_source: str) -> dict[str, dict[str, object]]:
    functions: dict[str, dict[str, object]] = {}
    for statement in clean_source.split(";"):
        normalized = _normalize_type(statement)
        if "(" not in normalized or ")" not in normalized:
            continue
        if normalized.startswith("typedef ") or normalized.startswith("#"):
            continue
        match = re.match(r"(?P<return>.+?)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>.*)\)$", normalized)
        if not match:
            continue
        name = match.group("name")
        functions[name] = {
            "return": _normalize_type(match.group("return")),
            "args": _split_args(match.group("args")),
        }
    return dict(sorted(functions.items()))


def build_manifest(header_path: Path = DEFAULT_HEADER) -> dict[str, object]:
    source = header_path.read_text(encoding="utf-8")
    clean_source = _strip_comments(source)
    functions = _parse_functions(clean_source)
    structs = _parse_structs(clean_source)
    data_type = _parse_enum(clean_source, "DataType")
    return {
        "schema_version": 1,
        "header": str(header_path.relative_to(ROOT)) if header_path.is_relative_to(ROOT) else str(header_path),
        "data_type": data_type,
        "structs": structs,
        "functions": functions,
        "counts": {
            "data_type": len(data_type),
            "structs": len(structs),
            "functions": len(functions),
        },
    }


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_manifests(expected: dict[str, object], actual: dict[str, object]) -> list[str]:
    failures = []
    for key in ["data_type", "structs", "functions"]:
        if expected.get(key) != actual.get(key):
            failures.append(f"ABI {key} changed")
    return failures


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or check the public C ABI manifest.")
    parser.add_argument("--header", default=str(DEFAULT_HEADER), help="Path to tensor_ops.h.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to ABI manifest JSON.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write the current ABI manifest.")
    mode.add_argument("--check", action="store_true", help="Check the current ABI against the manifest.")
    mode.add_argument("--print", action="store_true", help="Print the current ABI manifest.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    header_path = Path(args.header)
    manifest_path = Path(args.manifest)
    actual = build_manifest(header_path)

    if args.write:
        write_manifest(manifest_path, actual)
        print(f"Wrote ABI manifest: {manifest_path}")
        return 0

    if args.print:
        print(json.dumps(actual, indent=2, sort_keys=True))
        return 0

    if not manifest_path.exists():
        print(f"ERROR: ABI manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    expected = _load_manifest(manifest_path)
    failures = compare_manifests(expected, actual)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(f"Update intentionally with: {sys.executable} tools/abi_manifest.py --write", file=sys.stderr)
        return 1
    print("C ABI manifest check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
