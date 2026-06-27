# /**
#   ******************************************************************************
#   * @file        wheelhouse_smoke.py
#   * @author      Egor Izmaylov
#   * @brief       Verifies cibuildwheel wheelhouse artifacts for runtime wheel releases.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.package_smoke import _assert_runtime_wheel


def _wheel_tags(wheel: Path) -> tuple[str, str, str]:
    parts = wheel.name.removesuffix(".whl").split("-")
    if len(parts) < 5:
        raise RuntimeError(f"invalid wheel filename: {wheel.name}")
    return parts[-3], parts[-2], parts[-1]


def inspect_wheelhouse(
    wheelhouse: Path,
    required_python_tags: list[str] | None = None,
    required_platform: str = "manylinux",
) -> list[dict[str, str]]:
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"wheelhouse has no wheels: {wheelhouse}")

    inspected = []
    python_tags = set()
    for wheel in wheels:
        _assert_runtime_wheel(wheel)
        python_tag, abi_tag, platform_tag = _wheel_tags(wheel)
        if required_platform and required_platform not in platform_tag:
            raise RuntimeError(f"wheel platform tag must contain {required_platform!r}: {wheel.name}")
        python_tags.update(python_tag.split("."))
        inspected.append(
            {
                "wheel": wheel.name,
                "python_tag": python_tag,
                "abi_tag": abi_tag,
                "platform_tag": platform_tag,
            }
        )

    missing_tags = sorted(set(required_python_tags or []) - python_tags)
    if missing_tags:
        raise RuntimeError("wheelhouse is missing required Python tags: " + ", ".join(missing_tags))
    return inspected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect cibuildwheel wheelhouse artifacts.")
    parser.add_argument("wheelhouse", nargs="?", default="wheelhouse", help="Directory containing built wheels.")
    parser.add_argument("--require-python-tag", action="append", default=[], help="Require a Python tag such as cp312.")
    parser.add_argument("--require-platform", default="manylinux", help="Substring required in every platform tag.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inspected = inspect_wheelhouse(
            Path(args.wheelhouse),
            required_python_tags=args.require_python_tag,
            required_platform=args.require_platform,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for wheel in inspected:
        print(
            f"OK {wheel['wheel']} "
            f"python={wheel['python_tag']} abi={wheel['abi_tag']} platform={wheel['platform_tag']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
