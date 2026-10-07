"""Convenience runner — executes every numbered script in order.

Each script is self-contained and can also be invoked individually when
iterating on a single figure.
"""

from __future__ import annotations

import importlib
from pathlib import Path

SCRIPTS = sorted(
    p.stem for p in Path(__file__).parent.glob("[0-9][0-9]_*.py")
)


def main() -> None:
    for stem in SCRIPTS:
        print(f"\n=== {stem} ===")
        mod = importlib.import_module(stem)
        if hasattr(mod, "main"):
            mod.main()


if __name__ == "__main__":
    main()
