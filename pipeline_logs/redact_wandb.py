"""Replace the hard-coded W&B key with an environment lookup.

Python call sites build ``ENV = dict(os.environ, ..., WANDB_API_KEY="wandb_v1_...")``
so the value becomes ``os.environ["WANDB_API_KEY"]`` (a KeyError is the clearest
failure when the caller forgot to export it).  Shell call sites use
``export WANDB_API_KEY=wandb_v1_...`` and become ``${WANDB_API_KEY:?...}``.
"""

import re
import sys
from pathlib import Path

KEY = re.compile(r"wandb_v1_[A-Za-z0-9_]+")
PY_ASSIGN = re.compile(r"""WANDB_API_KEY=(["'])wandb_v1_[A-Za-z0-9_]+(["'])""")
SH_EXPORT = re.compile(r"export WANDB_API_KEY=wandb_v1_[A-Za-z0-9_]+")
SH_PREFIX = re.compile(r"WANDB_API_KEY=wandb_v1_[A-Za-z0-9_]+")


def convert(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not KEY.search(text):
        return False
    if path.suffix == ".py":
        new = PY_ASSIGN.sub('WANDB_API_KEY=os.environ["WANDB_API_KEY"]', text)
    else:
        new = SH_EXPORT.sub('export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"', text)
        new = SH_PREFIX.sub('WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"', new)
    if KEY.search(new):
        print(f"!! leftover key in {path}", file=sys.stderr)
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    changed = [p for p in paths if convert(p)]
    for p in changed:
        print(f"redacted {p}")
    print(f"{len(changed)}/{len(paths)} files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
