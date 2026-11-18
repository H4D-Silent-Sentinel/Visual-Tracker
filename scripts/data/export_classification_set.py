"""Freeze manually labeled classification frames into a self-contained test collection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from classification_data import export_collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "our_data")
    args = parser.parse_args(argv)
    try:
        exported = export_collection(args.root, args.name)
        print(json.dumps(exported, indent=2))
        print(exported["command"])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
