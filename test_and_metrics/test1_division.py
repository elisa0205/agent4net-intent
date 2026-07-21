from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCE_ROOT = Path("test1-results")
SIMPLE_ROOT = Path("test1-simple-results")
STRUCTURE_ROOT = Path("test1-structure-results")

SELECTIONS = {
    "db-cronjob-example": {
        "simple": "gpt-oss-120b_1",
        "structure": "gpt-oss-120b_2",
    },
    "frontend-backend-app": {
        "simple": "gpt-oss-120b_1",
        "structure": "gpt-oss-120b_2",
    },
    "HPA-example": {
        "simple": "gpt-oss-120b_1",
        "structure": "DeepSeek-V4-Flash_1",
    },
    "multi-tenants-example": {
        "simple": "gpt-oss-120b_1",
        "structure": "DeepSeek-V4-Flash_1",
    },
    "php-guestbook-example": {
        "simple": "DeepSeek-V4-Flash_1",
        "structure": "gpt-oss-120b_1",
    },
    "postgres-example": {
        "simple": "DeepSeek-V4-Flash_1",
        "structure": "gpt-oss-120b_1",
    },
    "prod-dev-example": {
        "simple": "gpt-oss-120b_1",
        "structure": "gpt-oss-120b_2",
    },
    "secure-stateful-app": {
        "simple": "gpt-oss-120b_1",
        "structure": "DeepSeek-V4-Flash_1",
    },
    "stateful-app": {
        "simple": "gpt-oss-120b_1",
        "structure": "gpt-oss-120b_2",
    },
    "stateless-app": {
        "simple": "gpt-oss-120b_1",
        "structure": "gpt-oss-120b_2",
    },
}

FILE_SUFFIXES = {".yaml", ".txt", ".stats"}


def copy_variant(source_root: Path, dest_root: Path, variant: str) -> None:
    for model_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for temp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("temp_")):
            for example_dir in sorted(p for p in temp_dir.iterdir() if p.is_dir()):
                example_name = example_dir.name
                if example_name not in SELECTIONS:
                    continue

                selected_stem = SELECTIONS[example_name][variant]
                matching_files = [
                    file_path
                    for file_path in example_dir.iterdir()
                    if file_path.is_file()
                    and file_path.suffix in FILE_SUFFIXES
                    and file_path.stem == selected_stem
                ]

                if not matching_files:
                    print(
                        f"[WARN] missing files for {model_dir.name}/{temp_dir.name}/{example_name} "
                        f"with stem {selected_stem}"
                    )
                    continue

                destination_dir = dest_root / model_dir.name / temp_dir.name / example_name
                destination_dir.mkdir(parents=True, exist_ok=True)

                for source_file in matching_files:
                    shutil.copy2(source_file, destination_dir / source_file.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild test1-simple-results and test1-structure-results from test1-results."
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--simple-root", type=Path, default=SIMPLE_ROOT)
    parser.add_argument("--structure-root", type=Path, default=STRUCTURE_ROOT)
    parser.add_argument("--clean", action="store_true", help="Delete destination roots before copying.")
    args = parser.parse_args()

    if args.clean:
        for root in (args.simple_root, args.structure_root):
            if root.exists():
                shutil.rmtree(root)

    copy_variant(args.source_root, args.simple_root, "simple")
    copy_variant(args.source_root, args.structure_root, "structure")


if __name__ == "__main__":
    main()