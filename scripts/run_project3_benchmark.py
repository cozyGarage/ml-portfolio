import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "projects" / "03-model-benchmark" / "src"
RESULTS_DIR = ROOT / "projects" / "03-model-benchmark" / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wine", "breast_cancer", "iris"], default="wine")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"benchmark_{args.dataset}.json"

    cmd = [
        sys.executable,
        str(PROJECT_SRC / "benchmark_compare.py"),
        "--dataset",
        args.dataset,
        "--output-json",
        str(output_json),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print("\nProject 3 benchmark completed.")


if __name__ == "__main__":
    main()
