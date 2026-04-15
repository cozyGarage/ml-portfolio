import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "projects" / "02-mnist-classifier" / "src"
RESULTS_DIR = ROOT / "projects" / "02-mnist-classifier" / "results"


def run(cmd):
    print("\n=== Running:", " ".join(cmd), "===")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["classical", "pytorch", "all"],
        default="classical",
        help="Which part of Project 2 to run.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="Artifact directory for PyTorch outputs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs for PyTorch training.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in {"classical", "all"}:
        run([
            sys.executable,
            str(PROJECT_SRC / "baseline_digits_sklearn.py"),
            "--output-json",
            str(RESULTS_DIR / "classical_results.json"),
        ])
        run([
            sys.executable,
            str(PROJECT_SRC / "confusion_analysis.py"),
            "--model",
            "svc_rbf",
            "--output-dir",
            str(RESULTS_DIR),
        ])

    if args.stage in {"pytorch", "all"}:
        cmd = [
            sys.executable,
            str(PROJECT_SRC / "pytorch_mnist_train.py"),
            "--epochs",
            str(args.epochs),
            "--model",
            "cnn",
        ]
        if args.artifact_dir:
            cmd.extend(["--artifact-dir", args.artifact_dir])
        run(cmd)

    print("\nProject 2 pipeline completed.")


if __name__ == "__main__":
    main()
