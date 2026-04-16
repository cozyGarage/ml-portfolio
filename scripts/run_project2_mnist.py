import argparse
import json
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


def load_best_model(results_json: Path) -> str:
    if not results_json.exists():
        return "svc_rbf"
    try:
        data = json.loads(results_json.read_text(encoding="utf-8"))
        return data.get("best_model", "svc_rbf")
    except Exception:
        return "svc_rbf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["classical", "pytorch", "all"], default="classical")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data/mnist")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_json = RESULTS_DIR / "classical_results.json"

    if args.stage in {"classical", "all"}:
        run([
            sys.executable,
            str(PROJECT_SRC / "baseline_digits_sklearn.py"),
            "--output-json",
            str(results_json),
        ])
        best_model = load_best_model(results_json)
        print(f"Using best classical model for confusion analysis: {best_model}")
        run([
            sys.executable,
            str(PROJECT_SRC / "confusion_analysis.py"),
            "--model",
            best_model,
            "--output-dir",
            str(RESULTS_DIR),
        ])

    if args.stage in {"pytorch", "all"}:
        artifact_dir = args.artifact_dir or str(RESULTS_DIR / "pytorch")
        cmd = [
            sys.executable,
            str(PROJECT_SRC / "pytorch_mnist_train.py"),
            "--epochs",
            str(args.epochs),
            "--model",
            args.model,
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--data-dir",
            args.data_dir,
            "--patience",
            str(args.patience),
            "--artifact-dir",
            artifact_dir,
        ]
        run(cmd)

    print("\nProject 2 pipeline completed.")


if __name__ == "__main__":
    main()
