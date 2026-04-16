import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "01-housing-price-predictor" / "src"


def run_step(script_name: str) -> int:
    script_path = PROJECT / script_name
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    return result.returncode


def main():
    steps = [
        "baseline_train.py",
        "crossval_compare.py",
        "tune_random_forest.py",
        "feature_importance.py",
    ]

    for step in steps:
        code = run_step(step)
        if code != 0:
            print(f"Stopped because {step} failed with exit code {code}.")
            sys.exit(code)

    print("\nProject 1 pipeline completed.")
    print("Next: update the notes templates and final summary.")


if __name__ == "__main__":
    main()
