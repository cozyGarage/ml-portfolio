import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()

    print("Starting training pipeline")
    if args.config:
        print(f"Using config: {args.config}")


if __name__ == "__main__":
    main()
