import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("init-project")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        print(f"Running command: {args.command}")


if __name__ == "__main__":
    main()
