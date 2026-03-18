"""CLI for oracleai."""
import sys, json, argparse
from .core import Oracleai

def main():
    parser = argparse.ArgumentParser(description="The Oracle's Paradox — Detecting and correcting self-fulfilling prophecies in AI prediction systems. Performative prediction research.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Oracleai()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.detect(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"oracleai v0.1.0 — The Oracle's Paradox — Detecting and correcting self-fulfilling prophecies in AI prediction systems. Performative prediction research.")

if __name__ == "__main__":
    main()
