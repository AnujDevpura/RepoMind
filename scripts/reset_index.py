import argparse
import sys

import chromadb

from src.config import CHROMA_PATH


COLLECTION_NAME = "repomind_codebase"


def reset_index(confirm: bool = False):
    if not confirm:
        raise SystemExit("Refusing to reset index without --yes")

    db = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        db.delete_collection(COLLECTION_NAME)
        print(f"Deleted Chroma collection: {COLLECTION_NAME}")
    except Exception as error:
        message = str(error).lower()
        if "does not exist" in message or "not found" in message:
            print(f"Collection did not exist: {COLLECTION_NAME}")
        else:
            raise

    db.get_or_create_collection(COLLECTION_NAME)
    print(f"Created empty Chroma collection: {COLLECTION_NAME}")


def main():
    parser = argparse.ArgumentParser(description="Reset the RepoMind Chroma index.")
    parser.add_argument("--yes", action="store_true", help="Confirm index deletion.")
    args = parser.parse_args()

    try:
        reset_index(confirm=args.yes)
    except Exception as error:
        print(f"Failed to reset index: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
