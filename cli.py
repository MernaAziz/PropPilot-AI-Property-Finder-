"""Command-line version.

    python cli.py "2-bedroom near the university, under 800, with a balcony"
    python cli.py -v "quiet place near the metro"      # -v shows every step
"""

import argparse

from property_finder import create_finder
from property_finder.config import CURRENCY


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Property Finder")
    parser.add_argument("request", nargs="*", help="what you are looking for")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the output of each step")
    args = parser.parse_args()

    request = " ".join(args.request).strip() or input("What are you looking for? ")

    print("Loading models and building the index...")
    finder, _ = create_finder()
    res = finder.invoke({"request": request})

    if args.verbose:
        print("\n=== Step 1: parsed filters ===")
        print(res.filters.model_dump_json(indent=2))
        print("\n=== Step 2: districts ===")
        print(res.area_note)
        print("\n=== Step 3: retrieved candidates ===")
        print(", ".join(res.candidate_ids) or "none")
        for note in res.relax_notes:
            print(f"  ({note})")

    print(f"\n{res.summary}\n")
    for i, r in enumerate(res.results, 1):
        print(f"{i}. {r.title}  [{r.listing_id}]")
        print(f"   {r.district} | {CURRENCY}{r.price}/month | score {r.match_score}/100")
        if r.issues:
            print(f"   ⚠ {' · '.join(r.issues)}")
        print(f"   + {' | '.join(r.pros)}")
        print(f"   - {' | '.join(r.cons)}")
        print(f"   {r.reason}\n")


if __name__ == "__main__":
    main()
