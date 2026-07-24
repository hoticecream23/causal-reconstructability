"""The scatter plot, and the kill criteria evaluated against it."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

from rnar import analysis  # noqa: E402
from rnar.config import FIGURES, PRESETS, get_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="debug", choices=sorted(PRESETS))
    args = p.parse_args()

    cfg = get_config(args.preset)
    rows = analysis.load_results(cfg.run_dir / "rescue.jsonl")

    out = analysis.scatter(rows, FIGURES / f"{cfg.name}_reconstructability.png",
                           title=f"{cfg.model_name} · {cfg.sae_id}")
    print(f"[plot] saved {out}")

    verdict = analysis.check_kill_criteria(rows)
    with open(cfg.run_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print("\n[verdict]")
    for k, v in verdict.items():
        flag = ""
        if k.startswith("KILL_"):
            flag = "  <-- TRIGGERED" if v else ""
        print(f"  {k:<32} {v}{flag}")

    if any(v for k, v in verdict.items() if k.startswith("KILL_")):
        print("\nAt least one kill criterion fired. Read docs/DESIGN.md before continuing.")


if __name__ == "__main__":
    main()
