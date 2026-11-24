"""Insert (or refresh) the recommended-reading bullets under each module in README.md.

Run after editing tools/papers.py:  python tools/update_readme.py
"""
import re
from pathlib import Path

from papers import readme_lines

README = Path(__file__).resolve().parents[1] / "README.md"


def main():
    lines = README.read_text().splitlines()
    # 1. drop any existing reading blocks so the script can be re-run safely
    cleaned, skipping = [], False
    for ln in lines:
        if ln.startswith("  - 📄") or ln.startswith("  - 🎥"):
            skipping = True
            continue
        if skipping and ln.startswith("    - "):
            continue
        skipping = False
        cleaned.append(ln)
    # 2. insert after each module's subtopic list
    out, current = [], None
    for ln in cleaned:
        m = re.match(r"^- \[[ x]\] \*\*(\d\d) ", ln)
        block_ended = current and not ln.startswith("  ")
        if block_ended or m:
            if current:
                out.extend(readme_lines(current))
            current = m.group(1) if m else None
        out.append(ln)
    if current:
        out.extend(readme_lines(current))
    # keep the blank line before headings that follow a module block
    README.write_text("\n".join(out) + "\n")
    print("README reading lists updated")


if __name__ == "__main__":
    main()
