"""Build course notebooks from plain-Python sources.

Sources live in tools/src/<module_dir>/:
  lesson.py     -> modules/<module_dir>/lesson.ipynb
  exercises.py  -> modules/<module_dir>/solutions.ipynb AND modules/<module_dir>/exercises.ipynb

Source format (jupytext "percent" style):
  # %% [markdown]      starts a markdown cell; following lines are "# "-prefixed markdown
  # %%                 starts a code cell
  #! SETUP             (alone in a code cell) expands to the standard setup cell

Inside exercises.py, solution code is wrapped like this:
      ### BEGIN SOLUTION
      #> # TODO: describe what to do        <- '#> ' lines become the exercise stub
      #> raise NotImplementedError
      real solution code                    <- only kept in solutions.ipynb
      ### END SOLUTION

Usage:
  python tools/build.py 00 01            # build modules whose dir starts with 00 / 01
  python tools/build.py all --execute    # also execute lesson + solutions with FAST_DEV_RUN=1
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from papers import reading_cell_markdown  # tools/papers.py: recommended reading per module

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "src"
OUT = ROOT / "modules"

SETUP_CELL = '''\
# Standard setup: make the repo root importable, pick a device, fix the random seed.
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parents[1]  # notebooks live in modules/XX_name/, so go up two levels
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.device import get_device, seed_everything, fast_dev_run

FAST_DEV_RUN = fast_dev_run()  # True only for automated checks; leave it False when learning
DATA_DIR = ROOT / "data"
device = get_device()
seed_everything(42)'''


def parse_cells(text: str):
    """Split a percent-format source into [(kind, source)]."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            # strip the leading "# " / "#" from each markdown line
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def transform_code(code: str, mode: str) -> str:
    """mode='solution' keeps real code; mode='exercise' keeps only '#>' stubs."""
    if code.strip() == "#! SETUP":
        return SETUP_CELL
    out, in_sol = [], False
    for line in code.splitlines():
        s = line.strip()
        if s == "### BEGIN SOLUTION":
            in_sol = True; continue
        if s == "### END SOLUTION":
            in_sol = False; continue
        if in_sol:
            m = re.match(r"^(\s*)#> ?(.*)$", line)
            if mode == "exercise" and m:
                out.append(m.group(1) + m.group(2))
            elif mode == "solution" and not m:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def make_notebook(cells, mode: str):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, body in cells:
        if kind == "markdown":
            nb.cells.append(new_markdown_cell(body))
        else:
            nb.cells.append(new_code_cell(transform_code(body, mode)))
    return nb


def execute(path: Path):
    env = dict(os.environ, FAST_DEV_RUN="1", MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=1800", str(path)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit(f"FAILED: {path.relative_to(ROOT)}")
    print(f"  executed OK: {path.relative_to(ROOT)}")


def build(module_dir: Path, run: bool):
    name = module_dir.name
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}]")
    lesson_src = module_dir / "lesson.py"
    ex_src = module_dir / "exercises.py"
    targets = []
    if lesson_src.exists():
        cells = parse_cells(lesson_src.read_text())
        reading = reading_cell_markdown(name[:2])
        if reading:
            cells.append(("markdown", reading))   # appended at the end of every lesson
        nbformat.write(make_notebook(cells, "solution"), dest / "lesson.ipynb")
        targets.append(dest / "lesson.ipynb")
    if ex_src.exists():
        cells = parse_cells(ex_src.read_text())
        nbformat.write(make_notebook(cells, "solution"), dest / "solutions.ipynb")
        nbformat.write(make_notebook(cells, "exercise"), dest / "exercises.ipynb")  # never executed: clean outputs
        targets.append(dest / "solutions.ipynb")
    if run:
        for t in targets:
            execute(t)


def main(argv):
    run = "--execute" in argv
    keys = [a for a in argv if not a.startswith("--")]
    dirs = sorted(d for d in SRC.iterdir() if d.is_dir())
    if keys and keys != ["all"]:
        dirs = [d for d in dirs if any(d.name.startswith(k) for k in keys)]
    if not dirs:
        raise SystemExit("no matching module sources")
    for d in dirs:
        build(d, run)


if __name__ == "__main__":
    main(sys.argv[1:])
