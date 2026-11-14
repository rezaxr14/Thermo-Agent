#!/usr/bin/env python3
"""Build the paper and check it.

    python tools/verify_paper.py [--no-build]

Fails (exit 1) on: LaTeX errors, undefined references or citations, overfull boxes above 1 pt, a cited key missing from refs.bib,
a \\val{} key missing from assets_summary.json, either of the two forbidden words anywhere in the tree, raster images, a referenced figure
missing as SVG or PDF. Warns on bib entries that are never cited. Prints the inventory of \\todo markers, placeholder figures and
bib entries marked TODO-verify.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEXT_EXT = {".tex", ".bib", ".py", ".json", ".md", ".svg", ".txt", ".sty", ".cls", ".bst"}
RASTER = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIRS = {"__pycache__", ".git"}
errors: list[str] = []
warnings: list[str] = []


def files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
            yield p


def build():
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=ROOT, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        errors.append("latexmk failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-25:]))


def check_log():
    log = (ROOT / "main.log").read_text(errors="replace") if (ROOT / "main.log").exists() else ""
    for m in re.finditer(r"^! .*$", log, re.M):
        errors.append("LaTeX error: " + m.group(0))
    for m in re.finditer(r"LaTeX Warning: (Reference|Citation) `([^']*)' on page \d+ undefined", log):
        errors.append(f"undefined {m.group(1).lower()}: {m.group(2)}")
    if "There were undefined references" in log or "There were undefined citations" in log:
        errors.append("log reports undefined references or citations")
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)([^\n]*)", log):
        if float(m.group(1)) > 1.0:
            errors.append(f"overfull hbox {m.group(1)} pt {m.group(2).strip()}")
    for m in re.finditer(r"Overfull \\vbox \(([\d.]+)pt too high\)", log):
        if float(m.group(1)) > 1.0:
            warnings.append(f"overfull vbox {m.group(1)} pt")
    if "Label(s) may have changed" in log:
        warnings.append("labels may have changed: run the build once more")
    if "undefined" in log.lower() and not errors:
        for l in log.splitlines():
            if "undefined" in l.lower() and "Warning" in l:
                errors.append(l.strip())


def check_bib():
    bib = (ROOT / "refs.bib").read_text()
    keys, tv, pending = {}, [], False
    for line in bib.splitlines():
        if line.startswith("% TODO-verify"):
            pending = True
        elif line.startswith("@"):
            m = re.match(r"@\w+\{([^,\s]+),", line)
            if m:
                keys[m.group(1)] = line
                if pending:
                    tv.append(m.group(1))
            pending = False
        elif not line.strip().startswith("%"):
            pending = False
    cited = set()
    for p in ROOT.glob("sections/*.tex"):
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", p.read_text()):
            cited.update(k.strip() for k in m.group(1).split(","))
    main = (ROOT / "main.tex").read_text()
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", main):
        cited.update(k.strip() for k in m.group(1).split(","))
    for k in sorted(cited - set(keys)):
        errors.append(f"cited key missing from refs.bib: {k}")
    for k in sorted(set(keys) - cited):
        warnings.append(f"bib entry never cited: {k}")
    return tv, len(keys), len(cited & set(keys))


def check_vals():
    summ = json.loads((ROOT / "assets_summary.json").read_text())["values"]
    used = collections.Counter()
    for p in list(ROOT.glob("sections/*.tex")) + [ROOT / "main.tex"]:
        for m in re.finditer(r"\\val\{([^}]*)\}", p.read_text()):
            used[m.group(1)] += 1
    for k in sorted(used):
        if k not in summ:
            errors.append(f"\\val key missing in assets_summary.json: {k}")
    vt = (ROOT / "generated" / "values.tex").read_text() if (ROOT / "generated" / "values.tex").exists() else ""
    for k in sorted(used):
        if f"val@{k}\\" not in vt:
            errors.append(f"\\val key missing in generated/values.tex: {k}")
    for k, v in summ.items():
        if not v.get("source"):
            errors.append(f"value without a source: {k}")
    return len(used), sum(used.values())


def check_words():
    pat = re.compile(r"\b(" + "the" + "sis|E" + "ES)\b", re.I)
    for p in files():
        if p.suffix.lower() in TEXT_EXT or p.name in ("Makefile",):
            for i, l in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if pat.search(l):
                    errors.append(f"forbidden word in {p.relative_to(ROOT)}:{i}")
        if pat.search(p.name):
            errors.append(f"forbidden word in file name {p.name}")
    pdf = ROOT / "main.pdf"
    if pdf.exists():
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, errors="replace").stdout
        if pat.search(txt):
            errors.append("forbidden word in main.pdf text")
    for p in (ROOT / "figures" / "pdf").glob("*.pdf"):
        txt = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True, errors="replace").stdout
        if pat.search(txt):
            errors.append(f"forbidden word in {p.name}")


def check_images_and_figures():
    for p in files():
        if p.suffix.lower() in RASTER:
            errors.append(f"raster image: {p.relative_to(ROOT)}")
    for p in (ROOT / "figures" / "svg").glob("*.svg"):
        if "<image" in p.read_text(errors="replace"):
            errors.append(f"raster image embedded in {p.name}")
    for p in list((ROOT / "figures" / "pdf").glob("*.pdf")) + ([ROOT / "main.pdf"] if (ROOT / "main.pdf").exists() else []):
        out = subprocess.run(["pdfimages", "-list", str(p)], capture_output=True, text=True, errors="replace").stdout.strip().splitlines()
        if len(out) > 2:
            errors.append(f"raster image in {p.relative_to(ROOT)}")
    used = set()
    for p in ROOT.glob("sections/*.tex"):
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", p.read_text()):
            used.add(m.group(1))
    for u in sorted(used):
        pdf = ROOT / u
        if not pdf.exists():
            errors.append(f"figure missing: {u}")
            continue
        svg = ROOT / "figures" / "svg" / (pdf.stem + ".svg")
        if not svg.exists():
            errors.append(f"SVG missing for {u}")
    man = ROOT / "figures" / "figure_manifest.json"
    names = set()
    if man.exists():
        names = {f["figure"] for f in json.loads(man.read_text())["figures"]}
        for n in sorted(names):
            if f"figures/pdf/{n}.pdf" not in used:
                errors.append(f"manifest figure not included in the paper: {n}")
    labels = set()
    for p in ROOT.glob("sections/*.tex"):
        labels.update(re.findall(r"\\label\{(fig:[^}]*)\}", p.read_text()))
    for n in sorted(names):
        if f"fig:{n}" not in labels:
            errors.append(f"figure label missing: fig:{n}")
    placeholders = []
    for p in sorted((ROOT / "figures" / "pdf").glob("*.pdf")):
        info = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, errors="replace").stdout.lower()
        if "placeholder" in info:
            placeholders.append(p.name)
    return placeholders, len(used)


def todo_inventory():
    out = collections.defaultdict(list)
    for p in sorted(ROOT.glob("sections/*.tex")) + sorted((ROOT / "generated").glob("*.tex")) + [ROOT / "main.tex"]:
        txt = p.read_text()
        for m in re.finditer(r"\\todo\{((?:[^{}]|\{[^{}]*\})*)\}", txt):
            out[m.group(1)].append(p.name)
    return out


def word_counts():
    res = {}
    for p in sorted(ROOT.glob("sections/*.tex")):
        t = p.read_text()
        t = re.sub(r"%.*", "", t)
        t = re.sub(r"\\begin\{(table|figure|longtable)\}.*?\\end\{\1\}", " ", t, flags=re.S)
        t = re.sub(r"\\(cite[a-z]*|cref|Cref|ref|label|val)\{[^}]*\}", " X ", t)
        t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
        t = re.sub(r"[{}$\\]", " ", t)
        res[p.name] = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", t))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-build", action="store_true"); a = ap.parse_args()
    if not a.no_build:
        build()
    check_log()
    tv, nbib, ncited = check_bib()
    nkeys, nuses = check_vals()
    check_words()
    placeholders, nfig = check_images_and_figures()
    pages = subprocess.run(["pdfinfo", str(ROOT / "main.pdf")], capture_output=True, text=True, errors="replace").stdout if (ROOT / "main.pdf").exists() else ""
    m = re.search(r"Pages:\s+(\d+)", pages)
    todos = todo_inventory()
    print(f"pages: {m.group(1) if m else 'n/a'}; bib entries: {nbib} ({ncited} cited); \\val keys used: {nkeys} ({nuses} uses); figures included: {nfig}")
    print("words per section: " + ", ".join(f"{k}: {v}" for k, v in word_counts().items()))
    print(f"\\todo markers: {sum(len(v) for v in todos.values())} in {len(todos)} distinct texts")
    for k, v in sorted(todos.items()):
        print(f"  [{len(v)}] {k}  ({', '.join(sorted(set(v)))})")
    print("placeholder figures: " + (", ".join(placeholders) if placeholders else "none"))
    print("bib entries marked TODO-verify: " + (", ".join(tv) if tv else "none"))
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("FAILED" if errors else "ok")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
