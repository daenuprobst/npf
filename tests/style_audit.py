"""Comment style audit. Not a test, run it as a script.

    python tests/style_audit.py [paths] [--list] [--max-line 160] [--uncommented 8]

Rules for comments and docstrings. ASCII only, no colon, no semicolon, no em dash or en dash, and comments stand on
the line before the code instead of trailing it. Also reported are statements joined by a semicolon and long lines.
Comments are read with tokenize and docstrings with ast, so string literals and code never count as comments.
"""
import argparse
import ast
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

RULES = ("non-ascii", "colon", "semicolon", "dash", "trailing", "stmt;stmt", "long-line")

# tool directives are not prose
DIRECTIVE = re.compile(r"#\s*(noqa|type:|pragma|fmt:|pylint:|!)")

# em dash and en dash, written as code points to keep this file ASCII
DASHES = (chr(0x2014), chr(0x2013))


def text_problems(text):
    found = []

    if not text.isascii():
        found.append("non-ascii")

    if ":" in text:
        found.append("colon")

    if ";" in text:
        found.append("semicolon")

    if any(d in text for d in DASHES):
        found.append("dash")

    return found


def docstrings(tree):
    """Line number and text of every docstring line."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue

        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            for offset, line in enumerate(first.value.value.split("\n")):
                yield first.lineno + offset, line


def audit(path, max_line):
    source = path.read_text()
    out = []
    code_on_line = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue

        if tok.type == tokenize.COMMENT:
            if DIRECTIVE.match(tok.string):
                continue

            out += [(tok.start[0], rule, "comment", tok.string) for rule in text_problems(tok.string)]

            if tok.start[0] in code_on_line:
                out.append((tok.start[0], "trailing", "comment", tok.string))

            continue

        for line in range(tok.start[0], tok.end[0] + 1):
            code_on_line.add(line)

        if tok.type == tokenize.OP and tok.string == ";":
            out.append((tok.start[0], "stmt;stmt", "code", tok.line.strip()))

    for number, line in docstrings(ast.parse(source)):
        out += [(number, rule, "docstring", line.strip()) for rule in text_problems(line)]

    for number, line in enumerate(source.split("\n"), 1):
        if len(line) > max_line:
            out.append((number, "long-line", "code", f"{len(line)} characters"))

    return sorted(set(out))


def uncommented(path, min_lines):
    """Functions of at least min_lines lines that carry neither a docstring nor a single comment."""
    source = path.read_text()
    comment_lines = {tok.start[0] for tok in tokenize.generate_tokens(io.StringIO(source).readline) if tok.type == tokenize.COMMENT}
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno - node.lineno + 1 >= min_lines:
            # a comment directly above the def counts as well
            lines = range(node.lineno - 1, node.end_lineno + 1)
            if ast.get_docstring(node) is None and not any(n in comment_lines for n in lines):
                found.append((node.lineno, node.name, node.end_lineno - node.lineno + 1))

    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[str(Path(__file__).resolve().parents[1])])
    ap.add_argument("--list", action="store_true", help="print every violation, not only the counts")
    ap.add_argument("--max-line", type=int, default=160)
    ap.add_argument("--uncommented", type=int, default=0, metavar="N", help="also list functions of N or more lines without any comment or docstring")
    args = ap.parse_args()
    files = []

    for p in map(Path, args.paths):
        files += sorted(q for q in p.rglob("*.py") if "__pycache__" not in q.parts and ".venv" not in q.parts) if p.is_dir() else [p]

    base = Path(args.paths[0]).resolve() if Path(args.paths[0]).is_dir() else Path.cwd()
    total, rows = Counter(), []
    for path in files:
        found = audit(path, args.max_line)
        counts = Counter(rule for _, rule, _, _ in found)
        where = Counter((rule, kind) for _, rule, kind, _ in found)
        total += counts
        name = str(path.resolve().relative_to(base)) if path.resolve().is_relative_to(base) else str(path)
        rows.append((name, counts, where, found))

    width = max(len(name) for name, *_ in rows)
    print(f"{'file':<{width}}  " + "  ".join(f"{r:>9}" for r in RULES) + "      total")

    for name, counts, _, _ in sorted(rows, key=lambda r: -sum(r[1].values())):
        print(f"{name:<{width}}  " + "  ".join(f"{counts[r]:>9}" for r in RULES) + f"  {sum(counts.values()):>9}")

    print(f"{'TOTAL':<{width}}  " + "  ".join(f"{total[r]:>9}" for r in RULES) + f"  {sum(total.values()):>9}")
    in_doc = sum(n for _, _, where, _ in rows for (rule, kind), n in where.items() if kind == "docstring")
    in_comment = sum(n for _, _, where, _ in rows for (rule, kind), n in where.items() if kind == "comment")
    print(f"\n{len(files)} files, {in_comment} findings in comments, {in_doc} in docstrings, {total['stmt;stmt'] + total['long-line']} in code")

    if args.list:
        for name, _, _, found in rows:
            for number, rule, kind, text in found:
                print(f"{name}:{number}: {rule} ({kind}) {text[:140]}")

    if args.uncommented:
        print(f"\nfunctions of {args.uncommented} or more lines without a docstring or comment")

        for path in files:
            for number, name, size in uncommented(path, args.uncommented):
                shown = str(path.resolve().relative_to(base)) if path.resolve().is_relative_to(base) else str(path)
                print(f"{shown}:{number}: {name} ({size} lines)")

    return 1 if sum(total.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
