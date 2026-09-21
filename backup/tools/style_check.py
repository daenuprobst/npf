"""List comment and docstring lines that break the style rules (ASCII, no colon / semicolon / dash characters, no trailing comments)."""
import ast
import io
import sys
import tokenize
from pathlib import Path

BAD = [":", ";", "—", "–"]


def check(path):
    text = path.read_text()
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            body = tok.string
            trailing = tok.line[:tok.start[1]].strip() != ""
            if not body.isascii() or any(c in body for c in BAD) or trailing:
                out.append((tok.start[0], "trailing comment" if trailing else "comment", body.strip()[:110]))
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc and (not doc.isascii() or any(c in doc for c in BAD)):
                line = node.body[0].lineno
                bad = [l.strip()[:100] for l in doc.splitlines() if (not l.isascii()) or any(c in l for c in BAD)]
                out.append((line, "docstring", " | ".join(bad)[:160]))
    return sorted(out)


if __name__ == "__main__":
    total = 0
    for root in sys.argv[1:]:
        for path in sorted(Path(root).rglob("*.py")):
            issues = check(path)
            total += len(issues)
            for line, kind, body in issues:
                print(f"{path}:{line}: {kind}: {body}")
    print("total", total)
