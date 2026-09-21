"""Move trailing comments to the line before the code and strip the characters the style forbids from comments and docstrings."""
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ASCII = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "×": "x", "•": "*", "σ": "sigma", "ᵀ": "^T",
         "—": ",", "–": "-", "−": "-", "·": "*", "≈": "~", "λ": "lambda", "θ": "theta", "μ": "mu",
         "∈": "in", "≠": "!=", "±": "+-", "π": "pi", "φ": "phi", "ψ": "psi", "Δ": "d", "∅": "empty",
         "⇔": "<=>", "⇒": "=>", "⊙": "*", "∑": "sum", "∞": "inf", "é": "e", "ø": "o", "Ø": "O"}


def clean(text):
    for k, v in ASCII.items():
        text = text.replace(k, v)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s*:\s+", ", ", text)
    text = re.sub(r":$", "", text)
    text = text.replace(":", " ")
    text = re.sub(r"\s*;\s*", ", ", text)
    return re.sub(r" {2,}", " ", text) if not text.startswith(" ") else text


def restyle(path):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    inserts = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type != tokenize.COMMENT or tok.string.startswith("#!"):
            continue
        row, col = tok.start
        line = lines[row - 1]
        body = "# " + clean(tok.string.lstrip("#").strip())
        if line[:col].strip():
            code = line[:col].rstrip()
            indent = code[:len(code) - len(code.lstrip())]
            lines[row - 1] = code + "\n"
            inserts.append((row - 1, indent + body + "\n"))
        else:
            lines[row - 1] = line[:col] + body + "\n"
    for index, new in sorted(inserts, reverse=True):
        lines.insert(index, new)
    text = "".join(lines)
    # docstrings
    tree = ast.parse(text)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False) is not None:
            d = node.body[0]
            spans.append((d.lineno, d.end_lineno))
    lines = text.splitlines(keepends=True)
    for first, last in spans:
        for k in range(first - 1, last):
            raw = lines[k]
            stripped = raw.rstrip("\n")
            lead = stripped[:len(stripped) - len(stripped.lstrip())]
            body = stripped.lstrip()
            quote_start = body.startswith(('"""', 'r"""'))
            quote_end = body.endswith('"""') and len(body) > 3
            core = body
            prefix = suffix = ""
            if quote_start:
                prefix, core = core[:core.index('"""') + 3], core[core.index('"""') + 3:]
            if core.endswith('"""'):
                core, suffix = core[:-3], '"""'
            lines[k] = lead + prefix + clean(core) + suffix + "\n"
    path.write_text("".join(lines))


if __name__ == "__main__":
    for root in sys.argv[1:]:
        for path in sorted(Path(root).rglob("*.py")):
            restyle(path)
