"""Split `a; b` statements into separate lines (token based, so strings and brackets are safe)."""
import io
import sys
import tokenize
from pathlib import Path

COMPOUND = ("if ", "for ", "while ", "with ", "def ", "class ", "try:", "else:", "elif ", "except", "finally:")


def split_file(path):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    cuts = {}
    depth = 0
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth -= 1
            elif tok.string == ";" and depth == 0:
                cuts.setdefault(tok.start[0], []).append(tok.start[1])
    manual = []
    for row in sorted(cuts, reverse=True):
        line = lines[row - 1]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith(COMPOUND) or stripped.startswith("lambda"):
            manual.append(row)
            continue
        pieces, last = [], 0
        for col in cuts[row]:
            pieces.append(line[last:col])
            last = col + 1
        pieces.append(line[last:])
        comment = ""
        new = []
        for k, piece in enumerate(pieces):
            piece = piece.strip() if k else piece.rstrip()
            if piece.strip():
                new.append((indent + piece.strip() if k else piece) + "\n")
        lines[row - 1:row] = new
    path.write_text("".join(lines))
    return manual


if __name__ == "__main__":
    for root in sys.argv[1:]:
        for path in sorted(Path(root).rglob("*.py")):
            manual = split_file(path)
            if manual:
                print(path, "needs manual splitting at lines", manual)
