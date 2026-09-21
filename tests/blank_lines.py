"""Blank line style. Not a test, run it as a script.

    python tests/blank_lines.py [paths]            # report the lines that lack a blank line before them
    python tests/blank_lines.py [paths] --write    # insert them

Three rules. A comment has a blank line before it unless it opens a block or continues a comment. A loop, a branch or
another compound statement has a blank line before and after it when it starts or ends a step, which is taken to be
always, except that a statement stays attached to the assignment right before it when it tests or fills what that
assignment made. The return at the end of a function has a blank line before it, a guard clause or an early return
has none, and neither has a return that is the only code of its function. The docstring of a function belongs to
its head, so the code or comment right after it is at the top of the function and gets no blank line. Nothing but
blank lines ever changes, and the syntax tree of every file is compared before and after.
"""
import argparse
import ast
import io
import sys
import tokenize
from pathlib import Path

COMPOUND = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try, ast.Match)
NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def physical_lines(source):
    """Lines that hold only a comment, and lines that continue a statement or a string from the line above."""
    comments, continued, start = set(), set(), None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NEWLINE:
            start = None

        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue

        if tok.type == tokenize.COMMENT and start is None:
            comments.add(tok.start[0])
            continue

        if start is None:
            start = tok.start[0]

        continued.update(range(start + 1, tok.end[0] + 1))

    return comments, continued


def names(node, store):
    kinds = (ast.Store,) if store else (ast.Load, ast.Del)

    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, kinds)} if node is not None else set()


def attached(prev, stmt):
    """A compound statement that tests or fills what the assignment right before it made is part of the same step."""
    if not isinstance(prev, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return False

    made = names(prev, store=True)

    if isinstance(stmt, (ast.If, ast.While)):
        return bool(made & names(stmt.test, store=False))

    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return bool(made & (names(stmt.iter, store=False) | {n for s in stmt.body for n in names(s, store=False)}))

    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return bool(made & {n for item in stmt.items for n in names(item.context_expr, store=False)})

    return False


def blocks(tree):
    """Every list of statements together with the node that owns it."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield node, field, stmts

        for handler in getattr(node, "handlers", []):
            yield handler, "body", handler.body

        for case in getattr(node, "cases", []):
            yield case, "body", case.body


def is_docstring(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def only_return(owner):
    """The return of a function that holds nothing else but a docstring."""
    code = [s for s in owner.body if not is_docstring(s)] if isinstance(owner, FUNCTIONS) else []
    if len(code) == 1 and isinstance(code[0], ast.Return):
        return code[0]


def unwanted(source):
    """Numbers of the blank lines between the docstring of a function and the first line after it."""
    lines, found = source.splitlines(), set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, FUNCTIONS) or not is_docstring(node.body[0]):
            continue

        n = node.body[0].end_lineno + 1
        while n <= len(lines) and not lines[n - 1].strip():
            found.add(n)
            n += 1

        # the docstring was the whole body, the blank lines after it separate two functions
        if len(node.body) == 1:
            found -= set(range(node.body[0].end_lineno + 1, n))

    return found


def wanted(source):
    """Numbers of the lines that should have a blank line before them and do not."""
    lines = source.splitlines()
    comments, continued = physical_lines(source)
    indent = lambda n: len(lines[n - 1]) - len(lines[n - 1].lstrip())
    blank = lambda n: n < 1 or not lines[n - 1].strip()
    found = set()
    tree = ast.parse(source)
    heads = {id(node.body[0]) for node in ast.walk(tree) if isinstance(node, FUNCTIONS) and is_docstring(node.body[0])}
    docstring_ends = {node.body[0].end_lineno for node in ast.walk(tree) if isinstance(node, FUNCTIONS) and is_docstring(node.body[0])}

    # a comment, unless it opens a block, which shows as a deeper indent than the statement that ends with a colon
    for n in sorted(comments):
        above = n - 1
        if blank(above) or above in comments or above in docstring_ends:
            continue

        opener = above
        while opener in continued:
            opener -= 1

        if lines[above - 1].rstrip().endswith(":") and indent(n) > indent(opener):
            continue

        found.add(n)

    inside_function = {id(child) for node in ast.walk(tree) if isinstance(node, FUNCTIONS) for child in ast.walk(node) if child is not node}
    for owner, field, stmts in blocks(tree):
        for prev, stmt in zip(stmts, stmts[1:]):
            if id(prev) in heads:
                continue

            compound = lambda s: isinstance(s, COMPOUND) or isinstance(s, NESTED) and id(s) in inside_function
            last_return = isinstance(stmt, ast.Return) and isinstance(owner, FUNCTIONS) and field == "body" and only_return(owner) is None
            if not (compound(stmt) and not attached(prev, stmt) or compound(prev) or last_return):
                continue

            # the blank line goes above the comment that belongs to the statement
            lead = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", [])])
            while lead - 1 in comments and indent(lead - 1) == indent(lead):
                lead -= 1

            if not blank(lead - 1):
                found.add(lead)

    return found


def rewrite(source):
    lines, found, dropped = source.splitlines(keepends=True), wanted(source), unwanted(source)
    out = "".join(("\n" if n in found else "") + line for n, line in enumerate(lines, 1) if n not in dropped)
    found = found | dropped

    # nothing but blank lines may change
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(source))
    assert [x for x in out.splitlines() if x.strip()] == [x for x in source.splitlines() if x.strip()]

    return out, found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["src", "benchmarks", "tests"])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    files = sorted(f for p in args.paths for f in ([Path(p)] if Path(p).is_file() else Path(p).rglob("*.py")))
    total = 0

    for file in files:
        source = file.read_text()
        out, found = rewrite(source)
        total += len(found)

        if found:
            print(f"{file}: {len(found)} blank lines {'changed' if args.write else 'to change'}")

        if args.write and found:
            file.write_text(out)

    print(f"{total} in {len(files)} files")
    sys.exit(0 if args.write or not total else 1)
