"""Molecular Transformer, Schwaller et al. 2019, trained on nested random subsets of the official USPTO-MIT training
file as a reference for data efficiency, since no compared paper reports such numbers. The published recipe is used,
mixed setting, 4 layers of width 256, 8 heads, one randomised copy of every training source, Noam schedule, beam 5,
with OpenNMT-py in an isolated environment. The subsets are those of benchmarks.chemistry.experiment --subset, and all
40,000 test reactions are in the denominator.

    uv run python -m benchmarks.chemistry.baselines.molecular_transformer prepare 4090
    uv run python -m benchmarks.chemistry.baselines.molecular_transformer train 4090 --steps 3000 --warmup 800
    uv run python -m benchmarks.chemistry.baselines.molecular_transformer score 4090
    uv run python -m benchmarks.chemistry.baselines.molecular_transformer rescore 4090   # the test metrics again, no GPU

Test predictions are scored three ways. exact is the published protocol, the canonical target string. major compares
the largest predicted molecule with the largest recorded one. found is the rule of the token game, the recorded main
product among the predicted molecules and every recorded molecule predicted or among the precursors.
"""

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

from rdkit import Chem, RDLogger

from npf.chem import canonical_product

from ..experiment import subset_lines

RDLogger.DisableLog("rdApp.*")

ONMT = [
    "uv",
    "run",
    "--no-project",
    "--python",
    "3.11",
    "--with",
    "OpenNMT-py==3.5.1",
    "--with",
    "numpy<2",
]
RAW = Path("data/uspto_mit/data")
TOKEN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
N_VAL = 1500


def folder(n, tag=""):
    return Path(f"results/uspto_mit/baselines/mt-sub{n}{tag}")


def tokenise(smiles):
    tokens = TOKEN.findall(smiles)
    assert "".join(tokens) == smiles, smiles

    return " ".join(tokens)


def plain(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol


def randomised(smiles, rng):
    """Random atom order within every molecule and random order of the molecules."""
    parts = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s), doRandom=True, canonical=False)
        for s in smiles.split(".")
    ]
    rng.shuffle(parts)

    return ".".join(parts)


def read(name, lines=None):
    """Canonical source and target without atom maps, None where RDKit cannot read the line."""
    rows = []
    for k, line in enumerate(open(RAW / name)):
        if lines is not None and k not in lines:
            continue

        try:
            precursors, product = line.split()[0].split(">>")
            rows.append(
                (Chem.MolToSmiles(plain(precursors)), Chem.MolToSmiles(plain(product)))
            )
        except Exception:
            rows.append(None)

    return rows


def prepare(n, augment=True):
    out = folder(n)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    train = [
        row
        for row in read("train.txt", set(subset_lines(n).tolist()))
        if row is not None
    ]
    pairs = list(train)

    if augment:
        pairs += [(randomised(src, rng), tgt) for src, tgt in train]

    rng.shuffle(pairs)
    val = [row for row in read("valid.txt", set(range(N_VAL))) if row is not None]
    test = read("test.txt")
    for name, rows in (("train", pairs), ("val", val), ("test", test)):
        # an unreadable test reaction keeps its line and counts as wrong
        (out / f"src-{name}.txt").write_text(
            "\n".join(tokenise(r[0]) if r else "C" for r in rows) + "\n"
        )
        (out / f"tgt-{name}.txt").write_text(
            "\n".join(tokenise(r[1]) if r else "C" for r in rows) + "\n"
        )

    (out / "test-readable.json").write_text(json.dumps([r is not None for r in test]))
    print(
        f"{len(train)} training reactions ({len(pairs)} with the randomised copies), {len(val)} validation, {len(test)} test"
    )


def config(n, steps, dropout, warmup=8000):
    out = folder(n)
    every = max(steps // 12, 500)
    text = f"""save_data: {out}/run
src_vocab: {out}/vocab.txt
tgt_vocab: {out}/vocab.txt
share_vocab: true
overwrite: true
data:
  corpus_1:
    path_src: {out}/src-train.txt
    path_tgt: {out}/tgt-train.txt
  valid:
    path_src: {out}/src-val.txt
    path_tgt: {out}/tgt-val.txt
save_model: {out}/model
save_checkpoint_steps: {every}
keep_checkpoint: 20
seed: 42
train_steps: {steps}
valid_steps: {every}
warmup_steps: {warmup}
report_every: 500
encoder_type: transformer
decoder_type: transformer
position_encoding: true
share_embeddings: true
layers: 4
heads: 8
hidden_size: 256
word_vec_size: 256
transformer_ff: 2048
dropout: [{dropout}]
attention_dropout: [{dropout}]
label_smoothing: 0.0
param_init: 0.0
param_init_glorot: true
batch_size: 4096
batch_type: tokens
normalization: tokens
accum_count: [4]
max_generator_batches: 32
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 2.0
max_grad_norm: 0.0
world_size: 1
gpu_ranks: [0]
"""
    (out / "config.yaml").write_text(text)

    return out / "config.yaml"


def train(n, steps, dropout, warmup=8000):
    cfg = config(n, steps, dropout, warmup)
    subprocess.run(
        ONMT + ["onmt_build_vocab", "-config", str(cfg), "-n_sample", "-1"], check=True
    )
    subprocess.run(ONMT + ["onmt_train", "-config", str(cfg)], check=True)


def translate(model, src, out, n_best):
    subprocess.run(
        ONMT
        + [
            "onmt_translate",
            "-model",
            str(model),
            "-src",
            str(src),
            "-output",
            str(out),
            "-beam_size",
            "5",
            "-n_best",
            str(n_best),
            "-batch_size",
            "64",
            "-max_length",
            "300",
            "-gpu",
            "0",
        ],
        check=True,
    )


def canonical(tokens):
    mol = Chem.MolFromSmiles(tokens.replace(" ", ""))

    return Chem.MolToSmiles(mol) if mol is not None else None


def molecules(tokens):
    """The molecules of a SMILES as a set of canonical SMILES without stereochemistry, None if it does not parse."""
    smiles = tokens.replace(" ", "")

    return (
        None
        if Chem.MolFromSmiles(smiles) is None
        else {canonical_product(f) for f in smiles.split(".")}
    )


def correct(prediction, target, source, rule):
    if rule == "exact":
        return canonical(prediction) is not None and canonical(prediction) == canonical(
            target
        )

    predicted = molecules(prediction)
    if predicted is None:
        return False

    if rule == "major":
        return canonical_product(prediction.replace(" ", "")) == canonical_product(
            target.replace(" ", "")
        )

    recorded = molecules(target)

    return canonical_product(
        target.replace(" ", "")
    ) in predicted and recorded <= predicted | (molecules(source) or set())


def top_k(predictions, targets, n_best, readable=None, sources=None, rule="exact"):
    """Share of reactions with a correct prediction among the first k, for k = 1..n_best, under one of the rules."""
    hits = [0] * n_best
    for row, target in enumerate(targets):
        if readable is not None and not readable[row]:
            continue

        source = sources[row] if sources is not None else ""
        right = [
            correct(p, target, source, rule)
            for p in predictions[row * n_best : (row + 1) * n_best]
        ]
        if True in right:
            for k in range(right.index(True), n_best):
                hits[k] += 1

    return [h / len(targets) for h in hits]


def test_metrics(out, n_best=5):
    """The test predictions in out scored under the three rules, with every test line in the denominator."""
    lines = lambda path: Path(path).read_text().splitlines()
    readable = json.loads((out / "test-readable.json").read_text())
    predictions, targets, sources = (
        lines(out / "pred-test.txt"),
        lines(out / "tgt-test.txt"),
        lines(out / "src-test.txt"),
    )
    metrics = {}

    for rule, prefix in (
        ("exact", "product"),
        ("major", "product_major"),
        ("found", "product_found"),
    ):
        metrics |= {
            f"{prefix}_top{k + 1}": a
            for k, a in enumerate(
                top_k(predictions, targets, n_best, readable, sources, rule)
            )
        }

    return metrics


def rescore(n):
    out = folder(n)
    result = json.loads((out / "result.json").read_text())
    result = {
        k: v for k, v in result.items() if not k.startswith("product_")
    } | test_metrics(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(
        "  ".join(
            f"{k} {v:.4f}"
            for k, v in result.items()
            if k.startswith("product") and k.endswith("top1")
        )
    )


def score(n):
    out = folder(n)
    lines = lambda path: Path(path).read_text().splitlines()
    checkpoints = sorted(
        out.glob("model_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1])
    )
    averaged = out / "model_average.pt"
    subprocess.run(
        ONMT
        + [
            "onmt_average_models",
            "-models",
            *map(str, checkpoints[-5:]),
            "-output",
            str(averaged),
        ],
        check=True,
    )
    val = {}
    for model in checkpoints[-6:] + [averaged]:
        translate(model, out / "src-val.txt", out / "pred-val.txt", 1)
        val[model.name] = top_k(
            lines(out / "pred-val.txt"), lines(out / "tgt-val.txt"), 1
        )[0]
        print(f"{model.name}: validation top-1 {val[model.name]:.4f}", flush=True)

    best = max(val, key=val.get)
    translate(out / best, out / "src-test.txt", out / "pred-test.txt", 5)
    readable = json.loads((out / "test-readable.json").read_text())
    result = {
        "model": "molecular-transformer",
        "subset_lines": n,
        "validation_top1": val,
        "selected": best,
        "n_test": len(readable),
        **test_metrics(out),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(
        "  ".join(
            f"{k} {v:.4f}"
            for k, v in result.items()
            if k.startswith("product") and k.endswith("top1")
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "train", "score", "rescore"])
    ap.add_argument("n", type=int, help="number of random lines of the training file")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--warmup",
        type=int,
        default=8000,
        help="warmup steps of the Noam schedule, the published 8,000, scaled down with --steps for a short run",
    )
    args = ap.parse_args()
    if args.step == "prepare":
        prepare(args.n)
    elif args.step == "train":
        train(args.n, args.steps, args.dropout, args.warmup)
    elif args.step == "score":
        score(args.n)
    else:
        rescore(args.n)


if __name__ == "__main__":
    main()
