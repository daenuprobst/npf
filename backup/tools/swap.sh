#!/bin/sh
# Swap the staged refactor into place. The old flat package stays available as a tarball and, unpacked, under
# backup/npf_flat/npf for the equivalence tests.
set -e
cd /home/daenu/Code/pgnn

if pgrep -f "python3? -m npf\." > /dev/null; then
    echo "jobs from the old tree are still running, not swapping"
    pgrep -af "python3? -m npf\." | cut -c1-160
    exit 1
fi

stamp=$(date +%F-%H%M)
tar czf backup/npf-flat-$stamp.tar.gz npf pyproject.toml uv.lock README.md THEORY.md .gitignore
rm -rf backup/npf_flat
mkdir -p backup/npf_flat
cp -r npf backup/npf_flat/npf
rm -rf backup/npf_flat/npf/__pycache__
cp pyproject.toml backup/npf_flat/pyproject.toml

# symlinks that were only needed while staging
[ -L refactor/data ] && rm refactor/data
[ -L refactor/results ] && rm refactor/results

rm -r npf
mv refactor/pyproject.toml pyproject.toml
mv refactor/README.md README.md
mv refactor/THEORY.md THEORY.md
mv refactor/.gitignore .gitignore
mv refactor/src src
mv refactor/benchmarks benchmarks
mv refactor/tests tests
find refactor -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rmdir refactor

uv sync
echo "swap done"
