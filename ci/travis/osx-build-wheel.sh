#!/bin/bash
set -e -x

python --version
pip install delocate
pip install cython>=0.27
pip wheel -e ./ -w raw-wheelhouse/

# Bundle external shared libraries into the wheels
for whl in raw-wheelhouse/*.whl; do
    if ! [[ "$whl" =~ minorminer ]]; then
        continue
    fi
    delocate-wheel "$whl" -w wheelhouse/
done

# Install packages and test
pip install minorminer --no-index -f wheelhouse/
pip install -r tests/requirements.txt
python -m nose . --exe
