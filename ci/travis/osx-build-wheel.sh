#!/bin/bash
set -e -x

python --version
pip install delocate
pip install -r ./python/requirements.txt
pip wheel -e ./python -w raw-wheelhouse/

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
