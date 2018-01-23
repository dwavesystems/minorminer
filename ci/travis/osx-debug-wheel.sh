#!/bin/bash
set -e -x

# Move to a virtualenv to isolate the debug install later
pip install virtualenv
python -m virtualenv debug_env
source debug_env/bin/activate

set CPPDEBUG=1
python --version
pip install delocate
pip install cython==0.27
pip wheel -e ./ -w raw-debug-wheelhouse/

# Bundle external shared libraries into the wheels
for whl in raw-debug-wheelhouse/*.whl; do
    if ! [[ "$whl" =~ minorminer ]]; then
        continue
    fi
    delocate-wheel "$whl" -w debug-wheelhouse/
done

# Install packages and test
pip install minorminer --no-index -f debug-wheelhouse/
pip install -r tests/requirements.txt
python -m nose . --exe
