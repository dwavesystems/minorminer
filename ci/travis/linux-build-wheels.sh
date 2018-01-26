#!/bin/bash
set -e -x

for PYBIN in /opt/python/*/bin; do
    if [[ ${PYBIN} =~ "2.6" ]]; then
        # bdist_wheel has an error under the version of 2.6 available
        continue
    fi
    "${PYBIN}/pip" install cython==0.27
    "${PYBIN}/pip" wheel -e /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    if ! [[ "$whl" =~ minorminer ]]; then
        continue
    fi
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ ${PYBIN} =~ "2.6" ]]; then
        # bdist_wheel has an error under the version of 2.6 available
        continue
    fi
    "${PYBIN}/pip" install -r /io/tests/requirements.txt
    "${PYBIN}/pip" install minorminer --no-index -f /io/wheelhouse/
    (cd /io/; "${PYBIN}/python" -m nose . --exe)
done
