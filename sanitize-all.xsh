#!/usr/bin/xonsh

container = 'nvcr.io/nvidia/pytorch'
cuda_to_container_versions = {
    '10.2': '19.12-py3',
    '11.0': '20.07-py3',
    '11.1': '20.12-py3',
    '11.2': '21.02-py3',
}

for cuda_version, container_version in cuda_to_container_versions.items():
    print('Running for CUDA:', cuda_version)
    docker run -it -v $PWD:/w -w /w @(f'{container}:{container_version}') bash -c f'''
        set -eux
        conda create -n env python=3.9 --yes
        export PYTHON_PATH=/opt/conda/envs/env/bin
        $PYTHON_PATH/pip install -r requirements.txt
        $PYTHON_PATH/python cuda-local-memory.py || true
        mv local-memory-usage.json local-memory-usage-{cuda_version}.json
    '''