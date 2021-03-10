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
    docker run -it -v $PWD:/w @(f'{container}:{container_version}') bash -c f'''
        set -eux
        cd /w
        pushd pytorch
        pip install dataclasses
        python setup.py clean
        python setup.py install
        popd
        conda create -n env python=3.9 --yes
        export PATH=/opt/conda/envs/env/bin:$PATH
        pip install -r requirements.txt
        python cuda-local-memory.py
        mv local-memory-usage.json local-memory-usage-{cuda_version}.json
    '''