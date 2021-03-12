import glob
import asyncio
import multiprocessing
import json
import colorama
import sys
import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--demangle', action='store_true')
args = parser.parse_args()

colorama.init()

ncpus = multiprocessing.cpu_count()

files = set(glob.glob('pytorch/aten/src/ATen/native/cuda/*.cu'))

nvcc = 'nvcc'
target = ['-dc', '-o', '/dev/null']
sanitize = ['-Xptxas=-Werror', '-Xptxas=-warn-lmem-usage,-warn-spills']
features = ['--extended-lambda', '--expt-relaxed-constexpr', '--keep']
archs = ['-gencode', 'arch=compute_70,code=sm_70']
defs = ['-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__']
includes = ['-Ipytorch', '-Ipytorch/aten/src/', '-Ipytorch/build', '-Ipytorch/build/aten/src', '-Ipytorch/build/caffe2/aten/src']
if not os.path.isdir('/usr/local/cuda/include/cub'):
    includes.append('-Ipytorch/third_party/cub')

keep_dir = 'keep'


def get_nvcc_command(file):
    flags = [*target, *sanitize, *features, *archs, *defs, *includes]
    command = ' '.join([nvcc, file, *flags])
    return command

errors = {}


def is_local_memory_error(text):
    if not text.startswith('ptxas error'):
        return False
    if 'Local memory' not in text and 'local memory' not in text:
        return False
    return True


async def demangle(symbol):
    proc = await asyncio.create_subprocess_shell(
        f'c++filt {symbol}',
        stdout=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    return stdout.decode().strip()


async def get_function_name(error):
    error = error.split("'")
    assert len(error) == 3
    return await demangle(error[1]) if args.demangle else error[1]


def save_or_remove_keep_files(original_filename, keep):
    _, fn = os.path.split(original_filename)
    pattern = fn[:-len('cu')] + '*'
    files = glob.glob(pattern)
    for f in files:
        if keep:
            shutil.move(f, keep_dir)
        else:
            os.remove(f)


async def run_single(file):
    command = get_nvcc_command(file)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    stderr_ = stderr.decode()
    if proc.returncode != 0:
        stderr = stderr_.split('\n')
        stderr = [await get_function_name(e) for e in stderr if is_local_memory_error(e)]
        if len(stderr) > 0:
            print(colorama.Fore.RED + 'FAIL:', file)
            errors[file] = stderr
            save_or_remove_keep_files(file, True)
        else:
            print(colorama.Fore.MAGENTA + 'UNKNOWN:', file)
            print(stderr_)
            save_or_remove_keep_files(file, False)
    else:
        print(colorama.Fore.GREEN + 'PASS:', file)
        save_or_remove_keep_files(file, False)


def rm_then_mkdir(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)


async def main():
    tasks = set()
    rm_then_mkdir(keep_dir)
    while len(files) > 0:
        if len(tasks) < ncpus:
            f = next(iter(files))
            tasks.add(asyncio.create_task(run_single(f)))
            files.remove(f)
        else:
            for t in tasks:
                if t.done():
                    break
            else:
                await asyncio.sleep(0.1)
            tasks.remove(t)
            await t


asyncio.run(main())

with open('local-memory-usage.json', 'w') as f:
    json.dump(errors, f)

if len(errors) > 0:
    sys.exit(1)
