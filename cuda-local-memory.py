import glob
import asyncio
import multiprocessing

ncpus = multiprocessing.cpu_count()
print(ncpus)

files = glob.glob('pytorch/aten/src/ATen/native/cuda/*.cu')

nvcc = 'nvcc'
target = ['-dc', '-o', '/dev/null']
sanitize = ['-Xptxas=-Werror', '-Xptxas=-warn-lmem-usage,-warn-spills']
features = ['--extended-lambda', '--expt-relaxed-constexpr']
defs = ['-D__CUDA_NO_HALF_OPERATORS__']
includes = ['-Ipytorch', '-Ipytorch/aten/src/', '-Ipytorch/build', '-Ipytorch/build/aten/src', '-Ipytorch/build/caffe2/aten/src']
flags = [*target, *sanitize, *features, *defs, *includes]


async def run_single(file):
    command = ' '.join([nvcc, file, *flags])
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    stderr = stderr.decode()
    if proc.returncode != 0:
        print(stderr)


async def main():
    tasks = set()
    for f in files:
        if len(tasks) < ncpus:
            tasks.add(asyncio.create_task(run_single(f)))
        else:
            for t in tasks:
                if t.done():
                    break
            else:
                await asyncio.sleep(0.1)
            tasks.remove(t)
            await t


asyncio.run(main())