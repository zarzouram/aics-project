# Reference
import subprocess


def get_gpu_memory():
    """Get the current gpu usage.
    Reference: https://stackoverflow.com/a/49596019

    Returns
    -------
    usage: list
        Values of memory usage per GPU as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ])
    # Convert lines into a dictionary
    result = result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]

    return gpu_memory


def get_gpus_avail():
    """Get the GPU ids that have memory usage less than 50%
    """
    memory_usage = get_gpu_memory()

    memory_usage_percnt = [m / 11178 for m in memory_usage]
    cuda_ids = [(i, m) for i, m in enumerate(memory_usage_percnt) if m <= 0.4]

    header = ["cuda id", "Memory usage"]
    no_gpu_mssg = "No available GPU"
    if cuda_ids:
        print(f"{header[0]:^10}{header[1]:^15}")
        for (idx, m) in cuda_ids:
            print(f"{idx:^10}{m:^15.2%}")
    else:
        print(f"{no_gpu_mssg:-^25}")
        print(f"{header[0]:^10}{header[1]:^15}")
        for idx, m in enumerate(memory_usage_percnt):
            print(f"{idx:^10}{m:^15.2%}")

    return sorted(cuda_ids, key=lambda tup: tup[1]) if cuda_ids else cuda_ids