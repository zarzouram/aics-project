# Reference
import subprocess


def get_gpu_memory():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ])
    # Convert lines into a dictionary
    result = result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    # gpu_memory_map = list(zip(range(len(gpu_memory)), gpu_memory))

    return gpu_memory
