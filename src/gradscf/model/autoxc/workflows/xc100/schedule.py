"""Plan disjoint same-node CPU sets and balance estimated shard times."""


def affinity_plan(cpus, *, gpu=True, cpu_workers=None, threads=4):
    """Return the coordinator/GPU mask first when a GPU is allocated."""
    cpus = sorted(set(cpus))
    count = len(cpus)
    if count < 1 or threads < 1 or (cpu_workers is not None and cpu_workers < 1):
        raise ValueError('Positive CPU count, worker count and threads are required')
    if gpu and count < 8:
        return (tuple(cpus),)
    host = min(8, max(4, count // 4)) if gpu else 0
    masks = [tuple(cpus[:host])] if gpu else []
    remaining = count - host
    workers = min(100, max(1, remaining // threads))
    if cpu_workers is not None:
        workers = min(cpu_workers, workers)
    size, extra = divmod(remaining, workers)
    start = host
    for worker in range(workers):
        stop = start + size + (worker < extra)
        masks.append(tuple(cpus[start:stop]))
        start = stop
    return tuple(masks)


def distribute(costs, workers):
    """Assign long cases first to the earliest available compatible worker."""
    if workers < 1 or any(len(row) != workers for row in costs.values()):
        raise ValueError('Supply one estimated time per system and worker')
    shards = [[] for _ in range(workers)]
    loads = [0.] * workers
    for sample, times in sorted(costs.items(), key=lambda item: (-max(item[1]), item[0])):
        rank = min(range(workers), key=lambda i: (loads[i] + times[i], i))
        shards[rank].append(sample)
        loads[rank] += times[rank]
    return tuple(tuple(sorted(shard)) for shard in shards)
