"""FIFO admission for complete agent segments and for numerical evaluation."""
import asyncio
from concurrent.futures import Future
from queue import Queue
from threading import Thread
from time import perf_counter


class AgentQueue:
    """Limit concurrent complete work segments, admitting waiters in arrival order."""

    def __init__(self, concurrency):
        self._admission = asyncio.Lock()
        self._slots = asyncio.BoundedSemaphore(concurrency)

    async def run(self, work, *args):
        requested = perf_counter()
        async with self._admission:
            await self._slots.acquire()
        task = asyncio.create_task(asyncio.to_thread(
            work, *args, agent_queue_seconds=perf_counter() - requested,
        ))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A running blocking call retains its seat until it actually exits.
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._slots.release()


class EvaluationQueue:
    """Serve ready candidates in submission order on a single numerical pool."""

    def __init__(self, evaluate):
        self._evaluate = evaluate
        self._ready = Queue()
        self._worker = Thread(target=self._serve, name='xc100-evaluation')

    def __enter__(self):
        self._worker.start()
        return self

    def __exit__(self, *args):
        self._ready.put(None)
        self._worker.join()

    def submit(self, candidate):
        future = Future()
        self._ready.put((candidate, perf_counter(), future))
        return future

    def _serve(self):
        while (request := self._ready.get()) is not None:
            candidate, submitted, future = request
            if not future.set_running_or_notify_cancel():
                continue
            started = perf_counter()
            try:
                result = self._evaluate(candidate)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result((result, started - submitted, perf_counter() - started))
