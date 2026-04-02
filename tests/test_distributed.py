import queue
import threading
import unittest

import numpy as np

import energizer


def _build_world_pair(port: int):
    worlds = queue.Queue()
    errors = queue.Queue()

    def coordinator():
        try:
            worlds.put(("rank0", energizer.World.init(0, 2, "127.0.0.1", port)))
        except Exception as exc:
            errors.put(exc)

    thread = threading.Thread(target=coordinator, daemon=True)
    thread.start()

    worker = energizer.World.init(1, 2, "127.0.0.1", port)
    role, coordinator_world = worlds.get(timeout=5)
    assert role == "rank0"
    thread.join(timeout=5)

    if not errors.empty():
        raise errors.get()

    return coordinator_world, worker


class TestDistributed(unittest.TestCase):
    def test_world_allreduce_averages_across_ranks(self):
        world0, world1 = _build_world_pair(29610)
        try:
            results = {}
            errors = queue.Queue()

            def run(rank, world, value):
                try:
                    results[rank] = world.allreduce(value)
                except Exception as exc:
                    errors.put(exc)

            t0 = threading.Thread(
                target=run,
                args=(0, world0, np.array([2.0, 4.0], dtype=np.float32)),
            )
            t1 = threading.Thread(
                target=run,
                args=(1, world1, np.array([6.0, 8.0], dtype=np.float32)),
            )
            t0.start()
            t1.start()
            t0.join(timeout=5)
            t1.join(timeout=5)
            if not errors.empty():
                raise errors.get()

            expected = np.array([4.0, 6.0], dtype=np.float32)
            np.testing.assert_allclose(results[0], expected)
            np.testing.assert_allclose(results[1], expected)
        finally:
            world0.close()
            world1.close()

    def test_data_parallel_syncs_initial_weights_and_optimizer_step(self):
        world0, world1 = _build_world_pair(29611)
        try:
            rank0_model = energizer.Linear(2, 1, bias=False)
            rank1_model = energizer.Linear(2, 1, bias=False)

            rank0_model.weight.data = np.array([[1.0, 3.0]], dtype=np.float32)
            rank1_model.weight.data = np.array([[9.0, 9.0]], dtype=np.float32)

            results = {}
            errors = queue.Queue()

            def run_rank0():
                try:
                    model = energizer.DataParallel(rank0_model, world0)
                    optimizer = energizer.SGD(model.parameters(), lr=0.5)
                    model.weight.grad = energizer.Tensor(
                        np.array([[2.0, 6.0]], dtype=np.float32)
                    )
                    optimizer.step()
                    results[0] = model.weight.data.copy()
                except Exception as exc:
                    errors.put(exc)

            def run_rank1():
                try:
                    model = energizer.DataParallel(rank1_model, world1)
                    optimizer = energizer.SGD(model.parameters(), lr=0.5)
                    synced_initial = model.weight.data.copy()
                    model.weight.grad = energizer.Tensor(
                        np.array([[4.0, 10.0]], dtype=np.float32)
                    )
                    optimizer.step()
                    results[1] = {
                        "initial": synced_initial,
                        "updated": model.weight.data.copy(),
                    }
                except Exception as exc:
                    errors.put(exc)

            t0 = threading.Thread(target=run_rank0)
            t1 = threading.Thread(target=run_rank1)
            t0.start()
            t1.start()
            t0.join(timeout=5)
            t1.join(timeout=5)
            if not errors.empty():
                raise errors.get()

            np.testing.assert_allclose(
                results[1]["initial"],
                np.array([[1.0, 3.0]], dtype=np.float32),
            )
            expected_weight = np.array([[-0.5, -1.0]], dtype=np.float32)
            np.testing.assert_allclose(results[0], expected_weight)
            np.testing.assert_allclose(results[1]["updated"], expected_weight)
        finally:
            world0.close()
            world1.close()


if __name__ == "__main__":
    unittest.main()
