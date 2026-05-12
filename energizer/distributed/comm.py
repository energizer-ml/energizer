from __future__ import annotations


def allreduce(world, tensor):
    return world.allreduce(tensor)
