"""Point-parallel evaluation of a PointFunctional (DESIGN.md Sec. 4.6).

The N points are cut into fixed-size chunks by GLOBAL index (chunk c = points [c*chunk, (c+1)*chunk), the last one padded
with zero-density points). Chunk c is evaluated by rank c mod size. Every chunk has the same shape (one compiled shape per
process), and its result does not depend on the rank count. The root gathers the chunks and sums the chunk energies in chunk
order, so the result is bitwise independent of the number of ranks.

comm: an mpi4py communicator or SerialComm (one process); the lowercase object methods (scatter, gather, bcast) are used.
"""
from __future__ import annotations


from ..functional.contract import PointFunctional, PointInputs, PointOutputs

CHUNK = 16384          # global constant: points per chunk (one compiled shape per process)


class SerialComm:
    """The subset of the mpi4py communicator interface used here, for one process."""

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        return obj

    def scatter(self, objs, root=0):
        return objs[0]

    def gather(self, obj, root=0):
        return [obj]

    def allreduce(self, obj, op=None):
        return obj

    def Barrier(self):
        return None


def evaluate_points(functional: PointFunctional, x: PointInputs | None, comm=None, chunk: int = CHUNK) -> PointOutputs | None:
    """Evaluate on all points (x given on the root only). Returns the full PointOutputs on the root, None elsewhere."""
    comm = comm or SerialComm()
    rank, size = comm.Get_rank(), comm.Get_size()
    bad = None
    if rank == 0:
        try:
            x.validate()
        except Exception as exc:
            bad = exc
    bad = comm.bcast(bad, root=0)
    if bad is not None:
        raise bad
    if rank == 0:
        n = x.n
        nchunk = max(1, -(-n // chunk))
        pieces = [x.slice(c * chunk, min((c + 1) * chunk, n)) for c in range(nchunk)]
        work = [[(c, pieces[c]) for c in range(r, nchunk, size)] for r in range(size)]
    else:
        work = None
    mine = comm.scatter(work, root=0)
    results = []
    error = None
    for c, piece in mine:
        try:
            out = functional.evaluate(piece.padded(chunk)).validate(chunk)
        except Exception as exc:                       # reported collectively below, then re-raised on every rank
            error = f"rank {rank}, chunk {c}: {type(exc).__name__}: {exc}"
            break
        results.append((c, out.e_sum, out.slice(0, piece.n)))
    errors = comm.gather(error, root=0)
    errors = comm.bcast([e for e in (errors or []) if e], root=0)
    if errors:
        raise RuntimeError("point evaluation failed: " + "; ".join(errors))
    gathered = comm.gather(results, root=0)
    if rank != 0:
        return None
    flat = sorted((item for part in gathered for item in part), key=lambda t: t[0])
    if [t[0] for t in flat] != list(range(nchunk)):
        raise RuntimeError("chunk bookkeeping error")
    e_sum = 0.0
    for _, e, _ in flat:                               # fixed (chunk) order: rank-count independent
        e_sum += e
    return PointOutputs.concatenate([t[2] for t in flat], e_sum)
