"""Benchmarks for random walk positional encoding."""
import hashlib
import itertools
import pathlib
from collections import ChainMap
from typing import Any, Collection, Iterable, Mapping, Optional, Tuple

import pystow
import torch
from pykeen.utils import resolve_device
from torch.utils.benchmark import Measurement, Timer
from tqdm.auto import tqdm

pystow_module = pystow.Module(base=pathlib.Path(__file__).parent.joinpath("data"))


class BenchmarkItem:
    def __init__(self, name: Optional[str] = None, kwargs: Optional[Mapping[str, Any]] = None) -> None:
        self.name = name or self.__class__.__name__
        self.kwargs = kwargs or {}

    def _build_statement(self, prepared_kwargs: Iterable[str]) -> str:
        return "".join(
            (
                self.name,
                "(",
                ", ".join(f"{key}={key}" for key in prepared_kwargs),
                ")",
            )
        )

    def measure(self) -> Measurement:
        # build statement
        prepared_kwargs = self.prepare(**self.kwargs)
        kwargs = dict(
            **{self.name: self.call},
            **prepared_kwargs,
        )
        stmt = self._build_statement(prepared_kwargs=prepared_kwargs)
        return Timer(stmt=stmt, globals=kwargs).blocked_autorange()

    def iter_key_parts(self) -> Iterable[Tuple[str, Any]]:
        yield from self.kwargs.items()

    def __str__(self) -> str:
        return "".join((f"{self.name}(", ", ".join(f"{k}={v}" for k, v in sorted(self.iter_key_parts())), ")"))

    def key(self):
        return hashlib.sha512(str(self).encode("utf8")).hexdigest()[:32]

    def buffered_measure(self, force: bool = False) -> Measurement:
        path = pystow_module.join(self.name, name=self.key() + ".pt")
        if not force and path.is_file():
            return torch.load(path)

        measurement = self.measure()
        torch.save(measurement, path)
        return measurement

    def call(self, **kwargs):
        raise NotImplementedError

    def prepare(self, **kwargs) -> Mapping[str, Any]:
        return kwargs


def dict_difference(d: Mapping[str, Any], exclude: Collection[str]) -> Mapping[str, Any]:
    return {k: v for k, v in d.items() if k not in exclude}


def create_sparse_matrix(n: int, density: float, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a sparse matrix of the given size and (maximum) density."""
    # TODO: the sparsity pattern may not be very natural for a graph
    nnz = int(n**2 * density)
    indices = torch.randint(n, size=(2, nnz), device=device)
    values = torch.ones(nnz, device=device)
    return torch.sparse_coo_tensor(indices=indices, values=values, size=(n, n))


def sparse_eye(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a sparse diagonal matrix.

    .. note ::
        this is a work-around as long as there is no torch built-in

    :param n:
        the size
    :return: shape: `(n, n)`, sparse
        a sparse diagonal matrix
    """
    indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    return torch.sparse_coo_tensor(indices=indices, values=torch.ones(n, device=device))


class DiagonalExtraction(BenchmarkItem):
    def prepare(self, n: int, density: float, device: Optional[torch.device] = None, **kwargs) -> Mapping[str, Any]:
        # TODO: the sparsity pattern may not be very natural for a graph
        matrix = create_sparse_matrix(n=n, density=density, device=device)
        return ChainMap(dict(matrix=matrix), kwargs)

    def iter_key_parts(self) -> Iterable[Tuple[str, Any]]:
        for key, value in super().iter_key_parts():
            if key == "device" and value is not None:
                assert isinstance(value, torch.device)
                value = value.type
            yield key, value


class ExplicitPython(DiagonalExtraction):
    """Extract diagonal by item access and Python for-loop."""

    def call(self, matrix: torch.Tensor, **kwargs):
        n = matrix.shape[0]
        d = torch.zeros(n, device=matrix.device)

        # torch.sparse.coo only allows direct numbers here, can't feed an eye tensor here
        for i in range(n):
            d[i] = matrix[i, i]

        return d


class Coalesce(DiagonalExtraction):
    def prepare(self, n: int, **kwargs) -> Mapping[str, Any]:
        kwargs = super().prepare(n=n, **kwargs)
        return ChainMap(dict(eye=sparse_eye(n=n, device=kwargs.get("matrix").device)), kwargs)

    def call(self, matrix: torch.Tensor, eye: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract diagonal using coalesce."""
        n = matrix.shape[0]
        d = torch.zeros(n, device=matrix.device)

        d_sparse = (matrix * eye).coalesce()
        indices = d_sparse.indices()
        values = d_sparse.values()
        d[indices] = values

        return d


class ManualCoalesce(DiagonalExtraction):
    def call(self, matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract diagonal using a manual implementation accessing private functions of an instable API."""
        n = matrix.shape[0]
        d = torch.zeros(n, device=matrix.device)

        indices = matrix._indices()
        mask = indices[0] == indices[1]
        diagonal_values = matrix._values()[mask]
        diagonal_indices = indices[0][mask]

        return d.scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)


def main():
    """Time the different variants."""
    # fb15k237 sparsity ~ 300k / 15_000**2 = 0.001
    density_grid = (1.0e-04, 1.0e-03, 1.0e-02)
    n_grid = (4_000, 8_000, 16_000, 32_000, 64_000)
    devices = sorted({torch.device("cpu"), resolve_device(device=None)})
    measurements = []
    with tqdm(
        (
            cls(kwargs=dict(n=n, density=density, device=device))
            for cls, n, density, device in itertools.product(
                (ExplicitPython, Coalesce, ManualCoalesce),
                n_grid,
                density_grid,
                devices,
            )
        )
    ) as progress:
        for task in progress:
            progress.set_description(str(task))
            measurements.append(task.buffered_measure())
    print(measurements)


if __name__ == "__main__":
    main()
