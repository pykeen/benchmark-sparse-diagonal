"""Benchmarks for random walk positional encoding."""
from contextlib import suppress
import hashlib
import itertools
import pathlib
from collections import ChainMap
from typing import Any, Collection, Iterable, Literal, Mapping, Optional, Tuple

import pystow
import torch
import pandas
import seaborn
import numpy
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


SparseMatrixFormat = Literal["coo", "csr"]


def create_sparse_matrix(
    n: int, density: float, device: Optional[torch.device] = None, matrix_format: SparseMatrixFormat = "coo"
) -> torch.Tensor:
    """Create a sparse matrix of the given size and (maximum) density."""
    # TODO: the sparsity pattern may not be very natural for a graph
    nnz = int(n**2 * density)
    indices = torch.randint(n, size=(2, nnz), device=device)
    values = torch.ones(nnz, device=device)
    x = torch.sparse_coo_tensor(indices=indices, values=values, size=(n, n))
    if matrix_format == "coo":
        return x
    return x.to_sparse_csr()


def sparse_eye(
    n: int, device: Optional[torch.device] = None, matrix_format: SparseMatrixFormat = "coo"
) -> torch.Tensor:
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
    x = torch.sparse_coo_tensor(indices=indices, values=torch.ones(n, device=device))
    if matrix_format == "coo":
        return x
    return x.to_sparse_csr()


class DiagonalExtraction(BenchmarkItem):
    def prepare(
        self,
        n: int,
        density: float,
        device: Optional[torch.device] = None,
        matrix_format: SparseMatrixFormat = "coo",
        **kwargs,
    ) -> Mapping[str, Any]:
        # TODO: the sparsity pattern may not be very natural for a graph
        matrix = create_sparse_matrix(n=n, density=density, device=device, matrix_format=matrix_format)
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


class ExplicitPythonCSR(DiagonalExtraction):
    """Extract diagonal by item access and Python for-loop."""

    def call(self, matrix: torch.Tensor, **kwargs):
        if not matrix.is_sparse_csr:
            raise NotImplementedError
        n = matrix.shape[0]
        d = torch.zeros(n, device=matrix.device)

        crow = matrix.crow_indices()
        col = matrix.col_indices()
        values = matrix.values()

        for i, (start, stop) in enumerate(zip(crow, crow[1:])):
            this_col = col[start:stop]
            this_values = values[start:stop]
            v = this_values[this_col == i]
            if v.numel():
                d[i] = v
        return d


class Coalesce(DiagonalExtraction):
    def prepare(self, n: int, matrix_format: SparseMatrixFormat = "coo", **kwargs) -> Mapping[str, Any]:
        kwargs = super().prepare(n=n, **kwargs)
        return ChainMap(
            dict(eye=sparse_eye(n=n, device=kwargs.get("matrix").device), matrix_format=matrix_format), kwargs
        )

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


class DenseDiag(DiagonalExtraction):
    """Extract diagonal by item access and Python for-loop."""

    def call(self, matrix: torch.Tensor, **kwargs):
        return torch.diag(matrix.to_dense())


def main():
    """Time the different variants."""
    # fb15k237 sparsity ~ 300k / 15_000**2 = 0.001
    density_grid = (1.0e-05, 1.0e-04, 1.0e-03, 1.0e-02)
    n_grid = (500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000)
    devices = sorted({torch.device("cpu"), resolve_device(device=None)}, key=lambda d: d.type)
    cls_grid = (ExplicitPython, ExplicitPythonCSR, Coalesce, ManualCoalesce, DenseDiag)
    format_grid = ("coo", "csr")
    data = []
    with tqdm(
        (
            cls(kwargs=dict(n=n, density=density, device=device, matrix_format=matrix_format))
            for cls, n, density, device, matrix_format in itertools.product(
                cls_grid,
                n_grid,
                density_grid,
                devices,
                format_grid,
            )
        ),
        total=numpy.prod(list(map(len, (cls_grid, n_grid, density_grid, devices, format_grid)))),
    ) as progress:
        for task in progress:
            progress.set_description(str(task))
            n, density, device, matrix_format = [task.kwargs[k] for k in ("n", "density", "device", "matrix_format")]
            # too slow
            if n > 8_000 and isinstance(task, ExplicitPython):
                continue
            nnz = n**2 * density
            # high memory consumption
            if nnz > 10_000_000:
                continue
            with suppress(NotImplementedError, RuntimeError):
                measurement = task.buffered_measure()
                data.append(
                    (
                        task.name,
                        n,
                        density,
                        device,
                        matrix_format,
                        measurement.median,
                        measurement.mean,
                        measurement.iqr,
                    )
                )
    # create table
    df = pandas.DataFrame(
        data=data, columns=["name", "n", "density", "device", "matrix_format", "median", "mean", "iqr"]
    )
    df["device"] = df["device"].astype(str)
    keys = ["device", "n", "density", "name"]
    df = df.sort_values(by=keys)
    # create plot
    grid = seaborn.relplot(
        data=df,
        x="n",
        y="median",
        col="device",
        size="density",
        kind="line",
        hue="name",
        style="matrix_format",
    )
    grid.set(yscale="log", xscale="log")
    grid.savefig("./img/comparison.svg")


if __name__ == "__main__":
    main()
