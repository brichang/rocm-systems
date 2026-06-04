# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the marker-CSV backend-suffix split in
``utils.utils_profile``: ``_parse_function_backend`` and
``_augment_marker_csv``."""

import csv
from pathlib import Path

import common  # noqa: F401
import pytest

from utils import utils_profile


@pytest.mark.parametrize(
    "raw, expect_function, expect_backend",
    [
        ("aten::add", "aten::add", "torch"),
        ("aten::add|torch", "aten::add", "torch"),
        ("triton.CompiledKernel.k|triton", "triton.CompiledKernel.k", "triton"),
        ("", "", "torch"),
        (None, "", "torch"),
    ],
)
def test_parse_function_backend(raw, expect_function, expect_backend):
    fn, backend = utils_profile._parse_function_backend(raw)
    assert fn == expect_function
    assert backend == expect_backend


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path):
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), reader.fieldnames


def test_augment_marker_csv_splits_backend_into_dedicated_column(tmp_path):
    src = tmp_path / "marker.csv"
    dst = tmp_path / "out.csv"
    _write_csv(
        src,
        ["Function", "Start", "End"],
        [
            {"Function": "aten::add", "Start": "1", "End": "2"},
            {"Function": "aten::mul|torch", "Start": "3", "End": "4"},
            {"Function": "triton.k|triton", "Start": "5", "End": "6"},
        ],
    )

    utils_profile._augment_marker_csv(str(src), str(dst))

    rows, fieldnames = _read_csv(dst)
    assert "Backend" in fieldnames
    assert [r["Function"] for r in rows] == ["aten::add", "aten::mul", "triton.k"]
    assert [r["Backend"] for r in rows] == ["torch", "torch", "triton"]


def test_augment_marker_csv_passthrough_when_function_column_missing(tmp_path):
    src = tmp_path / "weird.csv"
    dst = tmp_path / "out.csv"
    _write_csv(src, ["Name", "Value"], [{"Name": "x", "Value": "1"}])

    utils_profile._augment_marker_csv(str(src), str(dst))

    assert dst.read_bytes() == src.read_bytes()


def test_augment_marker_csv_default_backend_is_torch(tmp_path):
    src = tmp_path / "marker.csv"
    dst = tmp_path / "out.csv"
    _write_csv(src, ["Function"], [{"Function": "aten::sum"}])

    utils_profile._augment_marker_csv(str(src), str(dst))

    rows, _ = _read_csv(dst)
    assert rows == [{"Function": "aten::sum", "Backend": "torch"}]
