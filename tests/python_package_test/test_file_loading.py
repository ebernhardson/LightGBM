import contextlib
import itertools
import lightgbm
import numpy as np
import os
import pytest
import string
import tempfile


@contextlib.contextmanager
def relatedTempFile(f, append):
    with open(f.name + append, "w") as f:
        try:
            yield f
        finally:
            os.unlink(f.name)


def test_very_long_header():
    ncols = 20000
    cols = [''.join(x) for x in itertools.product(*([string.ascii_lowercase]*4))][:ncols]
    with tempfile.NamedTemporaryFile() as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join(['1', '0'] * (ncols/2)) + "\n")
        f.write(",".join(['0', '1'] * (ncols/2)) + "\n")
        f.flush()
        ds = lightgbm.Dataset(f.name, params={
            "min_data": 1, "min_data_in_bin": 1,
            "has_header": True,
        }).construct()
        assert ds.num_feature() == (ncols-1)
        assert np.array_equal([1.0, 0.0], ds.get_label())
        

def test_concatenated_files():
    with tempfile.NamedTemporaryFile() as f:
        f.write("1,0,1\n")
        f.write("0,1,1\n")
        f.write("0,1,0\n")
        f.write("1,0,0\n")
        f.flush()
        ds = lightgbm.Dataset(','.join([f.name] * 3), params={
            "min_data": 1, "min_data_in_bin": 1,
        }).construct()
        assert ds.get_label().shape[0] == 12


def test_concatenated_files_with_query():
    with tempfile.NamedTemporaryFile() as f, relatedTempFile(f, ".query") as f_q:
        f.write("1,0,1\n")
        f.write("0,1,1\n")
        f.write("0,1,0\n")
        f.write("1,0,0\n")
        f.flush()
        f_q.write("2\n2\n")
        f_q.flush()
        ds = lightgbm.Dataset(','.join([f.name] * 3), params={
            "min_data": 1, "min_data_in_bin": 1,
        }).construct()
        group = np.asarray(ds.get_group())
        assert group.shape[0] == 6
        assert all(group == 2)


def test_concatenated_files_with_weight():
    with tempfile.NamedTemporaryFile() as f, relatedTempFile(f, ".weight") as f_w:
        f.write("1,0,1\n")
        f.write("0,1,1\n")
        f.write("0,1,0\n")
        f.write("1,0,0\n")
        f.flush()
        f_w.write("1\n2\n3\n4\n")
        f_w.flush()
        ds = lightgbm.Dataset(','.join([f.name] * 3), params={
            "min_data": 1, "min_data_in_bin": 1,
        }).construct()
        weight = ds.get_weight()
        assert weight.shape[0] == 12
        assert np.array_equal([1, 2, 3, 4] * 3, weight)
