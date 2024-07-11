import struct
from pathlib import Path

import numpy as np
import pytest
from scape_util import ScapeImageDecoder, SCAPEVirtualStack

SAMPLE_PATH = Path(__file__).parent.joinpath("sample.3DU16")


@pytest.fixture
def file():
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH

    # _, z_scale, y_scale, x_scale
    scale = (3, 0.9, 0.455, 0.455)
    # _, T, C, Z, Y, X
    metadata = (5, 4, 2, 2, 2, 3)
    # C, Z, Y, Z, [data...]
    vol = "(2,2,2,3,1,2,3,4,5,6,0,0,0,0,0,0,7,8,9,10,11,12,2,2,2,2,2,2)"
    vol = eval(vol)
    raw = struct.pack(
        ">i3d6i4i24H4i24H4i24H4i24H",
        *scale,
        *metadata,
        *vol,
        *vol,
        *vol,
        *vol,
    )

    with open(SAMPLE_PATH, "wb") as f:
        f.write(raw)

    return SAMPLE_PATH


def testing_parser(file):
    with SCAPEVirtualStack(file) as stack:
        v1 = stack.get_volume_raw(0)
        v2 = stack.get_volume_raw(1)
        np.testing.assert_equal(v1, v2)


def testing_not_exist(file):
    data_path = file.with_name("not_exists.3du16")
    with pytest.raises(FileNotFoundError):
        print(ScapeImageDecoder.from_3DU16(data_path))


def test_readfile(file):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with SCAPEVirtualStack(SAMPLE_PATH) as stack:
        for i in range(4):
            for fmt in ("org", "u8", "f32"):
                out = test_out.joinpath(SAMPLE_PATH.stem + f"_t={i:0>5d}_{fmt}.tif")
                stack.save_volume_to_tiff(out, i, conversion=fmt)


def test_read_volume_fail(file):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with pytest.raises(IndexError):
        with SCAPEVirtualStack(SAMPLE_PATH) as stack:
            stack.get_volume_raw(100)
