import struct
from pathlib import Path

import numpy as np
import pytest
from scape_utils import ScapeImageDecoder, ScapeVirtualStack


T, C, Z, Y, X = 11, 2, 3, 5, 7


@pytest.fixture
def file():
    sample_path = Path(__file__).parent.joinpath("sample.3DU16")
    if sample_path.is_file():
        return sample_path

    # 3, z_scale, y_scale, x_scale
    z_scale, y_scale, x_scale = 0.9, 0.455, 0.455
    scale = (3, z_scale, y_scale, x_scale)

    # 5, T, C, Z, Y, X
    metadata = (5, T, C, Z, Y, X)
    # 1 volume of data
    # C, Z, Y, Z, [data...]
    data = [C, Z, Y, X, *range(C * Z * Y * X)]

    raw = struct.pack(">i3d6i" + f"4i{C*Z*Y*X:d}H" * T, *scale, *metadata, *[*data] * T)

    with open(sample_path, "wb") as f:
        f.write(raw)

    return sample_path


def testing_parser(file):
    with ScapeVirtualStack(file) as stack:
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
    with ScapeVirtualStack(SAMPLE_PATH) as stack, pytest.warns(UserWarning):
        for i in range(4):
            for fmt in ("org", "u8", "f32"):
                out = test_out.joinpath(SAMPLE_PATH.stem + f"_t={i:0>5d}_{fmt}.tif")
                stack.save_volume_to_tiff(out, i, conversion=fmt)


def test_read_volume_fail(file):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    with pytest.raises(IndexError):
        with ScapeVirtualStack(SAMPLE_PATH) as stack:
            stack.get_volume_raw(100)


def test_read_volume_as_imagej(file):
    with ScapeVirtualStack(file) as stack:
        # This method should return 1 volume of image stack with format (1, Z, C, Y, X)
        img = stack.get_imagej_volume(3)
        assert img.ndim == 5
        assert img.shape == (1, Z, C, Y, X)


def test_save_all_volume(file: Path):
    test_out = file.parent.joinpath("tmp")
    test_out.mkdir(exist_ok=True)
    out = test_out.joinpath(file.stem + ".tif")
    with ScapeVirtualStack(SAMPLE_PATH) as stack:
        stack.save_all_volume_to_tiff(out)
