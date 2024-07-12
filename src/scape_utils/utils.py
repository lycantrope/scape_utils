import mmap
import struct
from io import BufferedReader
from pathlib import Path
from typing import NamedTuple, Union
from warnings import warn

import attrs
import numpy as np
import numpy.typing as npt
import tifffile

LUT_TABLE = {
    "u8": np.floor(np.arange(65536) / 256).astype(np.uint8),
    "f32": np.linspace(0.0, 1.0, 65536, endpoint=True, dtype=np.float32),
}


class ScapeImageDecoder(NamedTuple):
    x_scale: float
    y_scale: float
    z_scale: float
    n_frame: int
    n_channel: int
    depth: int
    height: int
    width: int

    @classmethod
    def from_3DU16(cls, file: Union[str, Path]) -> "ScapeImageDecoder":
        if file.suffix.lower() != ".3du16":
            raise TypeError(f"Only support (*.3d16 or *.3DU16): {file.name}")
        try:
            # skip bytes from 0 to 3 and 28-31
            raw = Path(file).open("rb").read(52)
            vals = struct.unpack(">i3d6i", raw)
            (z_scale, y_scale, x_scale) = vals[1:4]
            (n_frame, n_channel, depth, height, width) = vals[5:10]
        except struct.error as e:
            raise TypeError(f"This file might not be a valid 3DU16 file: {e}")

        return cls(x_scale, y_scale, z_scale, n_frame, n_channel, depth, height, width)

    def get_volume(self, raf: mmap, index: int) -> npt.NDArray:
        if (index < 0) | (index >= self.n_frame):
            raise IndexError(f"Index is out of boundary: {index}")

        offset = 68 + index * self.bytes_per_volume
        raw = raf[offset : offset + self.bytes_per_volume]
        dt = np.dtype("u2")
        # newbyteorder is not inplace reassign is required.
        dt = dt.newbyteorder(">")
        _, C, Z, Y, X = self.shape
        img = np.frombuffer(
            raw,
            dtype=dt,
            count=self.pixels_per_volume,
        ).reshape((C, Z, Y, X))
        return img

    @property
    def bytes_per_xy(self) -> int:
        return self.height * self.width * 2

    @property
    def bytes_per_xyz(self) -> int:
        return self.height * self.width * 2 * self.depth

    @property
    def bytes_per_volume(self) -> int:
        gap = 16
        return self.pixels_per_volume * 2 + gap

    @property
    def pixels_per_volume(self) -> int:
        return self.height * self.width * self.depth * self.n_channel

    @property
    def shape(self):
        # TCZYX
        return (self.n_frame, self.n_channel, self.depth, self.height, self.width)


@attrs.define
class ScapeVirtualStack:

    filepath: Path
    header: ScapeImageDecoder = attrs.field(init=False)

    raf: mmap.mmap = attrs.field(init=False, default=None, repr=False)
    handler: BufferedReader = attrs.field(init=False, default=None, repr=False)

    def __attrs_post_init__(self):
        self.header = ScapeImageDecoder.from_3DU16(self.filepath)

    def __enter__(self):
        self.handler = open(self.filepath, "rb")
        self.raf = mmap.mmap(self.handler.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        self.raf.close()
        self.handler.close()

    def get_volume_raw(self, index: int) -> npt.NDArray:
        return self.header.get_volume(self.raf, index)

    def get_imagej_volume(self, index: int, conversion=None):

        # CZYX
        stack = self.get_volume_raw(index)
        # CZYX -> TCZYX
        stack = stack[np.newaxis, :, :, :, :]
        # TCZYX -> TZCYX
        stack = stack.transpose((0, 2, 1, 3, 4))

        if conversion is None:
            return stack

        # type conversion
        lut = LUT_TABLE.get(conversion)
        if lut is None:
            warn(
                UserWarning(
                    f"Unsupported conversion: {conversion}, output the original format (uint16)"
                )
            )
            return stack

        return lut[stack]

    def save_volume_to_tiff(self, path: Path, index: int, conversion=None):

        stack = self.get_imagej_volume(index, conversion)
        dtype = stack.dtype
        _, C, Z, Y, X = self.header.shape
        tifffile.imwrite(
            path,
            stack,
            imagej=True,
            shape=(1, Z, C, Y, X),
            dtype=dtype,
            resolution=(1.0 / self.header.x_scale, 1.0 / self.header.y_scale),
            metadata={
                "spacing": self.header.z_scale,
                "unit": "um",
                "axes": "TZCYX",
            },
            photometric="minisblack",
        )

    def save_all_volume_to_tiff(self, path, conversion=None):
        dtype = conversion or "u2"
        T, C, Z, Y, X = self.header.shape

        frames = (
            self.get_imagej_volume(idx, conversion)
            for idx in range(self.header.n_frame)
        )
        tifffile.imwrite(
            path,
            frames,
            imagej=True,
            resolution=(1.0 / self.header.x_scale, 1.0 / self.header.y_scale),
            metadata={
                "spacing": self.header.z_scale,
                "unit": "um",
                "axes": "TZCYX",
            },
            photometric="minisblack",
            shape=(T, Z, C, Y, X),
            dtype=dtype,
        )
