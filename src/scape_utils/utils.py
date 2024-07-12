import itertools
import mmap
import struct
from io import BufferedReader
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union
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
    def shape(self) -> Tuple[int, int, int, int, int]:
        # TCZYX
        return (self.n_frame, self.n_channel, self.depth, self.height, self.width)

    @property
    def scales(self) -> Tuple[float, float, float]:
        # z, y, x
        return (self.z_scale, self.y_scale, self.x_scale)


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

    def get_volume(
        self,
        index: int,
        conversion: Optional[str] = None,
        imagej: bool = False,
    ) -> npt.NDArray:

        T, C, Z, Y, X = self.header.shape

        bytes_per_volume = self.header.bytes_per_volume
        pixels_per_volume = self.header.pixels_per_volume

        if (index < 0) | (index >= T):
            raise IndexError(f"Index is out of boundary: {index}")

        gap = 16
        offset = 52 + index * bytes_per_volume
        raw = self.raf[offset + gap : offset + bytes_per_volume]

        # newbyteorder is not inplace reassign is required.
        dt = np.dtype("u2").newbyteorder(">")

        stack = np.frombuffer(
            raw,
            dtype=dt,
            count=pixels_per_volume,
        ).reshape((C, Z, Y, X))

        # CZYX -> TCZYX
        stack = stack[np.newaxis, :, :, :, :]

        if imagej:
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

    def get_multi_volumes(
        self,
        start: int,
        end: int,
        conversion: Optional[str] = None,
        imagej: bool = False,
    ) -> npt.NDArray:
        T, C, Z, Y, X = self.header.shape
        bytes_per_volume = self.header.bytes_per_volume

        if (start < 0) | (start >= T):
            raise IndexError(f"Index is out of boundary: start={start}")

        if (end < 0) | (end >= T):
            raise IndexError(f"Index is out of boundary: end={end}")

        if start > end:
            start, end = end, start

        T = end - start + 1

        gap = 16
        offset = 52 + start * bytes_per_volume

        raw = self.raf[offset : offset + bytes_per_volume * T]

        # newbyteorder is not inplace reassign is required.
        dt = np.dtype("u2").newbyteorder(">")

        stack = (
            np.frombuffer(
                raw,
                dtype=dt,
            )
            .reshape((T, -1))[:, gap // 2 :]
            .reshape((T, C, Z, Y, X))
        )

        if imagej:
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
        stack = self.get_volume(index, conversion=conversion, imagej=True)
        dtype = stack.dtype
        z_scale, y_scale, x_scale = self.header.scales
        tifffile.imwrite(
            path,
            stack,
            imagej=True,
            shape=stack.shape,
            dtype=dtype,
            resolution=(1.0 / x_scale, 1.0 / y_scale),
            metadata={
                "spacing": z_scale,
                "unit": "um",
                "axes": "TZCYX",
            },
            photometric="minisblack",
        )

    def save_all_volumes_to_tiff(self, path, conversion=None, chunksize=10):
        dtype = conversion or "u2"
        T, C, Z, Y, X = self.header.shape

        def frames():
            for chunk in itertools.batched(range(T), chunksize):
                start = min(chunk)
                end = max(chunk)
                stacks = self.get_multi_volumes(start, end, conversion, imagej=True)
                stacks = np.expand_dims(stacks, 0)
                for stack in stacks:
                    yield stack

        tifffile.imwrite(
            path,
            frames(),
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
