import mmap
import os
from io import BufferedReader
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Optional, Tuple, Union

import attrs
import numpy as np
from numpy.typing import NDArray

LUT_TABLE: Dict[str, NDArray]

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
    def from_3DU16(cls, filename: os.PathLike | str) -> "ScapeImageDecoder": ...
    @property
    def bytes_per_xy(self) -> int: ...
    @property
    def bytes_per_xyz(self) -> int: ...
    @property
    def bytes_per_volume(self) -> int: ...
    @property
    def pixels_per_volume(self) -> int: ...
    @property
    def shape(self) -> Tuple[int, int, int, int, int]: ...
    @property
    def scales(self) -> Tuple[float, float, float]: ...

@attrs.define
class ScapeVirtualStack:
    filepath: Path
    header: ScapeImageDecoder = attrs.field(init=False)

    raf: mmap.mmap = attrs.field(init=False, default=None, repr=False)
    handler: BufferedReader = attrs.field(init=False, default=None, repr=False)

    def __attrs_post_init__(self): ...
    def __enter__(self): ...
    def __exit__(self, *args): ...
    def get_volume(
        self,
        index: int,
        conversion: Optional[Literal["u8", "f32", "u16"]] = None,
        imagej: bool = False,
    ) -> Union[NDArray[np.uint8], NDArray[np.uint16], NDArray[np.float32]]: ...
    def get_multi_volumes(
        self,
        start: int,
        end: int,
        conversion: Optional[Literal["u8", "f32", "u16"]] = None,
        imagej: bool = False,
    ) -> Union[NDArray[np.uint8], NDArray[np.uint16], NDArray[np.float32]]: ...
    def save_volume_to_tiff(
        self,
        filename: os.PathLike | str,
        index: int,
        conversion: Optional[Literal["u8", "f32", "u16"]] = None,
    ) -> None: ...
    def save_all_volumes_to_tiff(
        self,
        filename: os.PathLike | str,
        conversion: Optional[Literal["u8", "f32", "u16"]] = None,
        chunksize: int = 10,
    ) -> None: ...
    def save_all_volumes_to_hdf(
        self,
        filename: os.PathLike | str,
        conversion: Optional[Literal["u8", "f32", "u16"]] = None,
        chunksize: int = 10,
    ) -> None: ...
