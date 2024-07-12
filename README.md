# Scape Utils

`scape_utils` is a Python module designed for handling `.3DU16` files, providing tools for reading and manipulating volumetric data. The module includes two main classes: `ScapeVirtualStack` and `ScapeImageDecoder`.

## Installation

To install `scape_utils`, use pip:

```bash
pip install scape_utils
```

## Usage

To read a `.3DU16` file and manipulate its data, use the `ScapeVirtualStack` class. Below are examples of how to get raw volumes and ImageJ formatted volumes.

```python
from scape_utils import ScapeVirtualStack
import numpy as np
from pathlib import Path

file_path = Path("path/to/your/file.3DU16")

with ScapeVirtualStack(file_path) as stack:
    # show information
    print(stack)

    # Get the 1st volume in format (TCZYX)
    volume_0 = stack.get_volume(0)
    
    # Get the 3rd volume in ImageJ format (TZCYX)
    volume_ij = stack.get_volume(2, imagej=True)

    # Get multiple stacks at once (from t0 to t10)
    volume_multi = stack.get_multi_volumes(0, 10)    
```

## License

This project is licensed under the MIT License.
