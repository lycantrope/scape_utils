# scape-utils

# Usage
```python
from scape_utils import ScapeVirtualStack
import numpy as np
from pathlib import Path

file_path = Path("path/to/your/file.3DU16")

with ScapeVirtualStack(file_path) as stack:
    # get 1st volume with format (TCZYX)
    volume_0 = stack.get_volume_raw(0)
    # get 1st volume as imagej format (TZCYX)
    volume_ij = stack.get_imagej_volume(2)
```
