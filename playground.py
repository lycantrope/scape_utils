from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import tifffile
from skimage import measure

DEFAULT_DATAPATH = Path(r"c:\Users\kuan\Desktop\WT_binary.tif")


def main():
    print(DEFAULT_DATAPATH)
    src = tifffile.memmap(DEFAULT_DATAPATH)
    labels = measure.label(src)

    regions = [
        max(
            (roi for roi in measure.regionprops(label, intensity_image=raw)),
            key=lambda x: x.area,
        )
        for label, raw in zip(labels, src)
    ]

    nrows = 5
    ncols = 4
    step = 24

    nsample = (nrows - 1) * ncols

    fig = plt.figure()
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        wspace=0.08,
        hspace=0.08,
    )

    axs = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nsample)]
    data = []
    for ax, idx in zip(axs, range(0, len(regions), step)):
        roi = regions[idx]
        y1, x1, y2, x2 = roi.bbox
        mask = labels[idx, y1:y2, x1:x2].copy()
        mask = mask == roi.label
        im = src[idx, y1:y2, x1:x2].copy()
        im[~mask] = 0
        ax.imshow(im)
        ax.set_title(f"T:{idx:d}")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        data.append(roi.moments_hu)

    data = np.array(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    np.linalg.norm()
    data = (data - mean) / std

    ax_last = fig.add_subplot(gs[-1, :])
    ax_last.plot(data)
    ax_last.set_xticks(np.arange(len(data)))
    ax_last.set_xticklabels([f"{i * step:d}" for i in range(len(data))])

    ax_last.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    plt.show()


if __name__ == "__main__":
    main()
