#!/usr/bin/env python3
"""Diagnostic: show infield/ring separation for a sweep of opening radii."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import binary_dilation, binary_opening, disk

import centerline_extractor as ce
import process_real_map as prm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map-yaml", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--radii", type=int, nargs="+", default=[35, 45, 55, 70])
    args = p.parse_args()

    data = ce.load_map_yaml(args.map_yaml)
    data = prm.clean_grid(data, 40, 2)
    free = data.grid == 0
    interior = ce._interior_free_mask(free)
    blob = prm._largest_component(interior)

    fig, ax = plt.subplots(1, len(args.radii), figsize=(6 * len(args.radii), 6))
    for k, R in enumerate(args.radii):
        infield = binary_dilation(binary_opening(blob, disk(R)), disk(3))
        ring = prm._largest_component(blob & ~infield)
        img = np.zeros(blob.shape)
        img[blob] = 1
        img[infield] = 2
        img[ring] = 3
        ax[k].imshow(img, origin="lower", cmap="viridis")
        n = ndimage.label(blob & ~infield)[1]
        ax[k].set_title(f"R={R}px  ring_px={int(np.count_nonzero(ring))}  frags={n}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
