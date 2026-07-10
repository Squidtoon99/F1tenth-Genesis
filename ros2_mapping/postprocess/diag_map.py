#!/usr/bin/env python3
"""Diagnostic: visualize cleaned occupancy + interior-free connected components."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, disk

import centerline_extractor as ce


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map-yaml", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-wall-px", type=int, default=40)
    p.add_argument("--close-radius", type=int, default=2)
    args = p.parse_args()

    data = ce.load_map_yaml(args.map_yaml)
    occ = data.grid == 1
    labeled, n = ndimage.label(occ)
    sizes = ndimage.sum(occ, labeled, range(1, n + 1)) if n else []
    keep = {lab for lab, sz in enumerate(sizes, start=1) if sz >= args.min_wall_px}
    occ_dn = np.isin(labeled, list(keep)) if keep else np.zeros_like(occ)
    occ_cl = binary_closing(occ_dn, disk(args.close_radius)) if args.close_radius else occ_dn

    grid2 = data.grid.copy()
    grid2[(data.grid == 1) & ~occ_cl] = 2
    grid2[occ_cl] = 1
    free = grid2 == 0
    interior = ce._interior_free_mask(free)

    ilab, ni = ndimage.label(interior)
    isizes = ndimage.sum(interior, ilab, range(1, ni + 1)) if ni else []
    order = np.argsort(isizes)[::-1] if ni else []
    print(f"occupied comps: raw={n}, kept={len(keep)}")
    print(f"interior-free comps: {ni}")
    for rank, idx in enumerate(order[:6]):
        lab = idx + 1
        per = ce._component_perimeter_m(ilab == lab, data.resolution)
        print(f"  comp {lab}: area={int(isizes[idx])}px  perimeter={per:.1f} m")

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].imshow(occ, origin="lower", cmap="Greys")
    ax[0].set_title(f"raw occupied ({n} comps)")
    ax[1].imshow(occ_cl, origin="lower", cmap="Greys")
    ax[1].set_title(f"cleaned occupied (kept {len(keep)}, close r={args.close_radius})")
    comp_img = np.zeros(interior.shape)
    for rank, idx in enumerate(order[:8]):
        comp_img[ilab == (idx + 1)] = rank + 1
    ax[2].imshow(comp_img, origin="lower", cmap="tab10")
    ax[2].set_title(f"interior-free comps ({ni}) by size")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
