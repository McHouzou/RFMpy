#example usage: python sim_visualize.py 0 Output --lattice-size 128 --lattice-type triangular --periodic-vertical --save

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def idx(i: int, j: int, lattice_size: int) -> int:
    return i * lattice_size + j


def wrap_j(j: int, lattice_size: int) -> int:
    return j % lattice_size


def get_node_neighbors(i: int, j: int, lattice_size: int, lattice_type: str = "square", periodic_vertical: bool = False):
    lattice = lattice_type.lower()

    if lattice == "square":
        candidates = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    elif lattice == "triangular":
        if (i % 2) == 0:
            candidates = [
                (i, j - 1), (i, j + 1),
                (i - 1, j - 1), (i - 1, j),
                (i + 1, j - 1), (i + 1, j),
            ]
        else:
            candidates = [
                (i, j - 1), (i, j + 1),
                (i - 1, j), (i - 1, j + 1),
                (i + 1, j), (i + 1, j + 1),
            ]
    else:
        raise ValueError("lattice_type must be 'square' or 'triangular'")

    neighbors = []
    for ni, nj in candidates:
        if not (0 <= ni < lattice_size):
            continue
        if periodic_vertical:
            nj = wrap_j(nj, lattice_size)
        elif not (0 <= nj < lattice_size):
            continue
        neighbors.append((ni, nj))

    return neighbors


def build_bond_map(lattice_size: int, lattice_type: str = "square", periodic_vertical: bool = False):
    bond_map = {}
    bond_index = 0

    for i in range(lattice_size):
        for j in range(lattice_size):
            p = idx(i, j, lattice_size)
            for ni, nj in get_node_neighbors(i, j, lattice_size, lattice_type=lattice_type, periodic_vertical=periodic_vertical):
                q = idx(ni, nj, lattice_size)
                bond = (min(p, q), max(p, q))
                if bond not in bond_map:
                    bond_map[bond] = bond_index
                    bond_index += 1

    return bond_map


def get_node_coordinates(lattice_size: int, lattice_type: str = "square"):
    x = np.zeros(lattice_size * lattice_size, dtype=np.float64)
    y = np.zeros(lattice_size * lattice_size, dtype=np.float64)
    tri_y_scale = np.sqrt(3.0) / 2.0

    for i in range(lattice_size):
        for j in range(lattice_size):
            p = idx(i, j, lattice_size)
            if lattice_type.lower() == "triangular":
                x[p] = j + 0.5 * (i % 2)
                y[p] = i * tri_y_scale
            else:
                x[p] = j
                y[p] = i

    return x, y


def load_array(path: Path, *, min_dims: int = 1):
    if not path.exists():
        raise FileNotFoundError(f"Missing required output file: {path}")
    array = np.loadtxt(path)
    array = np.asarray(array)
    while array.ndim < min_dims:
        array = np.expand_dims(array, axis=0)
    return array


def load_simulation_outputs(output_dir: Path, run_no: int):
    precrack_order_path = output_dir / f"precrack_bond_order_{run_no}.txt"
    burned_order_path = output_dir / f"burned_bond_order_{run_no}.txt"
    precrack_path = output_dir / f"precrack_bond_positions_{run_no}.txt"
    burned_path = output_dir / f"burned_bond_positions_{run_no}.txt"
    avalanche_path = output_dir / f"avalanche_history_{run_no}.txt"
    voltage_current_path = output_dir / f"voltage_current_{run_no}.txt"
    final_voltage_path = output_dir / f"final_voltage_{run_no}.txt"
    thresholds_path = output_dir / f"failure_thresholds_{run_no}.txt"

    precrack_order = load_array(precrack_order_path, min_dims=1).astype(np.int32) if precrack_order_path.exists() else None
    burned_order = load_array(burned_order_path, min_dims=1).astype(np.int32) if burned_order_path.exists() else None
    precrack_positions = load_array(precrack_path, min_dims=2) if precrack_path.exists() else None
    burned_positions = load_array(burned_path, min_dims=2)
    avalanche_history = load_array(avalanche_path, min_dims=1).astype(np.int32)
    voltage_current = load_array(voltage_current_path, min_dims=2)
    final_voltage = load_array(final_voltage_path, min_dims=1) if final_voltage_path.exists() else None
    failure_thresholds = load_array(thresholds_path, min_dims=1) if thresholds_path.exists() else None

    if burned_positions.shape[1] != 2:
        raise ValueError(f"Expected two columns in {burned_path}, got shape {burned_positions.shape}")
    if voltage_current.shape[1] != 2:
        raise ValueError(f"Expected two columns in {voltage_current_path}, got shape {voltage_current.shape}")

    return precrack_order, burned_order, precrack_positions, burned_positions, avalanche_history, voltage_current, final_voltage, failure_thresholds


def infer_lattice_settings(lattice_size: int, bond_count: int, fallback_lattice_type: str, fallback_periodic_vertical: bool):
    candidates = [
        ("square", False),
        ("square", True),
        ("triangular", False),
        ("triangular", True),
    ]

    for lattice_type, periodic_vertical in candidates:
        if len(build_bond_map(lattice_size, lattice_type=lattice_type, periodic_vertical=periodic_vertical)) == bond_count:
            return lattice_type, periodic_vertical

    return fallback_lattice_type, fallback_periodic_vertical


def _rounded_position_keys(points: np.ndarray, decimals: int = 3):
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals=decimals)
    return {tuple(row) for row in rounded}


def build_domain_plot(
    output_dir: Path,
    run_no: int,
    lattice_size: int = 128,
    lattice_type: str = "triangular",
    periodic_vertical: bool = False,
    save_fig: bool = False,
):
    precrack_order, burned_order, precrack_positions, burned_positions, avalanche_history, voltage_current, final_voltage, failure_thresholds = load_simulation_outputs(output_dir, run_no)

    if failure_thresholds is not None:
        lattice_type, periodic_vertical = infer_lattice_settings(
            lattice_size,
            len(failure_thresholds),
            lattice_type,
            periodic_vertical,
        )

    bond_map = build_bond_map(lattice_size, lattice_type=lattice_type, periodic_vertical=periodic_vertical)
    node_x, node_y = get_node_coordinates(lattice_size, lattice_type=lattice_type)

    bond_pairs = np.array(list(bond_map.keys()), dtype=np.int32)
    p_all = bond_pairs[:, 0]
    q_all = bond_pairs[:, 1]

    x0 = node_x[p_all]
    y0 = node_y[p_all]
    x1 = node_x[q_all]
    y1 = node_y[q_all]

    segments = np.stack(
        [
            np.stack([x0, y0], axis=1),
            np.stack([x1, y1], axis=1),
        ],
        axis=1,
    )

    visible_mask = np.ones(len(segments), dtype=bool)
    if periodic_vertical:
        pj_all = p_all % lattice_size
        qj_all = q_all % lattice_size
        dj = np.abs(pj_all - qj_all)
        wrapped_plot_bonds = (lattice_size - dj) < dj
        visible_mask = ~wrapped_plot_bonds

    if precrack_order is not None and precrack_order.size:
        precrack_visible_mask = np.zeros(len(segments), dtype=bool)
        precrack_visible_mask[np.asarray(precrack_order, dtype=np.int32)] = True
        precrack_visible_mask &= visible_mask
    elif precrack_positions is not None:
        bond_midpoints = np.column_stack(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        precrack_keys = _rounded_position_keys(precrack_positions)
        precrack_visible_mask = np.array(
            [tuple(np.round(midpoint, 3)) in precrack_keys for midpoint in bond_midpoints], dtype=bool
        ) & visible_mask
    else:
        precrack_visible_mask = np.zeros(len(segments), dtype=bool)

    if burned_order is not None and burned_order.size:
        burned_visible_mask = np.zeros(len(segments), dtype=bool)
        burned_visible_mask[np.asarray(burned_order, dtype=np.int32)] = True
        burned_visible_mask &= visible_mask
    else:
        bond_midpoints = np.column_stack(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        burned_keys = _rounded_position_keys(burned_positions)
        burned_visible_mask = np.array(
            [tuple(np.round(midpoint, 3)) in burned_keys for midpoint in bond_midpoints], dtype=bool
        ) & visible_mask
    active_mask = (~burned_visible_mask) & visible_mask

    vc = np.asarray(voltage_current, dtype=np.float64)
    voltage_history = vc[:, 0]
    current_history = vc[:, 1]

    color_active_bonds = False

    output_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, stem):
        if save_fig:
            fig.savefig(output_dir / f"{stem}_{run_no}.png", bbox_inches="tight", transparent=True, dpi=1000)
            #fig.savefig(output_dir / f"{stem}_{run_no}.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    if np.any(active_mask):
        if color_active_bonds and final_voltage is not None and failure_thresholds is not None and len(failure_thresholds) == len(segments):
            bond_current = np.abs(final_voltage[p_all] - final_voltage[q_all]) / (failure_thresholds + 1e-12)
            active_collection = LineCollection(
                segments[active_mask],
                cmap="Reds",
                linewidths=0.9,
                zorder=1,
            )
            active_collection.set_clim(vmin=0.0, vmax=float(np.max(bond_current[active_mask]) if np.any(active_mask) else 1.0))
            active_collection.set_array(bond_current[active_mask])
            ax.add_collection(active_collection)
        else:
            ax.add_collection(
                LineCollection(
                    segments[active_mask],
                    colors="gray",
                    linewidths=1.0,
                    zorder=1,
                )
            )
    if np.any(precrack_visible_mask):
        ax.add_collection(
            LineCollection(
                segments[precrack_visible_mask],
                colors="white",
                linewidths=1.0,
                zorder=2,
            )
        )
    if np.any(burned_visible_mask):
        ax.add_collection(
            LineCollection(
                segments[burned_visible_mask],
                colors="red",
                linewidths=1.5,
                zorder=3,
            )
        )
    ax.set_xlim(float(np.min(node_x) - 0.75), float(np.max(node_x) + 0.75))
    ax.set_ylim(float(np.min(node_y) - 0.75), float(np.max(node_y) + 0.75))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title("Fracture domain")
    save_figure(fig, "domain_plot")

    
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(voltage_history, current_history, color="black", lw=1.2, marker="o", ms=2)
    ax.set_xlabel("Voltage")
    ax.set_ylabel("Current")
    ax.set_title("I-V curve")
    save_figure(fig, "iv_curve")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bins = min(24, max(8, int(np.sqrt(len(avalanche_history)))))
    ax.hist(avalanche_history, bins=bins, color="0.3", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Avalanche size")
    ax.set_ylabel("Count")
    ax.set_title("Avalanche histogram")
    ax.set_yscale("log")
    save_figure(fig, "avalanche_histogram")
    
    

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RFM output files.")
    parser.add_argument("run_no", nargs="?", type=int, default=0, help="Run number used in the output file names.")
    parser.add_argument("output_dir", nargs="?", default="Output", help="Directory containing the saved text files.")
    parser.add_argument("--lattice-size", type=int, default=128, help="Lattice size used in the simulation.")
    parser.add_argument("--lattice-type", choices=("square", "triangular"), default="triangular", help="Simulation lattice type.")
    parser.add_argument("--periodic-vertical", action="store_true", help="Use periodic vertical boundary handling when reconstructing the lattice.")
    parser.add_argument("--save", action="store_true", help="Save PNG/PDF outputs in addition to displaying the figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_domain_plot(
        output_dir=Path(args.output_dir),
        run_no=args.run_no,
        lattice_size=args.lattice_size,
        lattice_type=args.lattice_type,
        periodic_vertical=args.periodic_vertical,
        save_fig=args.save,
    )


if __name__ == "__main__":
    main()