# RFMpy

> **GPU status:** The current implementation is **CPU-only**. It uses SciPy sparse linear algebra and Numba's CPU JIT compilation; there is no CUDA, CuPy, Numba-CUDA, or GPU backend to enable at runtime. Installing CUDA or setting a GPU flag will not activate GPU execution. GPU support would require a separate solver and kernel implementation.

RFMpy is a Python simulator for quasi-static random fuse-model (RFM) fracture on a two-dimensional lattice. It progressively increases the applied voltage, solves the lattice voltages, breaks overloaded bonds, and follows the resulting failure avalanches until the network can no longer carry meaningful current.

The repository includes:

- `cpu_rfm_triangular.py` — simulation engine and command-line interface.
- `sim_visualize.py` — reconstruction and visualization of saved simulation results.

## What the simulation does

The model represents a resistor/fuse network whose bonds have random failure thresholds:

1. Construct a square or triangular lattice of nodes and nearest-neighbor bonds.
2. Optionally wrap the lattice horizontally using periodic boundary handling.
3. Assign each bond a random threshold using a reproducible seed.
4. Apply fixed voltage to the top and bottom rows.
5. Solve the sparse Kirchhoff system for node voltages.
6. Compute the absolute voltage difference across every active bond.
7. Increase the applied voltage until one or more bonds exceed their thresholds.
8. Burn the primary failed bond, rebuild the network matrix, and solve again.
9. Burn any additional overloaded bonds as an avalanche.
10. Record the voltage/current history, bond failure order, avalanche sizes, and final voltage field.

The default lattice is `64 × 64` nodes. The default topology is triangular, the default threshold distribution is Weibull, and the default disorder parameter is `D=0.5`.

### Lattice topologies

- **Square:** four nearest neighbors per interior node.
- **Triangular:** six nearest neighbors per interior node, with alternating row offsets.

The simulator can use ordinary horizontal boundaries or periodic horizontal wrapping. The visualization hides wrapped bonds that would otherwise be drawn across the figure.

### Pre-crack

`crack_size_cells` removes a horizontal segment of bonds across the middle row interface before the simulation starts. It must be an even number smaller than the lattice size. Use `0` to disable the pre-crack.

### Threshold distributions

- `weibull`: Weibull-distributed thresholds, with shape controlled by `D`.
- `powerlaw_unit`: thresholds generated from the unit-interval power-law inverse transform `I = U^D`.

The run number is used as the random seed, so repeating a run with the same model parameters and run number reproduces the same threshold sequence.

## Installation

Python 3.9+ is recommended. Install the runtime dependencies:

```bash
python -m pip install numpy scipy numba matplotlib
```

No GPU-specific package is required for the current implementation.

## Run a simulation

The positional command-line arguments are:

```text
python cpu_rfm_triangular.py \
    [run_no] [disorder_param] [output_dir] \
    [lattice_type] [periodic_vertical] [crack_size_cells] \
    [failure_distribution]
```

Example using the defaults:

```bash
python cpu_rfm_triangular.py
```

Example with a triangular lattice, horizontal periodic wrapping, a pre-crack, and power-law thresholds:

```bash
python cpu_rfm_triangular.py \
    7 0.5 Output triangular true 16 powerlaw_unit
```

Boolean values accepted by `periodic_vertical` include `true`, `false`, `yes`, `no`, `1`, and `0`.

The simulation prints progress for each primary fracture and avalanche event. Large lattices or long fracture sequences can take substantial CPU time.

## Visualize a completed run

After a simulation has written its files to `Output`, reconstruct the lattice and display the figures with:

```bash
python sim_visualize.py 7 Output \
    --lattice-size 64 \
    --lattice-type triangular \
    --periodic-vertical \
    --save
```

Visualization options:

- `--lattice-size` — node count along each lattice dimension. Use the same size used to generate the data.
- `--lattice-type` — `square` or `triangular`.
- `--periodic-vertical` — reconstruct horizontal periodic wrapping.
- `--save` — write PNG figures to the output directory in addition to displaying them.

When threshold data is available, the visualizer also uses the number of bonds to infer the lattice topology and periodic setting when possible. Explicit command-line settings remain the fallback.

## Visualizations

The visualizer produces three views:

### Fracture domain

The lattice is drawn as a bond network:

- Gray lines are active bonds.
- Red lines are bonds burned during the simulation.
- White lines identify the initial pre-crack.

Wrapped periodic bonds are omitted from the drawing when they would create long lines across the plot. The axes are hidden and the aspect ratio is fixed so the lattice geometry remains readable.

### I–V curve

The voltage-current plot shows the recorded applied voltage against the measured current crossing the central row boundary. The curve captures the progressive degradation of network conductance as bonds fail.

### Avalanche histogram

The histogram shows how often each avalanche size occurs. Its y-axis is logarithmic, which helps expose the distribution when small avalanches are much more common than large events.

With `--save`, the figures are written as:

```text
Output/domain_plot_<run_no>.png
Output/iv_curve_<run_no>.png
Output/avalanche_histogram_<run_no>.png
```

## Output files

For a run number `N`, the simulator writes:

| File | Contents |
| --- | --- |
| `precrack_bond_order_N.txt` | Bond indices removed to create the pre-crack. |
| `precrack_bond_positions_N.txt` | Midpoint coordinates of pre-crack bonds. |
| `burned_bond_order_N.txt` | Bond indices in the order they failed, including avalanche failures. |
| `burned_bond_positions_N.txt` | Midpoint coordinates of failed bonds. |
| `avalanche_history_N.txt` | Number of failures recorded for each fracture event. |
| `voltage_current_N.txt` | Applied voltage and measured current history. |
| `final_voltage_N.txt` | Final voltage at every node. |
| `failure_thresholds_N.txt` | Failure threshold assigned to every bond. |

These text files make it possible to analyze the run independently with NumPy, pandas, or another scientific workflow.

## Performance notes

The solver is optimized for CPU execution:

- Sparse connectivity matrices are represented with SciPy CSR matrices.
- The free-node voltage system is solved with SciPy's conjugate-gradient solver.
- Bond-current and failure checks use Numba `@njit` CPU kernels.
- Frequently reused sparse submatrices are cached during a solve sequence.
- Row-current calculations use vectorized NumPy operations.

The script sets `MKL_NUM_THREADS=0` and `NUMEXPR_NUM_THREADS=0` so linked numerical libraries can choose their available CPU parallelism. These settings are applied by the simulation script before the numerical work begins.

## Limitations and reproducibility

- The lattice dimensions are currently module-level constants in `cpu_rfm_triangular.py` (`L=64` by default), rather than command-line options.
- The visualizer's `--lattice-size` must match the data that was generated.
- The solver uses a fixed voltage boundary on the first and last lattice rows.
- The simulation currently caps the primary fracture loop at 20,000 events.
- Random thresholds are seeded from `run_no`; changing the run number changes the disorder realization.
- The repository does not currently include automated tests, packaging metadata, or a GPU execution path.

## License

RFMpy is released under the MIT License. See [`LICENSE`](LICENSE).
