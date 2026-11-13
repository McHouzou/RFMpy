# RFMpy

> **GPU acceleration:** RFMpy now includes a separate JAX GPU implementation in `gpu_rfm_jax_optimal.py`. It uses JAX JIT compilation, float32 device arrays, matrix-free conjugate-gradient solves, and GPU-parallel bond/failure calculations. The CPU and GPU scripts are separate implementations with different defaults and capabilities; the CPU script does not switch to GPU execution through a command-line flag.

RFMpy is a Python simulator for quasi-static random fuse-model (RFM) fracture on a two-dimensional lattice. It progressively increases the applied voltage, solves the lattice voltages, breaks overloaded bonds, and follows the resulting failure avalanches until the network can no longer carry meaningful current.

The repository includes:

- `cpu_rfm_triangular.py` — simulation engine and command-line interface.
- `gpu_rfm_jax_optimal.py` — JAX GPU implementation optimized for a small square lattice and GPU execution.
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

For GPU execution, install JAX using the variant that matches the CUDA version and platform on the machine. For example, a CUDA 12 environment can typically use:

```bash
python -m pip install -U "jax[cuda12]"
```

Check the current [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) for the supported CUDA wheel, driver, and accelerator combination. Verify that JAX can see the GPU before starting a long run:

```bash
python -c "import jax; print(jax.devices())"
```

The GPU script requests the GPU platform at import time and will fail rather than silently provide a CPU fallback when a compatible GPU runtime is unavailable.

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

## Run the GPU implementation

The GPU backend is launched separately and accepts an optional run number:

```bash
python gpu_rfm_jax_optimal.py [run_no]
```

Example:

```bash
python gpu_rfm_jax_optimal.py 7
```

The script reports the selected JAX device and uses:

- A square `16 × 16` lattice (`L=16` in the source).
- Float32 JAX arrays and high matmul precision.
- A matrix-free conjugate-gradient solve for free-node voltages.
- JAX-jitted bond-current, failure-detection, and row-current calculations.
- GPU-side random Weibull thresholds with shape `4.0` and scale `1.0`.
- A two-cell pre-crack and up to 10 primary fracture events by default.

These settings are currently constants in `gpu_rfm_jax_optimal.py`; the GPU script does not expose the CPU script's lattice, topology, disorder, crack-size, or output-directory arguments. Its results are written to `Output/` and can be visualized with the same visualization script when the expected files are present.

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

The GPU script currently writes a subset of the CPU output set:

| File | GPU availability |
| --- | --- |
| `burned_bond_positions_N.txt` | Written by the GPU script. |
| `avalanche_history_N.txt` | Written by the GPU script. |
| `voltage_current_N.txt` | Written by the GPU script. |
| `burned_bond_order_N.txt` | Not currently written by the GPU script. |
| `precrack_*_N.txt` | Not currently written by the GPU script. |
| `final_voltage_N.txt` | Not currently written by the GPU script. |
| `failure_thresholds_N.txt` | Not currently written by the GPU script. |

## Performance notes

The solver is optimized for CPU execution:

- Sparse connectivity matrices are represented with SciPy CSR matrices.
- The free-node voltage system is solved with SciPy's conjugate-gradient solver.
- Bond-current and failure checks use Numba `@njit` CPU kernels.
- Frequently reused sparse submatrices are cached during a solve sequence.
- Row-current calculations use vectorized NumPy operations.

The script sets `MKL_NUM_THREADS=0` and `NUMEXPR_NUM_THREADS=0` so linked numerical libraries can choose their available CPU parallelism. These settings are applied by the simulation script before the numerical work begins.

The GPU implementation uses a different optimization strategy:

- JAX compiles numerical kernels with `jit` for the selected accelerator.
- The voltage system is solved with a matrix-free Laplacian matvec, avoiding repeated sparse-matrix conversion.
- Bond currents, failure masks, reductions, and row-current observables are computed with JAX device operations.
- Conductance and burned-bond state are updated as device arrays.
- The implementation is tuned for float32 throughput and explicitly requests the GPU platform.

## Limitations and reproducibility

- The lattice dimensions are currently module-level constants in `cpu_rfm_triangular.py` (`L=64` by default), rather than command-line options.
- The visualizer's `--lattice-size` must match the data that was generated.
- The solver uses a fixed voltage boundary on the first and last lattice rows.
- The simulation currently caps the primary fracture loop at 20,000 events.
- Random thresholds are seeded from `run_no`; changing the run number changes the disorder realization.
- The CPU and GPU implementations do not currently share one configurable API.
- The GPU implementation currently has fixed lattice size, square topology, threshold settings, pre-crack size, fracture-event limit, and output directory.
- GPU output is currently a subset of CPU output, so some visualizer features that depend on threshold or final-voltage files are unavailable for GPU runs.
- The repository does not currently include automated tests or packaging metadata.

## License

RFMpy is released under the MIT License. See [`LICENSE`](LICENSE).
