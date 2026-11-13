import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import jit, vmap
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg

# GPU optimization flags for A100
os.environ['JAX_PLATFORMS'] = 'gpu'
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')

# Enable high-performance cuBLAS on A100
jax.config.update('jax_enable_x64', False)  # Use float32 for 2x A100 throughput
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'highest')

# Cache for sparse matrix conversions and device arrays
_sparse_matrix_cache = {}
_gpu_device_cache = {}

# Parameters
L = 16
N = L * L
h_step = 1.0
LOG_EVERY_PRIMARY = 25

def idx(i, j):
    return i * L + j

# ====================================================================
# MATRIX / BONDS / THRESHOLDS
# ====================================================================

def build_connectivity_matrix_gpu():
    """Build connectivity matrix optimized for GPU (pre-allocate full size)."""
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rows, cols, vals = [], [], []

    for i in range(L):
        for j in range(L):
            p = idx(i, j)
            diag = 0
            for dx, dy in stencil:
                ni, nj = i + dx, j + dy
                if 0 <= ni < L and 0 <= nj < L:
                    q = idx(ni, nj)
                    rows.append(p)
                    cols.append(q)
                    vals.append(-1.0)
                    diag += 1
            rows.append(p)
            cols.append(p)
            vals.append(float(diag))

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


def generate_failure_thresholds(shape_param=2.0, scale_param=1.0, seed=42):
    """Generate Weibull thresholds using GPU random."""
    key = jax.random.PRNGKey(seed)

    n_bonds = 0
    bond_map = {}
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i in range(L):
        for j in range(L):
            p = idx(i, j)
            for dx, dy in stencil:
                ni, nj = i + dx, j + dy
                if 0 <= ni < L and 0 <= nj < L:
                    q = idx(ni, nj)
                    bond = (min(p, q), max(p, q))
                    if bond not in bond_map:
                        bond_map[bond] = n_bonds
                        n_bonds += 1

    u = jax.random.uniform(key, shape=(n_bonds,), dtype=jnp.float32)
    thresholds = scale_param * (-jnp.log(1 - u)) ** (1 / shape_param)

    return bond_map, thresholds


def bond_map_to_arrays(bond_map, n_bonds):
    """Convert bond map to GPU arrays."""
    p_list = np.zeros(n_bonds, dtype=np.int32)
    q_list = np.zeros(n_bonds, dtype=np.int32)
    for bond, bidx in bond_map.items():
        p, q = bond
        p_list[bidx] = p
        q_list[bidx] = q
    return jnp.asarray(p_list, dtype=jnp.int32), jnp.asarray(q_list, dtype=jnp.int32)


def create_precrack(bond_map, crack_size_cells):
    """Create pre-cracked bonds."""
    if crack_size_cells % 2 != 0 or crack_size_cells >= L:
        raise ValueError("crack_size_cells must be an even number less than L")

    precrack_bonds = []
    middle_i = L // 2
    crack_start_j = (L - crack_size_cells) // 2
    crack_end_j = crack_start_j + crack_size_cells

    print(f"Creating pre-crack: row i={middle_i}, columns j={crack_start_j} to {crack_end_j - 1}")

    for j in range(crack_start_j, crack_end_j):
        node1 = idx(middle_i - 1, j)
        node2 = idx(middle_i, j)
        bond = (min(node1, node2), max(node1, node2))
        if bond in bond_map:
            bond_idx = bond_map[bond]
            precrack_bonds.append(bond_idx)
            print(f"  Removing bond {bond_idx}: nodes {node1} -- {node2}")

    return precrack_bonds


# ====================================================================
# GPU-OPTIMIZED CORE
# ====================================================================

@jit
def laplacian_ff_matvec(v, conductance, idx_both, i_both, j_both, idx_pf, i_pf, idx_qf, i_qf):
    """Matrix-free Laplacian matvec on free nodes only using precomputed topology indices."""
    n_free = v.shape[0]
    y = jnp.zeros(n_free, dtype=v.dtype)

    g_both = conductance[idx_both]
    diff = v[i_both] - v[j_both]
    y = y.at[i_both].add(g_both * diff)
    y = y.at[j_both].add(-g_both * diff)

    g_pf = conductance[idx_pf]
    g_qf = conductance[idx_qf]
    y = y.at[i_pf].add(g_pf * v[i_pf])
    y = y.at[i_qf].add(g_qf * v[i_qf])
    return y


@partial(jit, static_argnums=(8,))
def build_rhs_free(conductance, idx_pf, i_pf, q_fixed_pf, idx_qf, i_qf, p_fixed_qf, V_fixed, n_free):
    """Build RHS b = -A_fb * V_fixed on free-node system using precomputed topology indices."""
    b = jnp.zeros(n_free, dtype=V_fixed.dtype)
    b = b.at[i_pf].add(conductance[idx_pf] * V_fixed[q_fixed_pf])
    b = b.at[i_qf].add(conductance[idx_qf] * V_fixed[p_fixed_qf])
    return b


@jit
def solve_voltages_fast_gpu(conductance, idx_both, i_both, j_both, idx_pf, i_pf, idx_qf, i_qf, b_jax, x0=None):
    """Matrix-free CG solve for free-node voltages."""
    def matvec(v):
        return laplacian_ff_matvec(v, conductance, idx_both, i_both, j_both, idx_pf, i_pf, idx_qf, i_qf)

    V_free, _ = cg(matvec, b_jax, x0=x0, tol=1e-5, maxiter=10000)
    return V_free


@jit
def calculate_bond_currents_gpu(V, p_array, q_array, burned_bonds_mask):
    """Vectorized current calculation (no where clause = faster)."""
    diffs = jnp.abs(V[p_array] - V[q_array])
    return diffs * (~burned_bonds_mask).astype(jnp.float32)


@jit
def check_for_failure_gpu(currents, failure_thresholds, burned_bonds_mask):
    """GPU-optimized failure detection with parallel reduction."""
    exceeded_mask = (currents > failure_thresholds) & (~burned_bonds_mask)
    num_failed = jnp.sum(exceeded_mask)
    
    # Use argmax for first exceeded (GPU-friendly)
    exceeded_idx = jnp.argmax(exceeded_mask.astype(jnp.float32))
    
    # Compute highest ratio efficiently
    overcurrent_ratio = jnp.where(exceeded_mask, currents / (failure_thresholds + 1e-8), 0.0)
    highest_ratio_idx = jnp.argmax(overcurrent_ratio)

    return num_failed, exceeded_idx, highest_ratio_idx


def check_for_failure_py(V, p_array, q_array, burned_bonds_mask, failure_thresholds, avalanche=False):
    """Python wrapper for failure logic."""
    currents = calculate_bond_currents_gpu(V, p_array, q_array, burned_bonds_mask)
    num_failed, exceeded_idx, highest_ratio_idx = check_for_failure_gpu(
        currents, failure_thresholds, burned_bonds_mask
    )

    num_failed = int(num_failed)
    if num_failed == 0:
        return -1
    if num_failed == 1:
        return int(exceeded_idx)
    return int(highest_ratio_idx) if avalanche else -2


def prepare_topology_indices(free, p_array_np, q_array_np):
    """Precompute topology index arrays for fast matrix-free CG."""
    free_map_np = np.full(N, -1, dtype=np.int32)
    free_map_np[free] = np.arange(free.size, dtype=np.int32)

    fp = free_map_np[p_array_np]
    fq = free_map_np[q_array_np]

    both_free = (fp >= 0) & (fq >= 0)
    p_free_q_fixed = (fp >= 0) & (fq < 0)
    q_free_p_fixed = (fq >= 0) & (fp < 0)

    idx_both = np.where(both_free)[0].astype(np.int32)
    i_both = fp[both_free].astype(np.int32)
    j_both = fq[both_free].astype(np.int32)

    idx_pf = np.where(p_free_q_fixed)[0].astype(np.int32)
    i_pf = fp[p_free_q_fixed].astype(np.int32)
    q_fixed_pf = q_array_np[p_free_q_fixed].astype(np.int32)

    idx_qf = np.where(q_free_p_fixed)[0].astype(np.int32)
    i_qf = fq[q_free_p_fixed].astype(np.int32)
    p_fixed_qf = p_array_np[q_free_p_fixed].astype(np.int32)

    return (
        jnp.asarray(idx_both, dtype=jnp.int32),
        jnp.asarray(i_both, dtype=jnp.int32),
        jnp.asarray(j_both, dtype=jnp.int32),
        jnp.asarray(idx_pf, dtype=jnp.int32),
        jnp.asarray(i_pf, dtype=jnp.int32),
        jnp.asarray(q_fixed_pf, dtype=jnp.int32),
        jnp.asarray(idx_qf, dtype=jnp.int32),
        jnp.asarray(i_qf, dtype=jnp.int32),
        jnp.asarray(p_fixed_qf, dtype=jnp.int32),
    )


def solve_voltages_gpu(conductance, topology, free, fixed_idx, V_fixed, V_initial=None):
    """Solve voltages with matrix-free GPU CG using conductance mask."""
    idx_both, i_both, j_both, idx_pf, i_pf, q_fixed_pf, idx_qf, i_qf, p_fixed_qf = topology
    V_fixed_jax = jnp.asarray(V_fixed, dtype=jnp.float32)
    n_free = free.shape[0]
    b_jax = build_rhs_free(conductance, idx_pf, i_pf, q_fixed_pf, idx_qf, i_qf, p_fixed_qf, V_fixed_jax, n_free)

    if V_initial is not None:
        x0 = jnp.asarray(V_initial, dtype=jnp.float32)[free]
        V_free = solve_voltages_fast_gpu(conductance, idx_both, i_both, j_both, idx_pf, i_pf, idx_qf, i_qf, b_jax, x0=x0)
    else:
        V_free = solve_voltages_fast_gpu(conductance, idx_both, i_both, j_both, idx_pf, i_pf, idx_qf, i_qf, b_jax)

    V = jnp.zeros(N, dtype=jnp.float32)
    V = V.at[fixed_idx].set(V_fixed_jax[fixed_idx])
    V = V.at[free].set(V_free)
    return V


# ====================================================================
# FAST GPU OBSERVABLES (Vectorized)
# ====================================================================
@jit
def calculate_single_row_current_gpu(V, burned_bonds_mask, p_array, q_array, down_mask, up_mask):
    signed = V[p_array] - V[q_array]
    active = (~burned_bonds_mask).astype(jnp.float32)
    signed_active = signed * active

    down = down_mask.astype(jnp.float32)
    up = up_mask.astype(jnp.float32)
    return jnp.sum(signed_active * down) - jnp.sum(signed_active * up)


def precompute_single_row_crossing(p_array_np, q_array_np, row_idx):
    """Precompute row crossing masks."""
    r0_start, r0_end = row_idx * L, (row_idx + 1) * L
    r1_start, r1_end = (row_idx + 1) * L, (row_idx + 2) * L

    p_in_r0 = (p_array_np >= r0_start) & (p_array_np < r0_end)
    q_in_r0 = (q_array_np >= r0_start) & (q_array_np < r0_end)
    p_in_r1 = (p_array_np >= r1_start) & (p_array_np < r1_end)
    q_in_r1 = (q_array_np >= r1_start) & (q_array_np < r1_end)

    down_mask = jnp.asarray(p_in_r0 & q_in_r1, dtype=jnp.bool_)
    up_mask = jnp.asarray(p_in_r1 & q_in_r0, dtype=jnp.bool_)
    return down_mask, up_mask


def calculate_single_row_current_fast(V, burned_bonds_mask, p_array_np, q_array_np, down_mask, up_mask):
    """Calculate current crossing a row."""
    return float(calculate_single_row_current_gpu(V, burned_bonds_mask, p_array_np, q_array_np, down_mask, up_mask))


# ====================================================================
# FRACTURE SEARCH
# ====================================================================

def stretch_to_fracture(
    conductance,
    V,
    topology,
    free,
    fixed_idx,
    V_fixed_base,
    top_idx,
    p_array,
    q_array,
    burned_bonds_mask,
    failure_thresholds,
    initial_dv=h_step,
    max_iterations=50,
):
    """Quasistatic fracture with GPU optimization."""
    V_fixed = V_fixed_base.copy()
    dv = initial_dv
    safe_state = V_fixed.copy()

    iteration = 0
    while True:
        V_fixed = safe_state.copy()
        V_fixed[top_idx] += dv

        V = solve_voltages_gpu(conductance, topology, free, fixed_idx, V_fixed, V)
        failed_bond_idx = check_for_failure_py(V, p_array, q_array, burned_bonds_mask, failure_thresholds)

        if failed_bond_idx == -2:
            dv *= 0.5
        elif failed_bond_idx == -1:
            safe_state = V_fixed
        else:
            if dv < 1e-3:
                break
            dv *= 0.5

        iteration += 1
        if iteration >= max_iterations:
            print(f"Warning: stretch_to_fracture did not converge after {iteration} iterations, dv={dv:.2e}")
            break

    if failed_bond_idx == -2:
        print("Warning: stretch_to_fracture reached max iterations without isolating a single failed bond.")
        failed_bond_idx = check_for_failure_py(V, p_array, q_array, burned_bonds_mask, failure_thresholds, avalanche=True)

    V_fixed_base[top_idx] = V_fixed[top_idx]
    return failed_bond_idx, V


# ====================================================================
# MAIN
# ====================================================================

def run_quasistatic_fuse_simulation(run_no=0):
    print('Setting up GPU simulation...')
    print(f'Device: {jax.devices()[0]}')

    voltage_history = []
    current_history = []
    avalanche_history = []
    burned_bond_order = []

    bond_map, failure_thresholds = generate_failure_thresholds(shape_param=4.0, scale_param=1.0, seed=run_no)
    n_bonds = len(bond_map)
    p_array, q_array = bond_map_to_arrays(bond_map, n_bonds)

    p_array_np = np.asarray(jax.device_get(p_array))
    q_array_np = np.asarray(jax.device_get(q_array))

    burned_bonds_mask = jnp.zeros(n_bonds, dtype=jnp.bool_)

    fixed = np.zeros(N, dtype=bool)
    V_fixed = np.zeros(N, dtype=np.float32)
    for j in range(L):
        fixed[idx(0, j)] = True
        V_fixed[idx(0, j)] = 0.0
        fixed[idx(L - 1, j)] = True
        V_fixed[idx(L - 1, j)] = 0.0

    free = np.where(~fixed)[0]
    fixed_idx = np.where(fixed)[0]
    top_idx = np.arange((L - 1) * L, L * L, dtype=np.int32)
    topology = prepare_topology_indices(free, p_array_np, q_array_np)

    row_idx_for_current = (L // 2) - 1
    down_mask_single, up_mask_single = precompute_single_row_crossing(
        p_array_np, q_array_np, row_idx_for_current
    )

    conductance = jnp.ones(n_bonds, dtype=jnp.float32)

    crack_size_cells = 2
    precrack_bonds = create_precrack(bond_map, crack_size_cells)
    if len(precrack_bonds) > 0:
        precrack_idx = jnp.asarray(precrack_bonds, dtype=jnp.int32)
        burned_bonds_mask = burned_bonds_mask.at[precrack_idx].set(True)
        conductance = conductance.at[precrack_idx].set(0.0)

    V = solve_voltages_gpu(conductance, topology, free, fixed_idx, V_fixed)

    total_burned = len(precrack_bonds)
    print(f'Pre-crack created with {total_burned} bonds removed\n')

    applied_voltage = V_fixed[top_idx].max()
    total_current = calculate_single_row_current_fast(
        V, burned_bonds_mask, p_array_np, q_array_np, down_mask_single, up_mask_single
    )
    voltage_history.append(applied_voltage)
    current_history.append(abs(total_current))

    num_fractures = 10
    total_burned = 0
    for fracture_event in range(num_fractures):
        print(f'\n=== Primary Fracture Event {fracture_event + 1} ===')

        broken_bond_idx, V = stretch_to_fracture(
            conductance,
            V,
            topology,
            free,
            fixed_idx,
            V_fixed,
            top_idx,
            p_array,
            q_array,
            burned_bonds_mask,
            failure_thresholds,
            initial_dv=h_step,
            max_iterations=20,
        )

        applied_voltage = V_fixed[top_idx].max()
        row_current = calculate_single_row_current_fast(
            V, burned_bonds_mask, p_array_np, q_array_np, down_mask_single, up_mask_single
        )
        current_history.append(abs(row_current))
        voltage_history.append(applied_voltage)

        burned_bonds_mask = burned_bonds_mask.at[broken_bond_idx].set(True)
        burned_bond_order.append(int(broken_bond_idx))
        total_burned += 1
        print(f'Primary failure: Burned bond {broken_bond_idx}')

        conductance = conductance.at[broken_bond_idx].set(0.0)

        V = solve_voltages_gpu(conductance, topology, free, fixed_idx, V_fixed, V)

        failed_bond_idx = check_for_failure_py(
            V, p_array, q_array, burned_bonds_mask, failure_thresholds, avalanche=True
        )

        avalanche_count = 0
        while failed_bond_idx != -1:
            avalanche_count += 1
            total_burned += 1
            print(f'\tAvalanche fracture {avalanche_count}: Burned bond {failed_bond_idx}')

            burned_bonds_mask = burned_bonds_mask.at[failed_bond_idx].set(True)
            burned_bond_order.append(int(failed_bond_idx))
            conductance = conductance.at[failed_bond_idx].set(0.0)

            V = solve_voltages_gpu(conductance, topology, free, fixed_idx, V_fixed, V)
            failed_bond_idx = check_for_failure_py(
                V, p_array, q_array, burned_bonds_mask, failure_thresholds, avalanche=True
            )

        if avalanche_count > 0:
            print(f'\tAvalanche completed with {avalanche_count} additional failures')

        applied_voltage = V_fixed[top_idx].max()
        row_current = calculate_single_row_current_fast(
            V, burned_bonds_mask, p_array_np, q_array_np, down_mask_single, up_mask_single
        )
        current_history.append(abs(row_current))
        voltage_history.append(applied_voltage)
        avalanche_history.append(avalanche_count + 1)

        print(f'Total burned bonds so far: {total_burned}')

        total_current = calculate_single_row_current_fast(
            V, burned_bonds_mask, p_array_np, q_array_np, down_mask_single, up_mask_single
        )
        if abs(total_current) < 1e-2 and V_fixed[top_idx].max() > 0.1:
            print('No more bonds can fail. Simulation complete.')
            break

    V_final = solve_voltages_gpu(conductance, topology, free, fixed_idx, V_fixed, V)

    print(f'\nSimulation ended. Total bonds burned: {total_burned}')
    return V_final, burned_bonds_mask, p_array_np, q_array_np, voltage_history, current_history, avalanche_history, burned_bond_order, failure_thresholds


if __name__ == '__main__':
    import sys
    run_no = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    t0 = time.time()
    final_V, burned_bonds, p_array_np, q_array_np, voltage_history, current_history, avalanche_history, burned_bond_order, failure_thresholds = (
        run_quasistatic_fuse_simulation(run_no=run_no)
    )
    final_V.block_until_ready()
    total_time = time.time() - t0

    print(f'\nSimulation completed in {total_time:.2f} seconds')

    burned_order = np.asarray(burned_bond_order, dtype=np.int32)
    p_ord = p_array_np[burned_order]
    q_ord = q_array_np[burned_order]
    xy = np.column_stack((((p_ord % L) + (q_ord % L)) / 2.0, ((p_ord // L) + (q_ord // L)) / 2.0))
    np.savetxt(f'Output/burned_bond_positions_{run_no}.txt', xy, fmt='%.6f', header='x y')
    np.savetxt(f'Output/avalanche_history_{run_no}.txt', np.asarray(avalanche_history, dtype=np.int32), fmt='%d')
    np.savetxt(f'Output/voltage_current_{run_no}.txt', np.column_stack((voltage_history, current_history)), fmt='%.5e', header='voltage current')