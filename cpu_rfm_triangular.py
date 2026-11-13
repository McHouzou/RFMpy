import os
import time
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from numba import jit, njit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# CPU optimizations
os.environ['MKL_NUM_THREADS'] = '0'  # Let scipy/numpy use all threads
os.environ['NUMEXPR_NUM_THREADS'] = '0'

# Parameters
L = 64
N = L * L
h_step = 0.1

# Cache for sparse matrix conversions and compiled functions
_sparse_matrix_cache = {}
_jitted_functions = {}


def idx(i, j):
    return i * L + j


def _wrap_j(j):
    return j % L


def get_node_neighbors(i, j, lattice_type='square', periodic_vertical=False):
    """Return nearest neighbors for node (i, j)."""
    lattice = lattice_type.lower()

    if lattice == 'square':
        candidates = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    elif lattice == 'triangular':
        # Triangular lattice with alternating row offsets:
        # x = j + 0.5*(i % 2), y = i*sqrt(3)/2
        # Six nearest neighbors at 60-degree bond orientations.
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
        raise ValueError(f"Unknown lattice_type='{lattice_type}'. Use 'square' or 'triangular'.")

    neighbors = []
    for ni, nj in candidates:
        if not (0 <= ni < L):
            continue
        if periodic_vertical:
            nj = _wrap_j(nj)
        elif not (0 <= nj < L):
            continue
        neighbors.append((ni, nj))

    return neighbors


def build_bond_map(lattice_type='square', periodic_vertical=False):
    """Build unique undirected bond map for the chosen lattice topology."""
    bond_map = {}
    n_bonds = 0

    for i in range(L):
        for j in range(L):
            p = idx(i, j)
            for ni, nj in get_node_neighbors(i, j, lattice_type=lattice_type, periodic_vertical=periodic_vertical):
                q = idx(ni, nj)
                if p == q:
                    continue
                bond = (min(p, q), max(p, q))
                if bond not in bond_map:
                    bond_map[bond] = n_bonds
                    n_bonds += 1

    return bond_map


def get_node_coordinates(lattice_type='square'):
    """Return node coordinates for plotting/saving bond positions."""
    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    tri_y_scale = np.sqrt(3.0) / 2.0

    for i in range(L):
        for j in range(L):
            p = idx(i, j)
            if lattice_type.lower() == 'triangular':
                x[p] = j + 0.5 * (i % 2)
                y[p] = i * tri_y_scale
            else:
                x[p] = j
                y[p] = i

    return x, y


def parse_bool_cli(value):
    """Parse bool values from CLI strings."""
    val = str(value).strip().lower()
    if val in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if val in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


# ====================================================================
# MATRIX / BONDS / THRESHOLDS
# ====================================================================

def build_connectivity_matrix(bond_map):
    """Build sparse connectivity matrix from undirected bond map."""
    rows, cols, vals = [], [], []
    diag = np.zeros(N, dtype=np.float64)

    for p, q in bond_map.keys():
        rows.append(p)
        cols.append(q)
        vals.append(-1.0)

        rows.append(q)
        cols.append(p)
        vals.append(-1.0)

        diag[p] += 1.0
        diag[q] += 1.0

    node_idx = np.arange(N, dtype=np.int32)
    rows.extend(node_idx.tolist())
    cols.extend(node_idx.tolist())
    vals.extend(diag.tolist())

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


def generate_failure_thresholds(n_bonds, disorder_param=1.0, distribution='weibull', seed=42):
    """Generate bond failure thresholds for the selected disorder distribution."""
    if disorder_param <= 0:
        raise ValueError("disorder_param (D) must be > 0")

    np.random.seed(seed)

    u = np.random.uniform(0, 1, size=n_bonds)

    dist = distribution.lower()
    if dist == 'weibull':
        weibull_modulus = 2.0 / disorder_param
        thresholds = (-np.log(1 - u)) ** (1.0 / weibull_modulus)
    elif dist == 'powerlaw_unit':
        # Requested CDF form: P(I) = I^(1/D), 0 < I < 1
        # Inverse transform gives I = U^D.
        thresholds = u ** disorder_param
    else:
        raise ValueError(
            f"Unknown threshold distribution '{distribution}'. "
            "Use 'weibull' or 'powerlaw_unit'."
        )

    return thresholds.astype(np.float64)


def bond_map_to_arrays(bond_map, n_bonds):
    """Convert bond map dictionary to numpy arrays."""
    p_list = np.zeros(n_bonds, dtype=np.int32)
    q_list = np.zeros(n_bonds, dtype=np.int32)
    for bond, bidx in bond_map.items():
        p, q = bond
        p_list[bidx] = p
        q_list[bidx] = q
    return p_list, q_list


def create_precrack(bond_map, crack_size_cells):
    """Create a horizontal pre-crack by removing all bonds crossing the middle interface."""
    if crack_size_cells % 2 != 0 or crack_size_cells >= L:
        raise ValueError("crack_size_cells must be an even number less than L")

    precrack_bonds = []
    middle_i = L // 2
    crack_start_j = (L - crack_size_cells) // 2
    crack_end_j = crack_start_j + crack_size_cells

    print(
        f"Creating pre-crack across interface rows ({middle_i - 1}, {middle_i}), "
        f"columns j={crack_start_j} to {crack_end_j - 1}"
    )

    upper_row = middle_i - 1
    lower_row = middle_i

    for (p, q), bond_idx in bond_map.items():
        pi, pj = divmod(p, L)
        qi, qj = divmod(q, L)

        # Keep only bonds crossing the selected row interface, independent of orientation.
        if not ((pi == upper_row and qi == lower_row) or (pi == lower_row and qi == upper_row)):
            continue

        # Exclude periodic wrap-around bonds (e.g. j=0 to j=L-1).
        # The crack is a local segment and should not remove boundary-crossing links.
        if abs(pj - qj) > 1:
            continue

        # Use the average column to decide whether bond lies inside crack span.
        # This captures both vertical and diagonal crossing bonds in triangular lattices.
        col_mid = 0.5 * (pj + qj)
        if crack_start_j <= col_mid < crack_end_j:
            precrack_bonds.append(bond_idx)
            print(f"  Removing bond {bond_idx}: nodes {p} -- {q}")

    precrack_bonds = sorted(precrack_bonds)
    print(f"  Total pre-crack bonds removed: {len(precrack_bonds)}")

    return precrack_bonds


# ====================================================================
# CPU-OPTIMIZED SOLVING (NumPy/SciPy)
# ====================================================================

def solve_voltages(A, free, fixed_idx, V_fixed, V_initial=None, tol=1e-5, maxiter=10000):
    """
    Solve for voltages at free nodes using sparse CG solver.
    Uses caching to avoid redundant matrix decompositions.
    """
    cache_key = id(A)
    if cache_key in _sparse_matrix_cache:
        cache_entry = _sparse_matrix_cache[cache_key]
        A_ff = cache_entry['A_ff']
        A_fb = cache_entry['A_fb']
    else:
        A_ff = A[free, :][:, free]
        A_fb = A[free, :][:, fixed_idx]
        _sparse_matrix_cache[cache_key] = {'A_ff': A_ff, 'A_fb': A_fb}

    b = -A_fb @ V_fixed[fixed_idx]

    if V_initial is not None:
        x0 = V_initial[free]
    else:
        x0 = None

    V_free, info = cg(A_ff, b, x0=x0, rtol=tol, maxiter=maxiter)
    
    if info > 0:
        print(f"Warning: CG did not converge after {info} iterations")
    elif info < 0:
        print(f"Warning: CG illegal input or breakdown")

    V = np.zeros(N, dtype=np.float64)
    V[fixed_idx] = V_fixed[fixed_idx]
    V[free] = V_free
    return V


@njit
def calculate_bond_currents_numba(V, p_array, q_array, burned_bonds_mask):
    """JIT-compiled bond current calculation."""
    diffs = np.abs(V[p_array] - V[q_array])
    return np.where(burned_bonds_mask, 0.0, diffs)


def calculate_bond_currents(V, p_array, q_array, burned_bonds_mask):
    """Calculate current through each bond."""
    return calculate_bond_currents_numba(V, p_array, q_array, burned_bonds_mask)


@njit
def check_for_failure_numba(currents, failure_thresholds, burned_bonds_mask):
    """JIT-compiled failure detection."""
    exceeded_mask = (currents > failure_thresholds) & (~burned_bonds_mask)
    num_failed = np.sum(exceeded_mask)

    if num_failed == 0:
        return 0, -1, -1
    
    # Find first exceeded index
    exceeded_idx = -1
    for i in range(len(exceeded_mask)):
        if exceeded_mask[i]:
            exceeded_idx = i
            break

    # Find highest ratio
    highest_ratio_idx = -1
    highest_ratio = 0.0
    for i in range(len(exceeded_mask)):
        if exceeded_mask[i]:
            ratio = currents[i] / failure_thresholds[i]
            if ratio > highest_ratio:
                highest_ratio = ratio
                highest_ratio_idx = i

    return num_failed, exceeded_idx, highest_ratio_idx


def check_for_failure(V, p_array, q_array, burned_bonds_mask, failure_thresholds):
    """Wrapper for failure checking with avalanche logic."""
    currents = calculate_bond_currents(V, p_array, q_array, burned_bonds_mask)
    num_failed, exceeded_idx, highest_ratio_idx = check_for_failure_numba(
        currents, failure_thresholds, burned_bonds_mask
    )

    num_failed = int(num_failed)
    if num_failed == 0:
        return -1, None
    if num_failed == 1:
        return 0, int(exceeded_idx)
    return -2, int(highest_ratio_idx)


# ====================================================================
# FAST CPU-SIDE OBSERVABLES (Vectorized NumPy)
# ====================================================================

def precompute_row_crossings(p_array_np, q_array_np):
    """Precompute which bonds cross each row boundary."""
    down_masks = []
    up_masks = []
    for row_idx in range(L - 1):
        r0_start, r0_end = row_idx * L, (row_idx + 1) * L
        r1_start, r1_end = (row_idx + 1) * L, (row_idx + 2) * L

        p_in_r0 = (p_array_np >= r0_start) & (p_array_np < r0_end)
        q_in_r0 = (q_array_np >= r0_start) & (q_array_np < r0_end)
        p_in_r1 = (p_array_np >= r1_start) & (p_array_np < r1_end)
        q_in_r1 = (q_array_np >= r1_start) & (q_array_np < r1_end)

        down_masks.append(p_in_r0 & q_in_r1)
        up_masks.append(p_in_r1 & q_in_r0)

    return np.stack(down_masks, axis=0), np.stack(up_masks, axis=0)


def calculate_all_row_currents_fast(V, burned_bonds_mask, p_array_np, q_array_np, down_masks, up_masks):
    """Calculate net current in each row (vectorized)."""
    signed = V[p_array_np] - V[q_array_np]
    active = ~burned_bonds_mask
    signed_active = signed * active
    return down_masks @ signed_active - up_masks @ signed_active


def precompute_single_row_crossing(p_array_np, q_array_np, row_idx):
    """Precompute masks for a single row boundary."""
    r0_start, r0_end = row_idx * L, (row_idx + 1) * L
    r1_start, r1_end = (row_idx + 1) * L, (row_idx + 2) * L

    p_in_r0 = (p_array_np >= r0_start) & (p_array_np < r0_end)
    q_in_r0 = (q_array_np >= r0_start) & (q_array_np < r0_end)
    p_in_r1 = (p_array_np >= r1_start) & (p_array_np < r1_end)
    q_in_r1 = (q_array_np >= r1_start) & (q_array_np < r1_end)

    down_mask = p_in_r0 & q_in_r1
    up_mask = p_in_r1 & q_in_r0
    return down_mask, up_mask


def calculate_single_row_current_fast(V, burned_bonds_mask, p_array_np, q_array_np, down_mask, up_mask):
    """Calculate current crossing a single row (vectorized)."""
    signed = V[p_array_np] - V[q_array_np]
    active = ~burned_bonds_mask
    signed_active = signed * active
    return float(np.sum(signed_active[down_mask]) - np.sum(signed_active[up_mask]))


def precompute_source_signs(p_array_np, q_array_np, source_nodes):
    """Precompute signs for source current contributions."""
    source_set = set(int(node) for node in source_nodes)
    signs = np.zeros(len(p_array_np), dtype=np.float64)
    for b in range(len(p_array_np)):
        p = int(p_array_np[b])
        q = int(q_array_np[b])
        if (p in source_set) and (q not in source_set):
            signs[b] = 1.0
        elif (q in source_set) and (p not in source_set):
            signs[b] = -1.0
    return signs


def calculate_total_current_fast(V, p_array_np, q_array_np, burned_bonds_mask, source_signs):
    """Calculate total current flowing through source nodes."""
    signed = V[p_array_np] - V[q_array_np]
    active = ~burned_bonds_mask
    return float(np.sum(source_signs * signed * active))


# ====================================================================
# FRACTURE SEARCH
# ====================================================================

def stretch_to_fracture(
    A,
    V,
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
    """
    Quasistatically increase voltage until the next bond fails.
    Uses binary search to find the voltage increment that causes the first failure.
    """
    V_fixed = V_fixed_base.copy()
    dv = initial_dv
    safe_state = V_fixed.copy()
    failed_state = None
    last_failure_idx = -1

    iteration = 0
    while True:
        V_fixed = safe_state.copy()
        V_fixed[top_idx] += dv

        V = solve_voltages(A, free, fixed_idx, V_fixed, V)
        failure_flag, failure_bond_idx = check_for_failure(V, p_array, q_array, burned_bonds_mask, failure_thresholds)

        if failure_flag == -2: # Multiple failures - need to reduce step, but save the last failure state.
            dv *= 0.5
            last_failure_idx = failure_bond_idx
            failed_state = V.copy()

        elif failure_flag == -1: # No failure - safe to increase voltage further
            safe_state = V_fixed

        else: # Single failure - found the next bond to break
            last_failure_idx = failure_bond_idx
            failed_state = V.copy()
            if dv < 1e-3:
                break
            dv *= 0.5

        iteration += 1
        if iteration >= max_iterations and last_failure_idx != -1:
            print(f"Warning: stretch_to_fracture did not converge after {iteration} iterations, dv={dv:.2e}")
            #Falling back to the most stressed bond among those that failed at the last step where we had a failure
            break

    if failure_flag == -2:
        print("Warning: stretch_to_fracture reached max iterations without isolating a single failed bond. Returning most stressed bond among those that failed.")
        #failed_bond_idx = check_for_failure(V, p_array, q_array, burned_bonds_mask, failure_thresholds, avalanche=True)

    V_fixed_base[top_idx] = V_fixed[top_idx]
    return last_failure_idx, failed_state


# ====================================================================
# SIMULATOR
# ====================================================================

def run_quasistatic_fuse_simulation(
    run_no=0,
    disorder_param=0.5,
    failure_distribution='weibull',
    lattice_type='square',
    periodic_vertical=False,
    crack_size_cells=0,
):
    print('Setting up simulation...')
    print(f'Lattice type: {lattice_type}')
    print(f'Periodic vertical boundaries: {periodic_vertical}')
    print(f'Failure threshold distribution: {failure_distribution}')
    print(f'Disorder parameter D: {disorder_param}')

    voltage_history = []
    current_history = []
    avalanche_history = []
    burned_bond_order = []

    bond_map = build_bond_map(
        lattice_type=lattice_type,
        periodic_vertical=periodic_vertical,
    )
    failure_thresholds = generate_failure_thresholds(
        len(bond_map),
        disorder_param=disorder_param,
        distribution=failure_distribution,
        seed=run_no,
    )
    n_bonds = len(bond_map)
    p_array, q_array = bond_map_to_arrays(bond_map, n_bonds)

    p_array_np = np.asarray(p_array, dtype=np.int32)
    q_array_np = np.asarray(q_array, dtype=np.int32)

    burned_bonds_mask = np.zeros(n_bonds, dtype=np.bool_)

    fixed = np.zeros(N, dtype=bool)
    V_fixed = np.zeros(N, dtype=np.float64)
    for j in range(L):
        fixed[idx(0, j)] = True
        V_fixed[idx(0, j)] = 0.0
        fixed[idx(L - 1, j)] = True
        V_fixed[idx(L - 1, j)] = 0.0

    free = np.where(~fixed)[0]
    fixed_idx = np.where(fixed)[0]
    top_idx = np.arange((L - 1) * L, L * L, dtype=np.int32)

    row_idx_for_current = (L // 2) - 1
    down_mask_single, up_mask_single = precompute_single_row_crossing(
        p_array_np, q_array_np, row_idx_for_current
    )

    A = build_connectivity_matrix(bond_map)

    precrack_bonds = create_precrack(bond_map, crack_size_cells)
    for bond_idx in precrack_bonds:
        burned_bonds_mask[bond_idx] = True

    for bond_idx in precrack_bonds:
        p = int(p_array_np[bond_idx])
        q = int(q_array_np[bond_idx])
        data = np.array([1.0, 1.0, -1.0, -1.0])
        rows = np.array([p, q, p, q])
        cols = np.array([q, p, p, q])
        A = (A + sp.coo_matrix((data, (rows, cols)), shape=A.shape)).tocsr()

    _sparse_matrix_cache.clear()
    V = solve_voltages(A, free, fixed_idx, V_fixed)

    total_burned = len(precrack_bonds)
    print(f'Pre-crack created with {total_burned} bonds removed\n')

    # Record initial state before any additional fractures
    applied_voltage = V_fixed[top_idx].max()
    total_current = calculate_single_row_current_fast(
        V, burned_bonds_mask, p_array_np, q_array_np, down_mask_single, up_mask_single
    )
    voltage_history.append(applied_voltage)
    current_history.append(abs(total_current))

    num_fractures = 20000
    total_burned = 0
    for fracture_event in range(num_fractures):
        print(f'\n=== Primary Fracture Event {fracture_event + 1} ===')

        broken_bond_idx, V = stretch_to_fracture(
            A,
            V,
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

        burned_bonds_mask[broken_bond_idx] = True
        burned_bond_order.append(int(broken_bond_idx))
        total_burned += 1
        p = int(p_array_np[broken_bond_idx])
        q = int(q_array_np[broken_bond_idx])
        print(f'Primary failure: Burned bond {broken_bond_idx}')

        data = np.array([1.0, 1.0, -1.0, -1.0])
        rows = np.array([p, q, p, q])
        cols = np.array([q, p, p, q])
        A = (A + sp.coo_matrix((data, (rows, cols)), shape=A.shape)).tocsr()
        _sparse_matrix_cache.clear()

        V = solve_voltages(A, free, fixed_idx, V_fixed, V)

        # Only retain failed bond and not the failure flag, since we will check for multiple failures in the next step.
        _, failed_bond_idx = check_for_failure(V, p_array, q_array, burned_bonds_mask, failure_thresholds)

        avalanche_count = 0
        while failed_bond_idx is not None: # While there are still failed bonds to process, keep burning them and updating the system until no more failures are detected.
            avalanche_count += 1
            total_burned += 1
            print(f'\tAvalanche fracture {avalanche_count}: Burned bond {failed_bond_idx}')

            burned_bonds_mask[failed_bond_idx] = True
            burned_bond_order.append(int(failed_bond_idx))

            p = int(p_array_np[failed_bond_idx])
            q = int(q_array_np[failed_bond_idx])
            data = np.array([1.0, 1.0, -1.0, -1.0])
            rows = np.array([p, q, p, q])
            cols = np.array([q, p, p, q])
            A = (A + sp.coo_matrix((data, (rows, cols)), shape=A.shape)).tocsr()
            _sparse_matrix_cache.clear()

            V = solve_voltages(A, free, fixed_idx, V_fixed, V)
            _, failed_bond_idx = check_for_failure(V, p_array, q_array, burned_bonds_mask, failure_thresholds)

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
        if abs(total_current) < 1e-1 and V_fixed[top_idx].max() > 0.1:
            print('No more bonds can fail. Simulation complete.')
            break

    V_final = solve_voltages(A, free, fixed_idx, V_fixed, V)

    print(f'\nSimulation ended. Total bonds burned: {total_burned}')
    return V_final, burned_bonds_mask, p_array_np, q_array_np, voltage_history, current_history, avalanche_history, burned_bond_order, precrack_bonds, failure_thresholds


# ====================================================================
# MAIN
# ====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Run a quasi-static fuse simulation.')
    parser.add_argument('run_no', nargs='?', type=int, default=0, help='Run number used for seeding and output file names.')
    parser.add_argument('disorder_param', nargs='?', type=float, default=0.5, help='Disorder parameter D.')
    parser.add_argument('output_dir', nargs='?', default='Output', help='Directory for saved output files.')
    parser.add_argument('lattice_type', nargs='?', choices=('square', 'triangular'), default='triangular', help='Lattice topology.')
    parser.add_argument('periodic_vertical', nargs='?', type=parse_bool_cli, default=False, help='Use periodic vertical boundaries.')
    parser.add_argument('crack_size_cells', nargs='?', type=int, default=0, help='Size of the pre-crack in lattice cells.')
    parser.add_argument('failure_distribution', nargs='?', choices=('weibull', 'powerlaw_unit'), default='weibull', help='Bond threshold distribution.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    t0 = time.time()
    final_V, burned_bonds, p_array_np, q_array_np, voltage_history, current_history, avalanche_history, burned_bond_order, precrack_bonds, failure_thresholds = (
        run_quasistatic_fuse_simulation(
            run_no=args.run_no,
            disorder_param=args.disorder_param,
            failure_distribution=args.failure_distribution,
            lattice_type=args.lattice_type,
            periodic_vertical=args.periodic_vertical,
            crack_size_cells=args.crack_size_cells,
        )
    )
    total_time = time.time() - t0

    print(f'\nSimulation completed in {total_time:.2f} seconds')

    # Save results
    burned_order = np.asarray(burned_bond_order, dtype=np.int32)
    precrack_order = np.asarray(precrack_bonds, dtype=np.int32)
    p_ord = p_array_np[burned_order]
    q_ord = q_array_np[burned_order]
    p_pre = p_array_np[precrack_order]
    q_pre = q_array_np[precrack_order]
    node_x, node_y = get_node_coordinates(lattice_type=args.lattice_type)
    xy = np.column_stack(((node_x[p_ord] + node_x[q_ord]) / 2.0, (node_y[p_ord] + node_y[q_ord]) / 2.0))
    xy_pre = np.column_stack(((node_x[p_pre] + node_x[q_pre]) / 2.0, (node_y[p_pre] + node_y[q_pre]) / 2.0))
    os.makedirs(args.output_dir, exist_ok=True)
    np.savetxt(args.output_dir + f'/precrack_bond_order_{args.run_no}.txt', precrack_order, fmt='%d')
    np.savetxt(args.output_dir + f'/precrack_bond_positions_{args.run_no}.txt', xy_pre, fmt='%.3f', header='x y')
    np.savetxt(args.output_dir + f'/burned_bond_order_{args.run_no}.txt', burned_order, fmt='%d')
    np.savetxt(args.output_dir + f'/burned_bond_positions_{args.run_no}.txt', xy, fmt='%.3f', header='x y')
    np.savetxt(args.output_dir + f'/avalanche_history_{args.run_no}.txt', np.asarray(avalanche_history, dtype=np.int32), fmt='%d')
    np.savetxt(args.output_dir + f'/voltage_current_{args.run_no}.txt', np.column_stack((voltage_history, current_history)), fmt='%.5e', header='voltage current')
    np.savetxt(args.output_dir + f'/final_voltage_{args.run_no}.txt', final_V, fmt='%.5e')
    np.savetxt(args.output_dir + f'/failure_thresholds_{args.run_no}.txt', failure_thresholds, fmt='%.5e')