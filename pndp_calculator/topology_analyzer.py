import math
from typing import List, Tuple, Optional, Dict
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import cvxpy as cp


def build_random_walk_matrix(graph: nx.Graph) -> np.ndarray:
    matrix = nx.to_numpy_array(graph)
    n = graph.number_of_nodes()
    degrees = matrix.sum(axis=1)
    mh_matrix = np.zeros_like(matrix)

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                mh_matrix[i, j] = 1.0 / max(degrees[i], degrees[j])
        mh_matrix[i, i] = 1.0 - mh_matrix[i].sum()

    return mh_matrix


def build_W_powers(W: np.ndarray, max_power: int) -> List[np.ndarray]:
    powers = [np.eye(W.shape[0])]
    for _ in range(max_power):
        powers.append(powers[-1] @ W)
    return powers


def build_B_a_sparse(W_powers: List[np.ndarray], R: int, T: int, K: int, target_node: int, graph: nx.Graph) -> sp.csc_matrix:
    n = W_powers[0].shape[0]
    observed_nodes = [target_node] + list(graph.neighbors(target_node))
    observed_nodes = sorted(set(observed_nodes))
    num_obs = len(observed_nodes)

    row_indices = []
    col_indices = []
    data = []

    for r in range(R):
        for k in range(K):
            row_v = r * K + k
            for q in range(r + 1):
                power_idx = (r - q) * K + k
                block = W_powers[power_idx]
                block_sub = block[observed_nodes, :]

                row_start = row_v * num_obs
                col_start = (q * T) * n

                nz_rows, nz_cols = np.nonzero(block_sub)
                vals = block_sub[nz_rows, nz_cols]

                for t in range(T):
                    t_col_start = col_start + t * n
                    row_indices.extend(nz_rows + row_start)
                    col_indices.extend(nz_cols + t_col_start)
                    data.extend(vals)

    B_a = sp.csc_matrix((data, (row_indices, col_indices)), shape=(R * K * num_obs, R * T * n))
    return B_a


def compute_row_space_projector(B: np.ndarray, rtol: Optional[float] = None) -> np.ndarray:
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    if rtol is None:
        tol = max(B.shape) * np.finfo(B.dtype).eps * s[0]
    else:
        tol = rtol * s[0]
    r = int(np.sum(s > tol))
    V_r = Vt[:r, :].T
    P = V_r @ V_r.T
    P = (P + P.T) / 2.0
    return P


def _select_sdp_solver() -> str:
    installed = cp.installed_solvers()
    for preferred in ["CLARABEL", "SCS"]:
        if preferred in installed:
            return preferred
    raise RuntimeError(f"No suitable SDP solver found. Installed: {installed}")


def _compute_kb_pairwise_variance_exact_sdp(
    P: np.ndarray, n_nodes: int, total_steps: int, k: int, b: int,
    strict: bool = True, allow_fallback: bool = False,
    solver: Optional[str] = None,
    solver_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    if solver_stats is None:
        solver_stats = {"optimal": 0, "optimal_inaccurate": 0, "fallback": 0, "other": 0}

    max_variances = np.zeros(n_nodes)
    actual_k = min(k, 1 + (total_steps - 1) // b) if b > 0 else 1

    H_param = cp.Parameter((actual_k, actual_k), symmetric=True)
    X = cp.Variable((actual_k, actual_k), PSD=True)
    objective = cp.Maximize(cp.sum(cp.multiply(H_param, X)))
    constraints = [cp.diag(X) <= 1.0]
    prob = cp.Problem(objective, constraints)

    if solver is None:
        solver = _select_sdp_solver()

    for node_idx in range(n_nodes):
        node_max_var = 0.0
        for start_step in range(total_steps - (actual_k - 1) * b):
            pi_indices = []
            for step_offset in range(actual_k):
                current_step = start_step + step_offset * b
                col_idx = current_step * n_nodes + node_idx
                pi_indices.append(col_idx)

            H_sub = P[np.ix_(pi_indices, pi_indices)]
            H_sub = np.real(H_sub)
            H_sub = (H_sub + H_sub.T) / 2.0

            upper_bound = float(np.sum(np.abs(H_sub)))
            if upper_bound <= node_max_var:
                continue

            H_param.value = H_sub
            try:
                prob.solve(solver=solver, verbose=False)
                status = prob.status
                if status == "optimal":
                    solver_stats["optimal"] += 1
                    if prob.value is not None:
                        current_variance = prob.value
                    else:
                        current_variance = upper_bound
                elif status == "optimal_inaccurate":
                    solver_stats["optimal_inaccurate"] += 1
                    if prob.value is not None:
                        current_variance = prob.value
                    else:
                        current_variance = upper_bound
                elif status == "infeasible":
                    solver_stats["other"] += 1
                    current_variance = 0.0
                else:
                    solver_stats["other"] += 1
                    if strict and not allow_fallback:
                        raise RuntimeError(
                            f"SDP solver failed: status={status}, solver={solver}, "
                            f"node={node_idx}, start_step={start_step}, "
                            f"upper_bound={upper_bound:.6e}"
                        )
                    current_variance = upper_bound
                    if allow_fallback:
                        solver_stats["fallback"] += 1
            except cp.error.SolverError as e:
                solver_stats["other"] += 1
                if strict and not allow_fallback:
                    raise RuntimeError(
                        f"SDP SolverError: {e}, solver={solver}, "
                        f"node={node_idx}, start_step={start_step}"
                    )
                current_variance = upper_bound
                if allow_fallback:
                    solver_stats["fallback"] += 1

            if current_variance > node_max_var:
                node_max_var = current_variance

        max_variances[node_idx] = float(node_max_var)

    return max_variances, solver_stats


def compute_victim_centric_effective_variances(
    W: np.ndarray, R: int, T: int, K: int, graph: nx.Graph,
    k_participation: int, b_interval: int,
    strict: bool = True, allow_fallback: bool = False,
    solver: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    n_nodes = W.shape[0]
    total_steps = R * T
    W_powers = build_W_powers(W, R * K)

    M = np.zeros((n_nodes, n_nodes))
    diag_info: Dict = {
        "solver_stats": {"optimal": 0, "optimal_inaccurate": 0, "fallback": 0, "other": 0},
        "projector_sym_errors": [],
        "projector_idemp_errors": [],
        "projector_eig_min": [],
        "projector_eig_max": [],
        "projector_ranks": [],
        "B_shapes": [],
    }

    for attacker in tqdm(range(n_nodes), desc="Computing Exact SDP (Victim-centric)"):
        B_a_full = build_B_a_sparse(W_powers, R, T, K, attacker, graph)

        attacker_cols = np.arange(total_steps) * n_nodes + attacker
        mask = np.ones(B_a_full.shape[1])
        mask[attacker_cols] = 0.0
        Mask_mat = sp.diags(mask)
        B_a = B_a_full @ Mask_mat

        B_dense = B_a.toarray()
        P = compute_row_space_projector(B_dense)

        sym_err = np.max(np.abs(P - P.T))
        idemp_err = np.max(np.abs(P @ P - P))
        eig_vals = np.linalg.eigvalsh(P)
        diag_info["projector_sym_errors"].append(sym_err)
        diag_info["projector_idemp_errors"].append(idemp_err)
        diag_info["projector_eig_min"].append(float(eig_vals[0]))
        diag_info["projector_eig_max"].append(float(eig_vals[-1]))
        diag_info["projector_ranks"].append(int(np.sum(eig_vals > 1e-10)))
        diag_info["B_shapes"].append(B_dense.shape)

        variances_for_attacker, local_stats = _compute_kb_pairwise_variance_exact_sdp(
            P, n_nodes, total_steps, k_participation, b_interval,
            strict=strict, allow_fallback=allow_fallback, solver=solver,
        )
        for kk in diag_info["solver_stats"]:
            diag_info["solver_stats"][kk] += local_stats[kk]
        M[attacker, :] = variances_for_attacker

    victim_variances = np.zeros(n_nodes)
    worst_attackers = np.zeros(n_nodes, dtype=int)
    worst_distances = np.zeros(n_nodes, dtype=int)
    for u in range(n_nodes):
        best_val = -1.0
        best_attacker = -1
        for a in range(n_nodes):
            if a == u:
                continue
            if M[a, u] > best_val:
                best_val = M[a, u]
                best_attacker = a
        victim_variances[u] = best_val
        worst_attackers[u] = best_attacker
        worst_distances[u] = nx.shortest_path_length(graph, source=u, target=best_attacker)

    return victim_variances, worst_attackers, worst_distances, M, diag_info


def compute_standard_ldp_effective_variance(R: int, T: int, b: int) -> float:
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if b <= 0:
        return float(R * T)

    max_eff_var = 0.0
    for offset in range(b):
        current_eff_var = 0.0
        for r in range(R):
            start_step = r * T
            end_step = r * T + T

            first_idx = start_step + (offset - start_step % b) % b
            if first_idx < end_step:
                z_r = 1 + (end_step - 1 - first_idx) // b
            else:
                z_r = 0

            current_eff_var += (z_r ** 2) / T

        if current_eff_var > max_eff_var:
            max_eff_var = current_eff_var

    return float(max_eff_var)
