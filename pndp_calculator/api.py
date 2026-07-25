import math
import numpy as np
import networkx as nx

from .topology_analyzer import (
    build_random_walk_matrix,
    compute_victim_centric_effective_variances,
    compute_standard_ldp_effective_variance,
)
from .core_accountant import calculate_optimal_sigma_rdp, calculate_optimal_sigma_gdp


class PNDPAccountant:
    def __init__(self, N_samples: int, batch_size: int, T_local_steps: int, R_rounds: int, K_gossip: int = 1):
        if R_rounds <= 0:
            raise ValueError(f"R_rounds must be > 0, got {R_rounds}")
        if T_local_steps <= 0:
            raise ValueError(f"T_local_steps must be > 0, got {T_local_steps}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.N_samples = N_samples
        self.batch_size = batch_size
        self.T_local_steps = T_local_steps
        self.R_rounds = R_rounds
        self.K_gossip = K_gossip
        
        self.b_interval = int(math.ceil(N_samples / batch_size))
        self.total_steps = int(R_rounds * T_local_steps)
        self.k_participation = int(math.ceil(self.total_steps / self.b_interval))
        self.graph = None
        self.original_nodes = []
        self.diag_info = None

    def set_graph(self, graph: nx.Graph):
        self.original_nodes = list(graph.nodes())
        self.graph = nx.convert_node_labels_to_integers(graph)
        
    def _ensure_variance_cache(self):
        if self.graph is None:
            raise ValueError("Graph must be set using set_graph() for decentralized algorithms.")
        if hasattr(self, '_M_cache'):
            return
        W = build_random_walk_matrix(self.graph)
        n_nodes = self.graph.number_of_nodes()
        if n_nodes <= 1:
            raise ValueError(
                f"PNDP requires at least 2 nodes, got {n_nodes}. "
                "Use LDP for single-node or trivial settings."
            )
        victim_variances, worst_attackers, worst_distances, M, diag_info = (
            compute_victim_centric_effective_variances(
                W, self.R_rounds, self.T_local_steps, self.K_gossip, self.graph,
                self.k_participation, self.b_interval
            )
        )
        attacker_avg_variances = np.zeros(n_nodes)
        for a in range(n_nodes):
            sum_variance = 0.0
            count = 0
            for u in range(n_nodes):
                if a != u:
                    sum_variance += M[a, u]
                    count += 1
            attacker_avg_variances[a] = sum_variance / count if count > 0 else 0.0

        ldp_var = compute_standard_ldp_effective_variance(
            self.R_rounds, self.T_local_steps, self.b_interval
        )

        self._M_cache = {
            'M': M,
            'avg_eff_var_mult': float(np.max(attacker_avg_variances)),
            'node_variances': victim_variances.copy(),
            'worst_attackers': worst_attackers,
            'worst_distances': worst_distances,
        }
        self.diag_info = {
            'b_interval': self.b_interval,
            'ldp_effective_variance': ldp_var,
            'node_variances': victim_variances.tolist(),
            'avg_eff_var_mult': float(np.max(attacker_avg_variances)),
            'max_strict_pndp_variance': float(np.max(victim_variances)),
            'solver_stats': dict(diag_info.get('solver_stats', {})),
        }

    def get_noise_multiplier(self, target_epsilon: float, target_delta: float, algorithm="PNDP", framework="RDP"):
        if algorithm in ("LDP", "LDP-per-round"):
            eff_var_mult = compute_standard_ldp_effective_variance(
                self.R_rounds, self.T_local_steps, self.b_interval
            )
            nm, metric = self._compute_final_multiplier(target_epsilon, target_delta, eff_var_mult, framework)
            return nm, metric
        elif algorithm in ("PNDP_strict", "PNDP"):
            self._ensure_variance_cache()
            cache = self._M_cache

            if algorithm == "PNDP_strict":
                personalized_noise_multipliers = {}
                for node_id, var_mult in enumerate(cache['node_variances']):
                    nm, _ = self._compute_final_multiplier(target_epsilon, target_delta, float(var_mult), framework)
                    original_name = self.original_nodes[node_id]
                    personalized_noise_multipliers[original_name] = nm
                return personalized_noise_multipliers
            elif algorithm == "PNDP":
                nm, metric = self._compute_final_multiplier(target_epsilon, target_delta, cache['avg_eff_var_mult'], framework)
                return nm, metric
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _compute_final_multiplier(self, target_epsilon, target_delta, eff_var_mult, framework):
        if framework == "RDP":
            noise_mult, metric = calculate_optimal_sigma_rdp(target_epsilon, target_delta, eff_var_mult)
        elif framework == "GDP":
            noise_mult, metric = calculate_optimal_sigma_gdp(target_epsilon, target_delta, eff_var_mult)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        return noise_mult, metric
