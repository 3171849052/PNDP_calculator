import sys
sys.path.insert(0, ".")

import csv
import math
from pndp_calculator.core_accountant import calculate_optimal_sigma_gdp
from pndp_calculator.topology_analyzer import compute_standard_ldp_effective_variance

R_rounds = 20
T_local_steps = 20
N_samples = 4000
batch_size = 256
b_interval = math.ceil(N_samples / batch_size)
delta = 1e-5

# ---- compute base noise (eff_var_mult = 1) for each epsilon ----
base_nm = {}
for eps in range(1, 11):
    nm, _ = calculate_optimal_sigma_gdp(eps, delta, 1.0)
    base_nm[eps] = nm

# ---- LDP ----
ldp_eff_var = compute_standard_ldp_effective_variance(R_rounds, T_local_steps, b_interval)
print(f"LDP effective variance multiplier: {ldp_eff_var:.6f}")

# ---- derive eff_var_mult from existing experiments at epsilon=8 ----
# from APNDP_FGDP/params.json: PNDP nm at eps=8 = 0.5639
# from APNDP_strict_FGDP/params.json: PNDP_strict per-node nm at eps=8
# from ALDP_FGDP/params.json: LDP nm at eps=8 = 0.7940
nm_pndp_at_8 = 0.5639004881727748
ldp_nm_at_8 = 0.7940362497203659
pndp_strict_vals_at_8 = [
    0.7940362496923729, 0.7697889707698234, 0.7724625458191844,
    0.7940362496923729, 0.776650513536127,  0.7851091605860328,
    0.7751421330035749, 0.7752147030961393, 0.7670246449147461,
    0.7851324943293746, 0.7940362496923729, 0.774821763223578,
    0.7517660557399921, 0.7940362496923727, 0.7940362496923734,
]

pndp_eff_var = (nm_pndp_at_8 / base_nm[8]) ** 2
pndp_strict_eff_var = [(v / base_nm[8]) ** 2 for v in pndp_strict_vals_at_8]

print(f"PNDP effective variance multiplier: {pndp_eff_var:.6f}")
print(f"PNDP_strict effective variance multiplier range: "
      f"[{min(pndp_strict_eff_var):.6f}, {max(pndp_strict_eff_var):.6f}]")
print()

rows = []
for eps in range(1, 11):
    nm_ldp = base_nm[eps] * math.sqrt(ldp_eff_var)
    nm_pndp = base_nm[eps] * math.sqrt(pndp_eff_var)
    nm_strict = [base_nm[eps] * math.sqrt(v) for v in pndp_strict_eff_var]
    rows.append({
        "epsilon": eps,
        "LDP": f"{nm_ldp:.4f}",
        "PNDP": f"{nm_pndp:.4f}",
        "PNDP_strict_min": f"{min(nm_strict):.4f}",
        "PNDP_strict_max": f"{max(nm_strict):.4f}",
    })
    print(f"  \u03b5={eps:2d}  LDP={nm_ldp:.4f}  PNDP={nm_pndp:.4f}  "
          f"PNDP_strict=[{min(nm_strict):.4f}, {max(nm_strict):.4f}]")

with open("noise_multipliers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["epsilon", "LDP", "PNDP", "PNDP_strict_min", "PNDP_strict_max"])
    w.writeheader()
    w.writerows(rows)
print("\nSaved noise_multipliers.csv")
