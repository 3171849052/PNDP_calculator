import csv
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

epsilons, ldp, pndp, pndp_min, pndp_max = [], [], [], [], []
with open("noise_multipliers.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epsilons.append(int(row["epsilon"]))
        ldp.append(float(row["LDP"]))
        pndp.append(float(row["PNDP"]))
        pndp_min.append(float(row["PNDP_strict_min"]))
        pndp_max.append(float(row["PNDP_strict_max"]))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(epsilons, ldp, label="LDP", color="#988ED5", linewidth=1)
ax.plot(epsilons, pndp, label="PNDP", color="#348ABD", linewidth=1)
ax.fill_between(epsilons, pndp_min, pndp_max, color="#E24A33", alpha=0.25, label="PNDP_strict")
ax.plot(epsilons, pndp_min, color="#E24A33", linewidth=0.5)
ax.plot(epsilons, pndp_max, color="#E24A33", linewidth=0.5)

ax.set_xlabel("ε")
ax.set_ylabel("Noise Multiplier (σ)")
ax.set_xticks(range(1, 11))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("noise_multiplier.png", dpi=150)
print("Saved noise_multiplier.png")
