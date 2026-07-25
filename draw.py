import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

exps_dir = "exps"
algo_names = {
    "APNDP_strict": "PNDP-strict",
    "APNDP": "PNDP",
    "ALDP": "LDP",
}
colors = {
    "PNDP-strict": "#E24A33",
    "PNDP": "#348ABD",
    "LDP": "#988ED5",
}

plt.rcParams.update({"font.size": 14})

fig, ax = plt.subplots(figsize=(8, 5))

data = {}
for keyword, label in algo_names.items():
    exp_dir = next(d for d in os.listdir(exps_dir) if keyword in d and "E3.0" in d)
    path = os.path.join(exps_dir, exp_dir, "accuracy.csv")
    rounds, accs = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            accs.append(float(row["mean_acc"]))
    ax.plot(rounds, accs, label=label, color=colors[label], linewidth=1)
    data[label] = (rounds, accs)

ax.set_xlabel("Round")
ax.set_ylabel("Average Accuracy (%)")
ax.legend()
ax.grid(alpha=0.3)

axins = ax.inset_axes([0.45, 0.05, 0.5, 0.5])
for label in ["PNDP-strict", "PNDP", "LDP"]:
    rounds, accs = data[label]
    axins.plot(rounds, accs, label=label, color=colors[label], linewidth=1)
axins.set_xlim(14.5, 20.5)
axins.set_ylim(85, 90)
axins.grid(alpha=0.3)
axins.tick_params(labelsize=9)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", linestyle="--", linewidth=0.8)

fig.tight_layout()
fig.savefig("average_acc.png", dpi=150)
print("Saved average_acc.png")
