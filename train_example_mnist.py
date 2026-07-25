import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import networkx as nx
import copy
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from pndp_calculator import PNDPAccountant

def _env_bool(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")

def _env_int(key, default):
    val = os.environ.get(key)
    return int(val) if val is not None else default

def _env_float(key, default):
    val = os.environ.get(key)
    return float(val) if val is not None else default

DATASET = os.environ.get("DATASET", "MNIST")
GRAPH = os.environ.get("GRAPH", "florentine_families")
BATCH_SIZE = _env_int("BATCH_SIZE", 64)
MAX_PHYSICAL_BATCH_SIZE = _env_int("MAX_PHYSICAL_BATCH_SIZE", 64)
T_LOCAL_STEPS = _env_int("T_LOCAL_STEPS", 10)
R_ROUNDS = _env_int("R_ROUNDS", 50)
K_GOSSIP = _env_int("K_GOSSIP", 1)
EPSILON = _env_float("EPSILON", 8.0)
DELTA = _env_float("DELTA", 1e-5)
CLIP_NORM = _env_int("CLIP_NORM", 1)
LR = _env_float("LR", 1e-3)
GPU = _env_int("GPU", 1)
FRAMEWORK = os.environ.get("FRAMEWORK", "GDP")
ALGORITHM = os.environ.get("ALGORITHM", "PNDP")
ENABLE_PRIVACY = _env_bool("ENABLE_PRIVACY", False)
_set_nm_raw = os.environ.get("SET_NM")
SET_NM = float(_set_nm_raw) if _set_nm_raw and _set_nm_raw.lower() != "none" else None
CACHE_NOISE = _env_bool("CACHE_NOISE", True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_graph(name):
    graph_fns = {
        "florentine_families": nx.florentine_families_graph,
    }
    if name not in graph_fns:
        raise ValueError(f"Unknown graph: {name}")
    return graph_fns[name]()


class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DecentralizedNode:
    def __init__(self, node_id, data_indices, dataset, noise_multiplier, num_classes=10, enable_privacy=True, init_state_dict=None):
        self.node_id = node_id
        self.enable_privacy = enable_privacy
        self.model = ModuleValidator.fix(MNISTCNN(num_classes))
        if init_state_dict is not None:
            self.model.load_state_dict(init_state_dict, strict=False)
        self.model = self.model.to(DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()

        local_subset = Subset(dataset, data_indices)
        self.dataloader = DataLoader(local_subset, batch_size=BATCH_SIZE, shuffle=True)

        self.noise_multiplier = noise_multiplier

        if enable_privacy:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=CLIP_NORM,
            )

    def local_update(self, local_steps, desc=None):
        self.model.train()
        accumulation_steps = BATCH_SIZE // MAX_PHYSICAL_BATCH_SIZE
        physical_steps_needed = local_steps * accumulation_steps
        physical_steps_done = 0
        step_pbar = tqdm(total=physical_steps_needed, desc=desc, leave=False)

        if self.enable_privacy:
            while physical_steps_done < physical_steps_needed:
                with BatchMemoryManager(
                    data_loader=self.dataloader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=self.optimizer
                ) as memory_safe_data_loader:
                    for batch in memory_safe_data_loader:
                        images, labels = batch
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        self.optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()

                        physical_steps_done += 1
                        step_pbar.update(1)
                        step_pbar.set_postfix(loss=f"{loss.item():.4f}")
                        if physical_steps_done >= physical_steps_needed:
                            break
        else:
            data_iterator = iter(self.dataloader)
            self.optimizer.zero_grad()

            while physical_steps_done < physical_steps_needed:
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.dataloader)
                    batch = next(data_iterator)

                images, labels = batch
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                physical_steps_done += 1

                if physical_steps_done % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                step_pbar.update(1)
                step_pbar.set_postfix(loss=f"{loss.item():.4f}")

        step_pbar.close()
        return loss.item()


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def main():
    if GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

    G = get_graph(GRAPH)
    nodes_list = list(G.nodes())
    num_nodes = len(nodes_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = os.environ.get("OUT_DIR") or os.path.join(
        "exps",
        f"{timestamp}_{DATASET}_{GRAPH}_E{EPSILON}_DP_{str(ENABLE_PRIVACY).lower()}_R{R_ROUNDS}_N{num_nodes}_K{K_GOSSIP}_T{T_LOCAL_STEPS}_B{BATCH_SIZE}_CN{CLIP_NORM}_LR{LR}_A{ALGORITHM}_F{FRAMEWORK}"
    )
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "train.log")
    print(f"Log: {log_path}")

    set_seed(42)
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    samples_per_node = len(train_dataset) // num_nodes

    node_nm_map = None
    nm = 0.0
    if ENABLE_PRIVACY:
        existing_params = os.path.join(out_dir, "params.json")
        if CACHE_NOISE and os.path.exists(existing_params):
            try:
                with open(existing_params, "r") as f:
                    prev = json.load(f)
                if "noise_multiplier_per_node" in prev:
                    node_nm_map = {k: v for k, v in prev["noise_multiplier_per_node"].items()}
                    nm = float(max(node_nm_map.values()))
                    print(f"[Privacy] Loaded per-node Noise Multipliers from existing params.json")
                elif "noise_multiplier" in prev:
                    nm = float(prev["noise_multiplier"])
                    print(f"[Privacy] Loaded Noise Multiplier: {nm:.4f} from existing params.json")
            except (json.JSONDecodeError, IOError, KeyError):
                nm = 0.0

        if SET_NM is not None:
            nm = SET_NM
            node_nm_map = None
            print(f"[Privacy] Using Set Noise Multiplier: {nm:.4f}")
        elif nm == 0.0:
            acc = PNDPAccountant(
                N_samples=samples_per_node,
                batch_size=BATCH_SIZE,
                T_local_steps=T_LOCAL_STEPS,
                R_rounds=R_ROUNDS,
                K_gossip=K_GOSSIP,
            )
            acc.set_graph(G)
            result = acc.get_noise_multiplier(EPSILON, DELTA, algorithm=ALGORITHM, framework=FRAMEWORK)
            if isinstance(result, dict):
                node_nm_map = result
                nm = float(max(node_nm_map.values()))
                print(f"[Privacy] Calculated per-node Noise Multipliers")
            else:
                nm, m = result
                print(f"[Privacy] Calculated Noise Multiplier: {nm:.4f}")

    params = {
        "timestamp": timestamp,
        "device": str(DEVICE),
        "DATASET": DATASET,
        "GRAPH": GRAPH,
        "num_nodes": num_nodes,
        "num_classes": 10,
        "N_SAMPLES_TOTAL": len(train_dataset),
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_PHYSICAL_BATCH_SIZE": MAX_PHYSICAL_BATCH_SIZE,
        "T_LOCAL_STEPS": T_LOCAL_STEPS,
        "R_ROUNDS": R_ROUNDS,
        "K_GOSSIP": K_GOSSIP,
        "EPSILON": EPSILON,
        "DELTA": DELTA,
        "FRAMEWORK": FRAMEWORK,
        "CLIP_NORM": CLIP_NORM,
        "LR": LR,
        "GPU": GPU,
        "ALGORITHM": ALGORITHM,
        "SET_NM": SET_NM,
        "noise_multiplier": float(nm),
        "samples_per_node": samples_per_node,
        "ENABLE_PRIVACY": ENABLE_PRIVACY,
    }
    if node_nm_map is not None:
        params["noise_multiplier_per_node"] = {str(k): float(v) for k, v in node_nm_map.items()}
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Setup] Output directory: {out_dir}")

    all_indices = np.random.permutation(len(train_dataset))
    node_objects = {}

    print("Initializing global model for consistent starting weights...")
    global_model = MNISTCNN(num_classes=10)
    global_state_dict = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    del global_model

    for i, node_name in enumerate(nodes_list):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node
        indices = all_indices[start_idx:end_idx]

        actual_nm = node_nm_map[node_name] if node_nm_map is not None else nm
        node_objects[node_name] = DecentralizedNode(
            node_id=node_name,
            data_indices=indices,
            dataset=train_dataset,
            noise_multiplier=actual_nm,
            num_classes=10,
            enable_privacy=ENABLE_PRIVACY,
            init_state_dict=global_state_dict,
        )

    print("Starting Decentralized Training...")
    round_mean_accs = []
    for round_idx in tqdm(range(R_ROUNDS), desc="Rounds"):
        print(f"\n--- Round {round_idx + 1}/{R_ROUNDS} ---")

        round_losses = []
        pbar = tqdm(enumerate(node_objects.items()), desc=f"Round {round_idx+1} Train", leave=False)
        for i, (node_name, node_obj) in pbar:
            pbar.set_postfix(node=i)
            loss = node_obj.local_update(T_LOCAL_STEPS, desc=f"  Node {i}")
            round_losses.append(loss)
            torch.cuda.empty_cache()
        print(f"Average Local Loss: {np.mean(round_losses):.4f}")

        weights_snapshot = {}
        for node_name, node_obj in node_objects.items():
            weights_snapshot[node_name] = {
                name: param.detach().cpu().clone()
                for name, param in node_obj.model.named_parameters()
                if param.requires_grad
            }

        degrees = {n: G.degree(n) for n in G.nodes()}

        for node_name in nodes_list:
            neighbors = list(G.neighbors(node_name))
            aggregate_set = neighbors + [node_name]

            new_state_dict = {}
            total_weight = 0.0

            for neighbor_name in aggregate_set:
                weight = 1.0 / (max(degrees[node_name], degrees[neighbor_name]) + 1)
                total_weight += weight
                neighbor_weights = weights_snapshot[neighbor_name]
                for k, v in neighbor_weights.items():
                    if k not in new_state_dict:
                        new_state_dict[k] = weight * v.float()
                    else:
                        new_state_dict[k] += weight * v.float()

            for k in new_state_dict:
                new_state_dict[k] = (new_state_dict[k] / total_weight).to(DEVICE)

            node_objects[node_name].model.load_state_dict(new_state_dict, strict=False)

        accuracies = []
        pbar = tqdm(enumerate(node_objects.items()), desc=f"Round {round_idx+1} Eval", leave=False)
        for i, (node_name, node_obj) in pbar:
            pbar.set_postfix(node=i)
            acc = evaluate_model(node_obj.model, test_loader, DEVICE)
            accuracies.append(acc)
        mean_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        min_acc = np.min(accuracies)
        round_mean_accs.append(mean_acc)
        print(f"[Eval] Avg Acc: {mean_acc:.2f}% | Min: {min_acc:.2f}% | Max: {max_acc:.2f}% | Consensus Gap: {max_acc - min_acc:.2f}%")

        csv_path = os.path.join(out_dir, "accuracy.csv")
        if round_idx == 0:
            with open(csv_path, "w") as f:
                f.write("round,mean_acc,max_acc,min_acc,consensus_gap,avg_loss\n")
        with open(csv_path, "a") as f:
            f.write(f"{round_idx + 1},{mean_acc:.4f},{max_acc:.4f},{min_acc:.4f},{max_acc - min_acc:.4f},{np.mean(round_losses):.4f}\n")
            f.flush()

    print("\nTraining Completed.")
    print(f"\n[Summary] Round Average Accuracies: {[f'{a:.2f}%' for a in round_mean_accs]}")
    print(f"[Summary] Final vs Initial: {round_mean_accs[-1]:.2f}% vs {round_mean_accs[0]:.2f}% (Δ={round_mean_accs[-1] - round_mean_accs[0]:+.2f}%)")


if __name__ == "__main__":
    main()
