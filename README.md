# PNDP Calculator：去中心化学习隐私噪声乘子计算工具

本项目提供一个面向去中心化学习（Decentralized Learning，DL）的差分隐私（Differential Privacy，DP）噪声乘子计算器，并给出基于 MNIST 的去中心化训练示例。

当前实现支持三种隐私分析模式：

- **LDP**：局部差分隐私（Local Differential Privacy），采用较保守的全公开观察假设。
- **PNDP**：成对网络差分隐私（Pairwise Network Differential Privacy），根据攻击节点的平均有效方差计算一个全局噪声乘子。
- **PNDP_strict**：严格的 PNDP 分析，对每个受害节点选择最不利攻击节点，并返回逐节点噪声乘子。

隐私会计支持：

- **RDP**：Rényi 差分隐私（Rényi Differential Privacy）。
- **GDP**：高斯差分隐私（Gaussian Differential Privacy）。

项目的理论背景参考了利用矩阵分解（Matrix Factorization，MF）统一描述去中心化学习算法、攻击者观察以及噪声相关性的研究思路。当前代码主要实现了基于网络拓扑、攻击者观察空间投影和半正定规划（Semidefinite Programming，SDP）的有效方差计算。

> 本仓库属于研究性质实现。正式使用前，应核对训练过程、采样方式、通信矩阵和隐私会计假设是否完全一致。

---

## 主要功能

1. 根据隐私预算$(\varepsilon, \delta)$计算噪声乘子 `noise_multiplier`。
2. 支持 RDP 和 GDP 两种会计框架。
3. 支持普通 LDP、全局 PNDP 和逐节点 PNDP_strict。
4. 根据 `networkx.Graph` 自动构建随机游走矩阵。
5. 构造攻击节点的观察矩阵，并计算其行空间投影矩阵。
6. 使用 SDP 求解不同攻击者—受害者组合下的有效方差。
7. 提供 MNIST、卷积神经网络（CNN）和 Opacus 的去中心化训练示例。
8. 输出实验参数和逐轮准确率，便于后续绘图和比较。

---

## 项目结构

```text
.
├── pndp_calculator/
│   ├── __init__.py
│   ├── api.py
│   ├── core_accountant.py
│   └── topology_analyzer.py
├── example.py
├── train_example_mnist.py
├── noise_multiplier.png
├── average_acc.png
└── README.md
```

各文件作用如下：

- `api.py`：提供统一接口 `PNDPAccountant`。
- `core_accountant.py`：实现 RDP、GDP 及部分采样 RDP 的基础会计函数。
- `topology_analyzer.py`：构建通信矩阵、攻击者观察矩阵、投影矩阵，并通过 SDP 计算有效方差。
- `example.py`：演示 LDP、PNDP 和 PNDP_strict 的噪声乘子计算。
- `train_example_mnist.py`：使用 MNIST、PyTorch 和 Opacus 进行去中心化训练。

---

## 安装依赖

基础隐私会计依赖：

```bash
cd pndp_calculator
pip install -r requirements_pndp.txt
```

运行 MNIST 训练示例还需要：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 运行噪声乘子计算示例

```bash
python example.py
```

示例首先计算不依赖图结构的 LDP，然后在 `florentine_families_graph` 图上计算 PNDP 和 PNDP_strict。

```python
import networkx as nx
from pndp_calculator import PNDPAccountant

acc = PNDPAccountant(
    N_samples=4490,
    batch_size=16,
    T_local_steps=10,
    R_rounds=50,
    K_gossip=1,
)

# LDP 不要求设置图
noise_multiplier, best_alpha = acc.get_noise_multiplier(
    target_epsilon=8.0,
    target_delta=1e-5,
    algorithm="LDP",
    framework="RDP",
)

# PNDP 和 PNDP_strict 必须设置通信图
G = nx.florentine_families_graph()
acc.set_graph(G)

noise_multiplier, mu = acc.get_noise_multiplier(
    target_epsilon=8.0,
    target_delta=1e-5,
    algorithm="PNDP",
    framework="GDP",
)

node_noise_multipliers = acc.get_noise_multiplier(
    target_epsilon=8.0,
    target_delta=1e-5,
    algorithm="PNDP_strict",
    framework="RDP",
)
```

### 2. 运行 MNIST 去中心化训练

启用 PNDP 与 GDP：

```bash
ENABLE_PRIVACY=true \
ALGORITHM=PNDP \
FRAMEWORK=GDP \
EPSILON=8.0 \
DELTA=1e-5 \
python train_example_mnist.py
```

启用逐节点噪声乘子：

```bash
ENABLE_PRIVACY=true \
ALGORITHM=PNDP_strict \
FRAMEWORK=RDP \
python train_example_mnist.py
```

直接指定噪声乘子并跳过自动计算：

```bash
ENABLE_PRIVACY=true \
SET_NM=1.2 \
python train_example_mnist.py
```

---

## `PNDPAccountant` 参数

```python
PNDPAccountant(
    N_samples,
    batch_size,
    T_local_steps,
    R_rounds,
    K_gossip=1,
)
```

| 参数              | 含义                                           |
| ----------------- | ---------------------------------------------- |
| `N_samples`     | 单个节点的本地样本数量，不是所有节点的样本总数 |
| `batch_size`    | 逻辑批大小                                     |
| `T_local_steps` | 每轮本地更新步数                               |
| `R_rounds`      | 总通信轮数                                     |
| `K_gossip`      | 每轮 Gossip（邻居通信/聚合）次数               |

---

## 支持的算法与返回值

| `algorithm`   | 是否需要图 | 返回值                            | 当前实现含义                                               |
| --------------- | ---------: | --------------------------------- | ---------------------------------------------------------- |
| `LDP`         |         否 | `(noise_multiplier, metric)`    | 使用标准 LDP 有效方差                                      |
| `PNDP`        |         是 | `(noise_multiplier, metric)`    | 对每个攻击者计算其对其他节点的平均有效方差，再取最坏攻击者 |
| `PNDP_strict` |         是 | `{node_name: noise_multiplier}` | 对每个受害节点取最坏攻击者，返回逐节点结果                 |

`metric` 的含义取决于 `framework`：

- `framework="RDP"`：返回搜索得到的最佳 Rényi 阶数 `alpha`。
- `framework="GDP"`：返回对应的 GDP 参数 `mu`。

---

## 方法概述

本文代码参考论文 *Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization* 中的统一隐私分析框架，将去中心化学习过程表示为矩阵形式，并重点实现了 **DP-D-SGD 在 LDP（本地差分隐私）和 PNDP（成对网络差分隐私）信任模型下的噪声标定**。需要强调的是：当前项目实现的是论文中的**隐私会计思想及其在独立噪声 DP-D-SGD 上的应用**，并不是论文提出的完整 MAFALDA-SGD 算法复现。

### 1. 论文原文的统一分析框架

论文将训练期间所有与数据相关的梯度堆叠为矩阵 $G$，将注入的高斯噪声堆叠为矩阵 $Z$，并把攻击者能够获得的信息统一表示为：

$$
O_{\mathcal A}=AG+BZ.
$$

其中：

- $A$ 描述梯度如何经过本地更新、网络传播和消息观测进入攻击者视图；
- $B$ 描述噪声如何进入攻击者视图；
- $O_{\mathcal A}$ 表示攻击者在整个训练过程中的可观测信息。

当存在矩阵 $C$ 使得 $A=BC$ 时，可以使用 Matrix Factorization（矩阵分解）机制分析隐私。论文给出的广义敏感度为：

$$
\operatorname{sens}_{\Pi}(C;B)=\max_{G\simeq_{\Pi}G'}\left\|C(G-G')\right\|_{B^\dagger B},
$$

其中 $B^\dagger B$ 是 $B$ 行空间上的正交投影。该投影会删除攻击者无法从消息中恢复的梯度方向，因此通常能够得到比“所有消息均公开”的 LDP 分析更小的隐私损失。参与模式 $\Pi$ 则描述同一条样本在多个训练步骤中可能出现的位置。

在不同信任模型下，攻击者视图不同：

| 信任模型 | 论文中的攻击者能力                                                           |
| -------- | ---------------------------------------------------------------------------- |
| LDP      | 假设网络中的全部通信消息均可被攻击者观察                                     |
| PNDP     | 攻击者是网络中的某个节点，只观察自己发送或接收的消息，并已知自己的梯度与噪声 |
| SecLDP   | 除消息外，还根据攻击者掌握的秘密噪声构造条件隐私保证                         |

当前代码只实现了前两种情况，不包含 SecLDP。

### 2. 论文中的 DP-D-SGD 与当前实现

对于使用独立高斯噪声的 DP-D-SGD，论文令噪声相关矩阵为：

$$
C=I.
$$

在 PNDP 模型下，攻击者 $a$ 只能观察与自己相连的局部消息。设对应的观测矩阵为 $B_a$，则隐私敏感度取决于：

$$
P_a=B_a^\dagger B_a.
$$

论文在 DP-D-SGD 的 PNDP 实验中使用的核心量正是攻击者观测矩阵行空间的投影，即：

$$
\operatorname{sens}_{\Pi}^{2}\lesssim\max_{\pi\in\Pi}\sum_{s,t\in\pi}\left|(P_a)_{s,t}\right|.
$$

当前实现沿用了这一核心思路，但没有直接使用元素绝对值之和作为最终界，而是针对每个攻击者、每个受保护节点和每种循环参与位置，进一步求解半正定规划，以计算对应的有效方差乘数。

### 3. 当前代码中的隐私会计流程

代码首先根据网络拓扑构造 Metropolis–Hastings（梅特罗波利斯–黑斯廷斯）形式的 Gossip（邻居平均）矩阵 $W$：

$$
W_{ij}=
\frac{1}{\max(d_i,d_j)},\qquad (i,j)\in E,
$$

对角元素设置为：

$$
W_{ii}=1-\sum_{j\ne i}W_{ij}.
$$

随后计算 $I,W,W^2,\ldots$，以描述某个节点产生的更新经过多轮邻居平均后，对其他节点可观测消息的影响。

对于每个攻击者节点 $a$，实现依次执行：

1. 将攻击者自身及其直接邻居视为可观测节点；
2. 根据 $W$ 的幂构造攻击者的消息观测矩阵 $B_a$；
3. 将攻击者自身对应的梯度列置零，表示攻击者自己的梯度不是需要保护的未知信息；
4. 通过奇异值分解计算行空间投影：

$$
P_a=B_a^\dagger B_a;
$$

5. 根据循环参与模式，从同一受害节点在不同时间的梯度列中提取子矩阵 $H$；
6. 求解以下半正定规划：

$$
\begin{aligned}
\max_X\quad & \operatorname{tr}(HX),\\
\text{s.t.}\quad & X\succeq 0,\\
& \operatorname{diag}(X)\leq 1.
\end{aligned}
$$

该问题的最优值作为攻击者 $a$ 对受害节点 $u$ 的有效方差乘数，记为：

$$
M_{a,u}.
$$

最终得到攻击者—受害者矩阵：

$$
M=
\begin{bmatrix}
M_{1,1} & \cdots & M_{1,n}\\
\vdots & \ddots & \vdots\\
M_{n,1} & \cdots & M_{n,n}
\end{bmatrix}.
$$

这一过程对应论文中“攻击者观测矩阵的行空间决定隐私损失”的思想，但半正定规划求解和后续聚合方式属于当前项目的具体实现。

### 4. 循环参与模式的对应关系

论文使用参与模式 $\Pi$ 描述同一条样本可能在哪些梯度步骤中出现。当前实现将本地数据访问近似为固定间隔的循环参与模式，并定义：

$$
b=\left\lceil\frac{N_{\text{samples}}}{\text{batch size}}\right\rceil,
$$

表示一条样本两次参与训练之间的近似步数；总本地更新步数为：

$$
S=R_{\text{rounds}}T_{\text{local}},
$$

同一条样本最多参与：

$$
k=\left\lceil\frac{S}{b}\right\rceil
$$

次。因此，代码检查的参与位置近似为：

$$
\pi(s)=\{s,s+b,s+2b,\ldots,s+(k-1)b\}.
$$

这对应论文中的 $(k,b)$ 循环参与模式，但当前训练脚本没有显式强制数据严格按照该固定周期出现，因此它是会计器采用的建模假设，而不是训练代码直接保证的采样序列。

### 5. 三种会计模式的含义

#### LDP

LDP 假设攻击者可以观察全部通信信息，因此不利用网络局部可见性带来的隐私放大。当前实现不显式构造完整的全局消息矩阵，而是根据每轮内同一条样本出现的次数计算有效方差：

$$
v_{\mathrm{LDP}}=\max_{\text{offset}}\sum_{r=1}^{R}\frac{z_r^2}{T},
$$

其中 $z_r$ 是某个参与偏移下，该样本在第 $r$ 轮本地更新中的出现次数。

#### PNDP_strict

对每个受保护节点 $u$，取所有其他攻击者中的最坏情况：

$$
v_u=\max_{a\ne u}M_{a,u}.
$$

然后为每个节点分别标定噪声：

$$
\sigma_u=\sigma_{\text{base}}\sqrt{v_u}.
$$

因此，`PNDP_strict` 返回每个节点独立的噪声乘数。它对应较严格的受害者中心分析，即每个节点都针对自己的最坏攻击者获得保护，能完全满足DP的保证。

#### PNDP

代码首先对每个攻击者计算其针对其他节点的平均有效方差：

$$
\bar v_a=
\frac{1}{n-1}\sum_{u\ne a}M_{a,u},
$$

再取：

$$
v_{\mathrm{PNDP}}=\max_a\bar v_a.
$$

所有节点使用同一个噪声乘数：

$$
\sigma_{\mathrm{PNDP}}=\sigma_{\text{base}}\sqrt{v_{\mathrm{PNDP}}}.
$$

该模式参考了*Muffliato: Peer-to-Peer Privacy Amplification for Decentralized Optimization and Averaging*.，便于使用统一噪声进行训练，但它是当前代码定义的“最坏攻击者平均风险”聚合方式，其中隐含了越近的邻居越值得信任的假设，并不完全满足DP的保证。

### 6. 从有效方差到噪声乘数

在得到有效方差乘数 $v$ 后，代码先求解单位有效方差下满足目标 $(\varepsilon,\delta)$ 的基础噪声 $\sigma_{\text{base}}$，再进行缩放：

$$
\sigma_{\text{final}}=\sigma_{\text{base}}\sqrt{v}.
$$

当前实现支持两种会计框架：

- RDP（Rényi Differential Privacy，Rényi 差分隐私）：遍历多个 Rényi 阶数 $\alpha$，选择产生最小 $\varepsilon$ 的阶数；
- GDP（Gaussian Differential Privacy，高斯差分隐私）：先求满足目标 $(\varepsilon,\delta)$ 的 $\mu$，再由 $\mu=1/\sigma$ 反解噪声。

论文主要以 GDP 推导统一理论，并指出 GDP 可以转换为 RDP 或 $(\varepsilon,\delta)$-DP；当前实现则直接提供 RDP 和 GDP 两种数值反解接口。

---

## MNIST 训练配置

`train_example_mnist.py` 通过环境变量读取主要配置。

| 环境变量                    |                  默认值 | 说明                                  |
| --------------------------- | ----------------------: | ------------------------------------- |
| `DATASET`                 |               `MNIST` | 当前代码实际只实现 MNIST              |
| `GRAPH`                   | `florentine_families` | 当前支持的图名称                      |
| `BATCH_SIZE`              |                 `256` | 逻辑批大小                            |
| `MAX_PHYSICAL_BATCH_SIZE` |                 `256` | Opacus 物理批大小上限                 |
| `T_LOCAL_STEPS`           |                  `20` | 每轮本地更新步数                      |
| `R_ROUNDS`                |                  `20` | 训练轮数                              |
| `K_GOSSIP`                |                   `1` | 会计器中的每轮 Gossip 次数            |
| `EPSILON`                 |                 `3.0` | 目标隐私预算\(\varepsilon\)           |
| `DELTA`                   |                `1e-5` | 目标隐私预算\(\delta\)                |
| `CLIP_NORM`               |                   `1` | 梯度裁剪阈值                          |
| `LR`                      |                `1e-3` | 学习率                                |
| `GPU`                     |                   `0` | 通过`CUDA_VISIBLE_DEVICES` 指定 GPU |
| `FRAMEWORK`               |                 `GDP` | `RDP` 或 `GDP`                    |
| `ALGORITHM`               |                `PNDP` | `LDP`、`PNDP` 或 `PNDP_strict`  |
| `ENABLE_PRIVACY`          |                `True` | 是否启用 Opacus 隐私训练              |
| `SET_NM`                  |                `None` | 是否手动指定统一噪声乘子              |
| `CACHE_NOISE`             |                `True` | 是否从已有缓存读取噪声乘子            |
| `OUT_DIR`                 |                自动生成 | 实验输出目录                          |

---

## 实验输出

默认输出目录位于：

```text
exps/<时间戳>_<数据集>_<图>_<参数组合>/
```

主要文件包括：

- `params.json`：保存实验参数、统一噪声乘子或逐节点噪声乘子。
- `accuracy.csv`：保存每轮平均准确率、最大准确率、最小准确率、一致性差距和平均损失。

`accuracy.csv` 的列为：

```text
round,mean_acc,max_acc,min_acc,consensus_gap,avg_loss
```

---

## 示例结果

#### 不同隐私预算下的噪声乘子

![不同隐私预算下的噪声乘子](./noise_multiplier.png)

在图示配置下，随着 $\varepsilon$ 增大，达到目标隐私预算所需的噪声乘子整体下降。PNDP 利用了攻击者局部观察和网络混合带来的隐私放大，因此所需噪声低于较保守的 LDP。PNDP_strict 按节点最坏攻击者进行校准，通常比平均化的 PNDP 更保守。

### 不同方法的平均准确率

![不同方法的平均准确率](./average_acc.png)

在图示实验中，三种方法的平均准确率均随训练轮数上升。PNDP 使用较低噪声乘子，因此获得了更高的准确率；PNDP_strict 与 LDP 的曲线较为接近。右下角插图放大了训练后期的差异。

> 上述结论仅对应当前图片中的实验设置，不能直接外推到其他图结构、模型、数据划分或隐私预算。

---

## 参考文献

> Aurelien Bellet, Edwige Cyffers, Davide Frey, Romaric Gaudel, Dimitri Lereverend, François Taïani.
> *Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization*. ICLR 2026.
> arXiv:2510.17480.

> Edwige Cyffers, Mathieu Even, Aurélien Bellet, Laurent Massoulié.
> *Muffliato: Peer-to-Peer Privacy Amplification for Decentralized Optimization and Averaging*. NeurIPS 2022.
> arXiv:2206.05091.

---

## 下一步

- [ ] 增加更多图结构和数据集。
- [ ] 保存并可视化完整的攻击者—受害者有效方差矩阵 `M`。
- [ ] 增加 SDP 结果缓存，避免重复计算。
- [ ] 结合更好的采样隐私放大算法，如Bnb
