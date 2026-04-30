# Radar Radiation Slot Simulation — Design Spec

- **Date**: 2026-04-30
- **Source requirement**: `docs/demos/radar-radiation/radar-radiation-slot-simulation.md`
- **Framework**: DeepXDE（本仓库）
- **Status**: Approved (brainstorming)

## 0. 决策摘要（基于 brainstorming 5 个问题的回答）

| 决策项 | 选择 |
| --- | --- |
| Q1 项目结构 | **A**：子项目目录 `examples/radiation_slot/`，按需求文档完整 6 模块 |
| Q2 FDFD 实现深度 | **B**：教学级（五点差分 + PEC + sin 端口激励 + 一阶 Mur + S 参数 + NTFF 远场） |
| Q3 PINN 频率策略 | **B**：单频训练 + 热启动迁移 |
| Q4 开发节奏 | **A**：MVP 优先，先 15 GHz 单频跑通全链路 |
| Q5 测试/后端/CI | 全部 **A**：完整 TDD（FDFD/几何/后处理）+ PyTorch 默认 + 不接入 CI |

## 1. 架构总览

### 1.1 计算域

```
       ┌──────────── 顶部 Mur 区（开放空间）─────────────────┐
       │                                                  │
       │              辐射槽（开口）                         │
       │       ┌──────────────────┐                        │
   ────┴───────┤                  ├────────────────────────┤
   |  Port 1  |       波导内腔    |        Port 2          |
   ────────────────────────────────────────────────────────┘
       PEC 上下壁 + Port 端口截面 + 顶部开放
```

- **域 1（波导）**：`Rectangle([0, 0], [L_wg, b])`，b=0.51 cm
- **域 2（缓冲区）**：`Rectangle([x_slot-Δ, b], [x_slot+L_slot+Δ, b+H_buf])`
- **计算域** = `域1 ∪ 域2`（CSGUnion，构造辐射槽连通拓扑）
- 一个 `boundary_marker(x) → label` 函数同时供 FDFD 和 PINN 使用

### 1.2 数据流

```
config(15GHz) ──┬──► geometry.build_dde_geometry() ─┬─► PINN solver ─► E_pinn(x,y)
                │                                    │
                └──► geometry.fdfd_grid() ───────────┴─► FDFD solver ─► E_fdfd(x,y)
                                                                   │
                                          common postprocess ◄─────┤
                                          (NTFF + S-params)        │
                                                                   ▼
                                                              comparator
```

## 2. 模块与接口

### 2.1 目录结构

```
examples/radiation_slot/
├── __init__.py                    # 公开 RadiationSlotGeometry, FDFDSolver, PINNSolver
├── geometry.py                    # 几何 + 网格 + 边界标记
├── fdfd_solver.py                 # FDFD 求解器
├── pinn_solver.py                 # DeepXDE PINN 求解器
├── postprocess.py                 # NTFF 远场 + S 参数 + 可视化
├── comparator.py                  # 多指标对比
├── functions.py                   # 共享工具
├── main.py                        # argparse 入口
├── conf/
│   └── radiation_slot.yaml        # 默认参数（YAML，无 Hydra）
├── tests/                         # pytest 单元测试
└── outputs/                       # gitignore，运行时产物
```

### 2.2 接口签名（按依赖顺序）

```python
# geometry.py
@dataclass
class RadiationSlotGeometry:
    # NOTE: 原需求文档 waveguide_width=1.02 是物理波导宽 a。本仿真把 "waveguide_width"
    # 重新定义为沿传播方向 z 的长度（投影到 2D 后的 x），需要 ~3-5λ 才能看到入射/反射
    # 驻波结构。15 GHz 时 λ≈2cm，故取 4cm。波导横截面高度 b=0.51cm 仍为 TE10 截止决定。
    waveguide_width: float = 4.0      # cm，沿传播方向（2D 中的 x）
    waveguide_height: float = 0.51    # cm，TE10 模 cutoff 决定（即物理 b）
    slot_length: float = 1.5
    slot_width: float = 0.16
    slot_position: float = 0.5
    buffer_height: float = 1.5
    medium_epsilon: float = 1.0
    medium_mu: float = 1.0

    def build_dde_geometry(self) -> dde.geometry.Geometry: ...
    def fdfd_grid(self, mesh_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (X, Y, mask)；mask=True 表示属于计算域"""
    def boundary_marker(self, x: np.ndarray) -> np.ndarray:
        """点向量 (N,2) → 标签 (N,) ∈ {'pec','port_in','port_out','radiation','interior'}"""

# fdfd_solver.py
class FDFDSolver:
    def __init__(self, geometry, frequency_ghz: float, mesh_size: float = 0.05): ...
    def solve(self) -> dict:
        """返回 {'X':..,'Y':..,'Ez':复值场, 'mask':...}"""

# pinn_solver.py
class PINNSolver:
    def __init__(self, geometry, frequency_ghz: float,
                 num_domain=8000, num_boundary=800, num_test=40000): ...
    def train(self, iterations=15000, lr=1e-3,
              restore_from: str | None = None,
              save_path: str | None = None): ...
    def predict_on_grid(self, X, Y, mask) -> np.ndarray: ...

# postprocess.py
def near_to_far_field(Ez_aperture, x_aperture, k0, theta) -> np.ndarray: ...
def s_parameters(Ez, geometry, k0) -> tuple[complex, complex]: ...   # S11, S21
def plot_field(Ez, X, Y, mask, save_path): ...
def plot_radiation_pattern(theta, pattern_db, save_path): ...

# comparator.py
def field_metrics(ref, pred, mask) -> dict:
    """{'l2_relative','linf','corrcoef'} —— 仅在 mask 内"""
def pattern_metrics(theta, p_ref, p_pred) -> dict:
    """{'main_lobe_deg_diff','side_lobe_db_diff'}"""
```

### 2.3 边界条件统一表

| 区域标签 | 物理含义 | FDFD 处理 | PINN 处理 |
| --- | --- | --- | --- |
| `pec` | 金属壁 E_z=0 | 行替换为 1，rhs=0 | `DirichletBC(component=0/1, value=0)` |
| `port_in` | TE10 入射 + Mur | ∂u/∂n−jku=−2jk·sin(πy/b) | `RobinBC` 对应公式 |
| `port_out` | Mur 出射 | ∂u/∂n+jku=0 | `RobinBC`（∂Re/∂n=−k·Im, ∂Im/∂n=+k·Re）|
| `radiation` | Mur（4 法向） | 同上 | 同上，按法向分 4 个 RobinBC |

## 3. 数学与数值方案

### 3.1 控制方程

\[
\nabla^2 E_z + k_0^2 \varepsilon_r E_z = 0, \quad k_0 = \frac{2\pi f}{c}
\]

DeepXDE 不支持复数，分解 `E_z = u + jv`，得到两条耦合实方程（域内解耦，仅 Mur 边界耦合）。

### 3.2 FDFD 五点差分

\[
\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2} + k_0^2 \varepsilon_r u_{i,j} = 0
\]

直接对**复数** `Ez` 装配 `scipy.sparse.csr_matrix(complex)`，用 `spsolve`。

**TE10 入射端口**（x=0）：
入射场 `E_z^{inc}(y) = sin(πy/b)`，等效 Mur：
\[
\frac{\partial E_z}{\partial x} - j k_0 E_z = -2 j k_0 E_z^{inc}(y)
\]

### 3.3 PINN 残差

```python
def pde(x, y):
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    k2 = self.k0 ** 2 * self.eps_r
    return [u_xx + u_yy + k2 * y[:, 0:1],
            v_xx + v_yy + k2 * y[:, 1:2]]
```

**网络**：`dde.nn.FNN([2, 256, 256, 256, 256, 2], "tanh", "Glorot uniform")`，soft BC + `loss_weights`。

**训练**：Adam 15k @ lr=1e-3 → L-BFGS（最多 5k）。热启动用 `model.train(model_save_path=...)` + `model.restore(...)`。

### 3.4 S 参数

端口截面 `x = x_port` 处，TE10 模剖面 `φ(y) = sin(πy/b)`，归一 `N = b/2`：

\[
S_{11} = \frac{1}{N}\int_0^b \big(E_z(x_{in},y) - E_z^{inc}(y)\big) \cdot \overline{\varphi(y)} \, dy
\]
\[
S_{21} = \frac{1}{N}\int_0^b E_z(x_{out},y) \cdot \overline{\varphi(y)} \, dy
\]

数值用 `np.trapz`。

### 3.5 远场方向图（NTFF, 2D Stratton-Chu）

辐射开口 y=b，x∈[x_slot, x_slot+L_slot]：
\[
E_\text{far}(\theta) \propto \int_{x_\text{slot}} E_z(x, b) \, e^{-jk_0 x \sin\theta} \, dx
\]

θ ∈ [−90°, 90°]，扫 360 角；归一化转 dB。

## 4. 测试策略

### 4.1 单元测试矩阵

| 文件 | 测试用例 | 验证方法 |
| --- | --- | --- |
| `test_geometry.py` | `test_csg_inside_outside` | 已知点的 `geom.inside(p)` 正确 |
| | `test_boundary_marker_coverage` | 1000 边界点全部命中 4 标签之一 |
| | `test_boundary_marker_pec_zero_overlap_with_radiation` | 标签互斥 |
| | `test_fdfd_grid_shape_and_mask` | 网格 shape + mask 面积一致 |
| `test_fdfd_solver.py` | `test_uniform_medium_plane_wave` | **关键**：去掉槽，断言场 ≈ `sin(πy/b)·exp(−jk_x x)`，L2 ≤ 2% |
| | `test_pec_dirichlet_zero` | PEC 节点 \|E_z\| < 1e-10 |
| | `test_mur_low_reflection` | Mur 反射 \|R\| ≤ 5% |
| `test_postprocess.py` | `test_s_parameters_pure_transmission` | \|S11\|≈0、\|S21\|≈1 |
| | `test_ntff_dipole_known_pattern` | 偶极子 cos(θ) 方向图 |
| `test_comparator.py` | `test_metrics_identity` | ref==pred → l2=0、corr=1 |
| | `test_metrics_known_diff` | 已知偏差数值正确 |
| `test_pinn_solver.py` | `test_smoke_train_50_steps` | 构建+训 50 步不报错 |

### 4.2 端到端验证标准（main.py --mode validate）

| 指标 | 阈值 | 测量条件 |
| --- | --- | --- |
| 场分布 L2 相对误差 | ≤ **5%** | 15 GHz，FDFD vs PINN，mask 内 |
| L∞ | ≤ 0.15·max(\|E_fdfd\|) | 同上 |
| 相关系数 | ≥ 0.98 | 同上 |
| 远场主瓣方向偏差 | ≤ **2°** | NTFF 归一化方向图 |
| \|S11\| 偏差 | ≤ **2 dB** | 单频，MVP 阶段宽松 |
| PINN 最终 PDE loss | < 1e-3 | Adam+L-BFGS 后 |

> 需求文档原写 \|S11\| ≤ 1 dB，但 MVP 用一阶 Mur（非 PML），FDFD 自身有 ~5% 反射，1 dB 太严，改为 2 dB；M6 扫频后再收紧。

## 5. 里程碑

| M | 内容 | 完成判据 |
| --- | --- | --- |
| **M1** 基础设施 | geometry.py + functions.py + tests/test_geometry.py + conf/radiation_slot.yaml | `pytest tests/test_geometry.py` 全过 |
| **M2** FDFD 基线 | fdfd_solver.py + tests/test_fdfd_solver.py | 平面波 ≤ 2%、Mur 反射 ≤ 5%、PEC=0 |
| **M3** 后处理 | postprocess.py + comparator.py + 单元测试 | NTFF/Sparam 单元测试全过 |
| **M4** PINN 求解器 | pinn_solver.py + 烟雾测试 | 50 步训练不报错；15 GHz Adam+L-BFGS 后 PDE loss < 1e-3 |
| **M5** 端到端单频对比 | main.py（`--mode single --freq 15`） | 满足 §4.2 验证标准 |
| **M6** 扩展：13 频点扫描 | main.py `--mode scan` + 热启动管理 + S 参数曲线 | 曲线平滑；GPU 总训练 ≤ 30 min |

**M1-M5 = MVP**；M6 在 MVP 跑通后再启动。

## 6. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
| --- | --- | --- | --- |
| PINN 在大计算域 + 高频上不收敛 | 中 | 高 | 缩 waveguide_width=4→2cm；启用 sin/Fourier 激活 |
| 一阶 Mur 反射污染对比 | 中 | 中 | M2 单元测试量化；超标升二阶 Mur |
| 多 RobinBC + 多 component 的 loss 平衡敏感 | 高 | 中 | `loss_weights` 网格搜索 fallback |
| FDFD 复数稀疏矩阵慢 | 低 | 低 | `spsolve(use_umfpack=True)` |
| 热启动在 freq 跳变时仍需较多 iter | 中 | 低 | M6 时调，不影响 MVP |

## 7. 范围与 YAGNI 约定

**包含**：MVP（M1-M5）+ 扩展（M6）。

**显式排除**：
- ❌ 3D 仿真（保持 2D Helmholtz）
- ❌ Hydra 配置（用纯 YAML + argparse）
- ❌ Mayavi/VTK 可视化（仅 matplotlib，VTU 列为可选）
- ❌ PML（MVP 用一阶 Mur，足够对比目的）
- ❌ 多输入 DeepONet / fPINN 等高级 PINN 变体
- ❌ 频率参数化 PINN（Q3 已排除）
- ❌ CI 集成（Q5c-A 排除）
