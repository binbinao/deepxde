# DeepXDE 用户文档

> 面向使用 DeepXDE 解决科学计算问题的科研人员与算法工程师。你会学到：安装与后端选择、核心对象（几何、PDE、网络、模型）、工作流模板、进阶技巧与故障排查。

---

## 1. DeepXDE 能做什么

- 使用 **PINN（物理信息神经网络）** 求解 ODE / PDE / IDE / fPDE / sPDE 的 **正问题** 和 **反问题**。
- 使用 **DeepONet / MIONet / Fourier-DeepONet** 学习算子映射。
- 使用 **Physics-Informed DeepONet** 融合物理先验做算子学习。
- 使用 **MFNN** 结合多精度数据。
- 使用 **硬约束**（hPINN）处理设计优化 / 拓扑优化问题。
- 支持 **不确定性量化**（MC dropout）、**复杂几何**（CSG）、**自适应采样**（RAR）、**混合精度**、**多 GPU 数据并行**（TF1 + Horovod）。

---

## 2. 安装

### 2.1 前置依赖

DeepXDE 至少需要以下任一深度学习后端：

| 后端 | 最低版本 | 额外依赖 |
| --- | --- | --- |
| TensorFlow 1.x（`tensorflow.compat.v1`） | TensorFlow ≥ 2.7.0 | 若 TF ≥ 2.16 使用 Keras 3，需安装 `tf-keras` 并设置 `TF_USE_LEGACY_KERAS=1` |
| TensorFlow 2.x | TensorFlow ≥ 2.3.0 | `tensorflow-probability ≥ 0.11.0` |
| PyTorch | PyTorch ≥ 2.0.0 | 无 |
| JAX | 最新版 | `jax` + `flax` + `optax` |
| PaddlePaddle | ≥ 2.6.0 | 无 |

基础依赖（`pyproject.toml`）：`matplotlib`、`numpy`、`scikit-learn`、`scikit-optimize ≥ 0.10.2`、`scipy`。

### 2.2 安装 DeepXDE

```bash
# PyPI
pip install deepxde

# conda-forge
conda install -c conda-forge deepxde

# 开发者（需要源码二次修改）
git clone https://github.com/lululxvi/deepxde.git
cd deepxde && pip install -e .
```

### 2.3 Docker（GPU）

```bash
# 官方镜像
nvidia-docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 \
    pescapil/deepxde:latest
```

---

## 3. 选择后端

DeepXDE 按以下优先级决定运行时后端：

1. 环境变量 `DDE_BACKEND`（推荐）。
2. `~/.deepxde/config.json` 中的 `{"backend": "..."}`。
3. 自动检测（会在首次成功后写回 `config.json`）。
4. 兜底：尝试安装 `paddle`。

```bash
# 临时指定
DDE_BACKEND=pytorch python my_pde.py

# 永久配置
python -m deepxde.backend.set_default_backend pytorch
```

合法值：`tensorflow.compat.v1` / `tensorflow` / `pytorch` / `jax` / `paddle`。

### 3.1 如何选

| 关注点 | 建议 |
| --- | --- |
| 特性覆盖度 | PaddlePaddle ≈ TF1 > TF2 ≈ PyTorch > JAX |
| 首选性能 | 通常 TF2 > PyTorch > TF1（依硬件 / 问题而变，建议实测） |
| 代码最简洁 | PyTorch（无需手动选择 `tf/torch/jnp/paddle` 前缀） |
| 最容易复现论文 | TF1（早期 DeepXDE 论文多用该后端） |
| 二次编译 / XLA 加速 | JAX / TF（GPU 默认启用 XLA） |

---

## 4. 全局配置（`dde.config`）

```python
import deepxde as dde

# 浮点精度：'float16' | 'float32'（默认） | 'float64' | 'mixed'
dde.config.set_default_float("float64")

# 全局随机种子（同时设定 Python/NumPy/后端）
dde.config.set_random_seed(2024)

# 自动微分模式：'reverse'（默认） | 'forward'
dde.config.set_default_autodiff("reverse")

# XLA JIT（仅 TF / JAX 有效）
dde.config.disable_xla_jit()
```

- 需要数值稳定（尤其 L-BFGS）时优先用 `float64`。
- 混合精度参考 Hayford 等 2024（CMAME）论文：TF 下自动启用 `mixed_float16`；PyTorch 下前后向 FP16、权重 FP32。

---

## 5. 核心对象一图流

```
┌────────────────┐    ┌──────────────┐     ┌──────────────┐
│ dde.geometry.* │    │ dde.icbc.*   │     │ dde.nn.*     │
│  Interval/..   │───▶│  BC / IC     │────▶│ FNN / PFNN / │
└────────────────┘    └──────────────┘     │ DeepONet /...│
         │                                 └──────────────┘
         ▼
   ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
   │ dde.data.*   │──▶ │ dde.Model    │──▶ │ train/predict│
   │ PDE / TimePDE│    │ compile/fit  │    │ save/restore │
   └──────────────┘    └──────────────┘    └─────────────┘
```

你要做的事情通常可以拆成：
1. 定义几何域 & 时空域（`dde.geometry`）。
2. 定义 PDE 残差（`dde.grad.jacobian`、`dde.grad.hessian`）。
3. 定义 IC / BC（`dde.icbc`）。
4. 组装数据对象（`dde.data.PDE` / `TimePDE` / ...）。
5. 选择网络（`dde.nn.FNN` / `DeepONet` / ...）。
6. `model = dde.Model(data, net); model.compile(...); model.train(...)`。
7. `model.predict(x)` / `model.predict(x, operator=pde)` 得到解 / 残差。

---

## 6. 几何：`dde.geometry`

### 6.1 原语

| 维度 | 类 | 构造 |
| --- | --- | --- |
| 1D | `Interval(l, r)` | 区间 |
| 2D | `Rectangle(xmin, xmax)`、`Disk(center, radius)`、`Ellipse`、`Triangle`、`Polygon`、`StarShaped` | 常见 2D 形状 |
| 3D | `Cuboid(xmin, xmax)`、`Sphere(center, radius)` | 立方体 / 球 |
| ND | `Hypercube(xmin, xmax)`、`Hypersphere(center, radius)` | 高维 |
| 点云 | `PointCloud(points)` | 任意几何 |

### 6.2 CSG 组合

```python
geom1 = dde.geometry.Rectangle([0, 0], [2, 2])
geom2 = dde.geometry.Disk([1, 1], 0.5)
geom  = geom1 - geom2         # difference, 等价 dde.geometry.CSGDifference(geom1, geom2)
geom  = geom1 | geom2         # union
geom  = geom1 & geom2         # intersection
```

### 6.3 时空域

```python
space = dde.geometry.Interval(-1, 1)
time  = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(space, time)  # x 与 t 拼接到一个 2D 输入
```

### 6.4 采样

```python
X = geom.random_points(1000, random="Hammersley")  # pseudo / LHS / Halton / Hammersley / Sobol
X = geom.uniform_points(1000, boundary=True)
X_bc = geom.random_boundary_points(100)
```

---

## 7. 边界与初值条件：`dde.icbc`

所有 BC 都是 `BC` 的子类，签名模式固定为：

```python
dde.icbc.DirichletBC(geom, func, on_boundary, component=0)
dde.icbc.NeumannBC(geom, func, on_boundary, component=0)
dde.icbc.RobinBC(geom, func, on_boundary, component=0)    # func(x, y)
dde.icbc.PeriodicBC(geom, component_x, on_boundary, derivative_order=0, component=0)
dde.icbc.OperatorBC(geom, op_func, on_boundary)           # op_func(X, y, X_shape)
dde.icbc.PointSetBC(points, values, component=0)          # 观测数据（反问题用）
dde.icbc.PointSetOperatorBC(points, values, operator)
dde.icbc.Interface2DBC(geom, func, on_boundary1, on_boundary2, ...)

# 时间域初值
dde.icbc.IC(geomtime, func, on_initial, component=0)
```

`on_boundary(x, on_boundary_flag)` 与 `on_initial(x, on_initial_flag)` 返回布尔数组，决定哪些样本点触发该条件。

**常用技巧**：`lambda _, on_boundary: on_boundary` 表示“所有边界点”。

---

## 8. 数据对象：`dde.data`

### 8.1 PDE 正问题

```python
data = dde.data.PDE(
    geom,                # 几何
    pde,                 # callable(x, y) -> residual（可为 list）
    bcs,                 # BC 或 BC 列表
    num_domain=2000,     # 域内采样点
    num_boundary=200,    # 边界采样点
    train_distribution="Hammersley",  # 低偏差序列
    anchors=None,        # 必含的锚定点（反问题常用）
    exclusions=None,     # 需要排除的点
    solution=func_ref,   # 有参考解时传入以启用 metrics
    num_test=2000,       # 用于验证/metric 的点数
    auxiliary_var_function=None,
)
```

### 8.2 TimePDE

```python
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic],
    num_domain=4000, num_boundary=80, num_initial=160,
    train_distribution="pseudo", solution=func, num_test=10000,
)
```

### 8.3 其它

| 类 | 用途 |
| --- | --- |
| `IDE` | 积分-微分方程 |
| `FPDE` / `TimeFPDE` | 分数阶 PDE |
| `PDEOperator` / `PDEOperatorCartesianProd` | PI-DeepONet 的数据 |
| `Triple` / `TripleCartesianProd` | 纯数据 DeepONet（输入函数、评估点、解） |
| `Quadruple` / `QuadrupleCartesianProd` | MIONet |
| `DataSet` / `MfDataSet` / `MfFunc` | 普通数据 / 多精度 |
| `FuncConstraint` | 函数约束 |

---

## 9. PDE 残差：`dde.grad`

```python
from deepxde import grad as dg

dy_x  = dg.jacobian(y, x, i=0, j=0)         # ∂y_0/∂x_0
dy    = dg.jacobian(y, x, i=0)              # 返回 y_0 对所有 x 的梯度行向量
dy_xx = dg.hessian(y, x, component=0, i=0, j=0)  # ∂²y_0/∂x_0∂x_0
```

JAX 下 `jacobian / hessian` 返回 `(value, fn)` 二元组，需要解包：
```python
dy_x, _ = dg.jacobian(y, x, i=0, j=0)
dy_xx, _ = dg.hessian(y, x, i=0, j=0)
```

自动微分模式：默认 `reverse`（反向传播），切换为 `forward`（前向）只需：
```python
dde.config.set_default_autodiff("forward")
```

---

## 10. 网络：`dde.nn`

### 10.1 全连接

```python
# [输入维度, 隐藏宽度, ..., 输出维度]
net = dde.nn.FNN([2] + [50] * 4 + [1], activation="tanh", kernel_initializer="Glorot normal")
```

- `activation`：`tanh / relu / sigmoid / swish / silu / sin / gelu / selu / elu` 等。
- `kernel_initializer`：`Glorot uniform/normal` / `He uniform/normal` / `LeCun ...` / `zeros` / `orthogonal`。
- 可选 `regularization=("l2", 1e-4)`、`dropout_rate=0.1`（PyTorch 支持整条 list）。

### 10.2 并行全连接 PFNN

每个输出一条子网络，或通过嵌套 list 精细控制：

```python
net = dde.nn.PFNN([2, [20, 20], [20, 20], 2], "tanh", "Glorot normal")
```

### 10.3 DeepONet

```python
net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[m, 40, 40],
    layer_sizes_trunk =[dim_x, 40, 40],
    activation="relu",
    kernel_initializer="Glorot normal",
)
```

- `DeepONet`：不配对输入 / 输出。
- `DeepONetCartesianProd`：所有 branch 与 trunk 的笛卡尔积（常用格式）。
- `PODDeepONet`、`MIONetCartesianProd`、`PODMIONet`：变种。
- TF 下还支持 `FourierNet`、`MsFFN`、`STMsFFN`、`ResNet`。

### 10.4 输入 / 输出变换

```python
def input_transform(x):
    return 2 * (x - 0.5)   # 归一化到 [-1, 1]

def output_transform(x, y):
    return x[:, 0:1] * (1 - x[:, 0:1]) * y   # 硬约束：边界处 y=0

net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)
```

**硬约束** 可以把 BC 直接编码到输出，避免 soft BC 权重调参。对复杂几何，建议搭配 `geom.boundary_constraint_factor(x)` 作为近似距离函数。

---

## 11. 训练：`dde.Model`

```python
model = dde.Model(data, net)

# 编译：选择优化器、损失、指标、学习率衰减、损失权重、可训练变量
model.compile(
    "adam",
    lr=1e-3,
    loss="MSE",
    metrics=["l2 relative error"],
    decay=("inverse time", 2000, 0.5),
    loss_weights=[1, 10, 10],            # 与 [pde] + bcs 长度一致
    external_trainable_variables=[C1],   # 反问题中的物理未知量
)

# 训练
losshistory, train_state = model.train(
    iterations=20000,
    display_every=1000,
    callbacks=[dde.callbacks.EarlyStopping(patience=2000)],
    model_save_path="model/ckpt",
)

# 二阶优化（通常在 Adam 之后再接 L-BFGS）
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 预测与残差
y = model.predict(X)
pde_residual = model.predict(X, operator=pde)

# 保存 / 恢复
path = model.save("model/pinn")
model.restore(path)
```

### 11.1 优化器

- 一阶：`adam` / `sgd` / `rmsprop` / `adamw`（PyTorch）。
- 二阶：`L-BFGS`（TF1 走 SciPy；TF2 走 TFP；PyTorch / Paddle 走原生 LBFGS）、`NNCG`（仅 PyTorch）。

调参：

```python
dde.optimizers.set_LBFGS_options(
    maxcor=100, ftol=0, gtol=1e-8, maxiter=15000, maxls=50
)
dde.optimizers.set_NNCG_options(rank=50, mu=1e-1, updatefreq=20)
```

### 11.2 损失权重

`loss_weights` 长度须与损失数量一致：PDE 残差数 + BC/IC 数。例如：

```python
# [pde_loss, bc_loss, ic_loss]
model.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
```

### 11.3 常见回调

```python
from deepxde import callbacks as cb

cb_list = [
    cb.EarlyStopping(patience=5000, monitor="loss_train"),
    cb.ModelCheckpoint("model/best", save_better_only=True, monitor="train loss"),
    cb.PDEPointResampler(period=100, pde_points=True, bc_points=False),
    cb.VariableValue([C1, C2], period=500, filename="variables.dat"),
    cb.DropoutUncertainty(period=5000),
    cb.Timer(available_time=60),  # 60 分钟
]
model.train(iterations=50000, callbacks=cb_list)
```

> ⚠️ `PDEPointResampler` 开启 `bc_points=True` 仅在 PyTorch / Paddle 后端有效；重采样后 BC 点数发生变化会触发 `num_bcs changed` 错误，需要重建 Model。

### 11.4 批处理

- 对 `dde.data.PDE / TimePDE`：**不要** 使用 `batch_size`，改用 `PDEPointResampler`（见 `examples/pinn_forward/diffusion_1d_resample.py`）。
- 对 `DeepONetCartesianProd`：`batch_size` 可以是整数（branch 维度）或元组（branch, trunk）。

---

## 12. 反问题（参数识别）

引入未知参数与观测数据：

```python
C1 = dde.Variable(1.0)   # 初值 1.0
C2 = dde.Variable(1.0)

def lorenz(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + 8/3 * y3,
    ]

observe_y0 = dde.icbc.PointSetBC(t_obs, y_obs[:, 0:1], component=0)
# ...

model.compile("adam", lr=1e-3,
              external_trainable_variables=[C1, C2])
variable_cb = dde.callbacks.VariableValue([C1, C2], period=500, filename="variables.dat")
model.train(iterations=20000, callbacks=[variable_cb])
```

要点：
- `dde.Variable(value)` 会创建当前后端的可训练变量。
- `compile` 时必须传入 `external_trainable_variables=[C1, C2]`（TF1 除外，它会自动收集）。
- 用 `PointSetBC` 承载观测数据的残差项。
- `VariableValue` 回调周期性导出参数值，便于后处理。

---

## 13. 算子学习（DeepONet）

### 13.1 纯数据驱动

```python
data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train,
                                    X_test=X_test, y_test=y_test)
net  = dde.nn.DeepONetCartesianProd([m, 40, 40], [dim_x, 40, 40], "relu", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["mean l2 relative error"])
model.train(iterations=10000)
```

### 13.2 物理信息算子（PI-DeepONet）

```python
geom   = dde.geometry.TimeDomain(0, 1)
pde    = dde.data.PDE(geom, pde_fn, ic, num_domain=20, num_boundary=2, num_test=40)
space  = dde.data.GRF(length_scale=0.2)
data   = dde.data.PDEOperatorCartesianProd(pde, space, eval_pts, 1000, num_test=100, batch_size=100)
net    = dde.nn.DeepONetCartesianProd([50, 128, 128, 128], [1, 128, 128, 128], "tanh", "Glorot normal")

# 把 IC 编码为硬约束
net.apply_output_transform(lambda inputs, outputs: outputs * transpose(inputs[1], [1, 0]))
model = dde.Model(data, net)
model.compile("adam", lr=5e-4)
model.train(iterations=40000)
```

### 13.3 ZCS 加速

当算子学习涉及二阶以上导数且 branch 样本数较多时，`dde.zcs` 提供“零坐标偏移”加速：
- 用 `dde.zcs.LazyLossModel` 替代 `dde.Model`；
- 在 PDE 残差中用 `dde.zcs.jacobian / hessian` 计算。

参考完整示例：
- `examples/operator/diff_rec_aligned_zcs_pideeponet.py`
- `examples/operator/stokes_aligned_zcs_pideeponet.py`

---

## 14. 不确定性量化与多精度

### 14.1 MC Dropout

```python
net = dde.nn.FNN([2, 50, 50, 1], "tanh", "Glorot normal", dropout_rate=0.1)
model.compile("adam", lr=1e-3)
model.train(iterations=20000,
            callbacks=[dde.callbacks.DropoutUncertainty(period=1000)])
y_std = model.train_state.y_std_test  # 标准差估计
```

### 14.2 多精度 MFNN

```python
data = dde.data.MfDataSet(X_lo, y_lo, X_hi, y_hi, X_test, y_test)
net  = dde.nn.MfNN([1, 20, 20, 1], [1, 20, 20, 1], "tanh", "Glorot uniform")
```

---

## 15. 保存与加载

```python
path = model.save("model/my_run")            # model/my_run-<iter>.pt/.ckpt/...
# 后续恢复
model.restore(path)                           # PyTorch 下可传 device="cpu" / "cuda:0"
```

- `protocol="backend"`（默认）可以 `restore`。
- `protocol="pickle"` 仅备份 `state_dict`，无法 restore。

---

## 16. 可视化

```python
import matplotlib.pyplot as plt

dde.saveplot(losshistory, train_state, issave=True, isplot=True)  # 导出 loss / best 预测
dde.utils.plot_loss_history(losshistory)                           # 单独画 loss
dde.utils.plot_best_state(train_state)

# PDE 残差
x = geom.uniform_points(1000, boundary=True)
r = model.predict(x, operator=pde)
plt.plot(x, r); plt.show()
```

录制训练过程动画（需要 ImageMagick）：

```python
movie = dde.callbacks.MovieDumper("model/movie", [-1], [1], period=100,
                                   save_spectrum=True, y_reference=func)
model.train(iterations=10000, callbacks=[movie])
```

---

## 17. 并行训练（TF1 + Horovod）

```bash
# 按权弱扩展（每 rank 的 num_domain）
DDE_BACKEND=tensorflow.compat.v1 \
mpirun -np 4 python pde.py
```

```python
dde.config.set_parallel_scaling("weak")   # 或 "strong"
# train_distribution 必须是 "pseudo"
data = dde.data.PDE(..., train_distribution="pseudo")

# 可选：自定义 Horovod 优化器参数
dde.optimizers.set_hvd_opt_options(
    compression=hvd.Compression.fp16,
    op=hvd.Average,
    backward_passes_per_step=1,
)
```

其它后端当前未内置 Horovod，若有需要请自行对接 `torch.distributed` / `jax.pmap`。

---

## 18. 故障排查 FAQ

| 问题 | 建议 |
| --- | --- |
| `AttributeError: module 'deepxde.nn.xxx' has no attribute 'YYY'` | 当前后端未实现该网络。可换后端或按开发者指南贡献实现。 |
| `API "xxx" is not supported by backend "yyy"` | 对应后端缺少该统一接口，建议换后端或在社区提交 Issue。 |
| 损失 NaN / 过早停训 | 检查 PDE 表达式是否用到了错误后端的数学函数；尝试 `dde.config.set_default_float("float64")`；减小学习率。 |
| L-BFGS 提前结束 | 设为 `float64`；调大 `maxiter / maxls`；检查是否有非光滑 BC。 |
| `num_bcs changed` | `PDEPointResampler` 触发 BC 点数变化；在回调外重建 `Model` 或减少 BC 点抽样变化。 |
| 反问题参数不收敛 | 增大观测点 `PointSetBC`；对 PDE 残差加 `loss_weights`；使用 `VariableValue` 定时观察。 |
| JAX 下 `jacobian` 返回元组 | 解包第一个元素：`dy_x, _ = dde.grad.jacobian(y, x, i=0, j=0)`。 |
| GPU OOM | 降低 `num_domain`；使用 `PDEPointResampler` 动态采样；开启混合精度 `set_default_float("mixed")`；对算子学习使用 `dde.zcs`。 |
| 复现性 | `dde.config.set_random_seed(seed)`；TF2 建议每次新起 Python 进程。 |

---

## 19. 最小可运行模板

### 19.1 PINN 正问题（1D Poisson）

```python
import deepxde as dde
import numpy as np
from deepxde.backend import tf   # TF 后端

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

def func(x):
    return np.sin(np.pi * x)

geom = dde.geometry.Interval(-1, 1)
bc   = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```

### 19.2 反问题（Lorenz 系统参数识别）

见 `examples/pinn_inverse/Lorenz_inverse.py`，核心片段已在第 12 节给出。

### 19.3 算子学习（反导算子）

见 `examples/operator/antiderivative_aligned.py`，要点已在第 13 节给出。

---

## 20. 延伸阅读

- 主论文：Lu, Meng, Mao, Karniadakis, *SIAM Review* 63(1), 2021.
- 官方文档：<https://deepxde.readthedocs.io>
- FAQ：`docs/user/faq.rst`
- 研究论文列表：`docs/user/research.rst`
- 示例目录：`examples/pinn_forward`、`examples/pinn_inverse`、`examples/operator`

下一步：阅读 [`tutorial.md`](./tutorial.md)，按“从入门到精通”10 个章节动手实践。
