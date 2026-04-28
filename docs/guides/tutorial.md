# DeepXDE 从入门到精通教程

> 本教程采取 **10 章递进** 的方式：从环境搭建到复杂算子学习与框架扩展。建议按顺序学习；若你只关心特定主题，可通过目录直达。所有示例均可在 `examples/` 目录找到原始脚本。

## 目录

- 第 1 章：环境搭建与 Hello PINN
- 第 2 章：几何域与采样 —— 把“求解域”告诉网络
- 第 3 章：边界与初值条件 —— 软约束与硬约束
- 第 4 章：时间相关 PDE —— 求解 Burgers 方程
- 第 5 章：自适应采样与优化器组合 —— 提升精度的关键
- 第 6 章：反问题 —— 从数据中恢复物理参数
- 第 7 章：复杂几何与高维 —— CSG、多尺度与分数阶
- 第 8 章：算子学习 —— DeepONet 与 PI-DeepONet
- 第 9 章：进阶加速 —— 混合精度、ZCS、并行
- 第 10 章：精通之路 —— 自定义回调、网络与后端

---

## 第 1 章：环境搭建与 Hello PINN

### 1.1 安装

```bash
# 只装一个后端即可
pip install torch              # PyTorch（最省心）
# 或 pip install "tensorflow>=2.3" "tensorflow-probability>=0.11"
# 或 pip install jax flax optax
# 或 pip install paddlepaddle

pip install deepxde            # 本体
```

设置后端（建议写入 shell 配置）：

```bash
export DDE_BACKEND=pytorch     # 选用 pytorch
```

### 1.2 Hello PINN：一维 Poisson 方程

目标：求解 `-u''(x) = π² sin(πx)`，`x ∈ (-1, 1)`，`u(±1) = 0`。真实解 `u(x) = sin(πx)`。

```python
# tutorial_ch1_hello.py
import deepxde as dde
import numpy as np

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # 为了跨后端通用，使用 numpy 的常量；非线性部分用 sin 有后端差异时请看下方注释
    return -dy_xx - (np.pi ** 2) * dde.backend.sin(np.pi * x)

def ref(x):
    return np.sin(np.pi * x)

geom = dde.geometry.Interval(-1, 1)
bc   = dde.icbc.DirichletBC(geom, ref, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(geom, pde, bc, num_domain=16, num_boundary=2,
                    solution=ref, num_test=100)

net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
loss, state = model.train(iterations=10000)

dde.saveplot(loss, state, issave=True, isplot=True)
```

> 不同后端的三角函数写法：TF 系 `tf.sin`，PyTorch `torch.sin`，JAX `jnp.sin`，Paddle `paddle.sin`。若追求单文件跨后端，可以用 `dde.backend.sin`（由统一接口层提供）。

运行：

```bash
DDE_BACKEND=pytorch python tutorial_ch1_hello.py
```

你会看到：`Compiling model... / Training model... / Step  Train loss  Test loss ... / Best model at step X`。最终 `l2 relative error` 应在 `1e-4` 量级。

---

## 第 2 章：几何域与采样

目标：掌握 DeepXDE 能描述什么几何，如何采样。

### 2.1 原语一览

```python
import deepxde as dde

# 1D
dde.geometry.Interval(0, 1)

# 2D
dde.geometry.Rectangle([0, 0], [1, 1])
dde.geometry.Disk([0, 0], 1.0)
dde.geometry.Triangle([0, 0], [1, 0], [0, 1])
dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
dde.geometry.Ellipse([0, 0], 2, 1)
dde.geometry.StarShaped([0, 0], 0.5, 0.2, 5)

# 3D
dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
dde.geometry.Sphere([0, 0, 0], 1.0)

# 高维 / 点云
dde.geometry.Hypercube([0]*5, [1]*5)
dde.geometry.Hypersphere([0]*4, 1.0)
dde.geometry.PointCloud(my_points)
```

### 2.2 CSG 组合

求解“L 形板”上的 Poisson 问题：

```python
# examples/pinn_forward/Poisson_Lshape.py 原始脚本
geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])

# 等价地用 Rectangle - Rectangle
# geom = dde.geometry.Rectangle([-1, -1], [1, 1]) - dde.geometry.Rectangle([0, 0], [1, 1])
```

### 2.3 时空域

```python
space = dde.geometry.Interval(-1, 1)
time  = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(space, time)  # 样本变成 (x, t)
```

### 2.4 采样分布

```python
X = geom.random_points(1000, random="Hammersley")
X = geom.random_points(1000, random="Sobol")
X = geom.random_points(1000, random="LHS")
```

`PDE(train_distribution=...)` 支持 `uniform/pseudo/LHS/Halton/Hammersley/Sobol`。经验：**小规模问题** 用 `Hammersley`（默认），**大规模** 或 **并行** 用 `pseudo`。

### 2.5 小练习

把第 1 章的 Poisson 问题换成 `num_domain=8` → `num_domain=64`，观察收敛速度与最终 l2 误差。

---

## 第 3 章：边界与初值条件 —— 软约束与硬约束

### 3.1 五类内置 BC

```python
dde.icbc.DirichletBC(geom, func_g, on_boundary)       # u = g(x)
dde.icbc.NeumannBC(geom, func_h, on_boundary)         # ∂u/∂n = h(x)
dde.icbc.RobinBC(geom, func_r, on_boundary)           # ∂u/∂n = r(x, u)
dde.icbc.PeriodicBC(geom, component_x=0, on_boundary=..., derivative_order=0)
dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x) + y, on_boundary)
```

`on_boundary(x, on_boundary_flag) -> bool` 用来筛选“哪些边界点”适用该条件。常见写法：

```python
# 所有边界
lambda _, on_b: on_b
# 左端点
lambda x, on_b: on_b and np.isclose(x[0], 0)
# 上下两条边
lambda x, on_b: on_b and np.isclose(abs(x[1]), 1)
```

### 3.2 软约束：默认方式

软约束把 BC 作为损失项；`loss_weights` 决定其权重：

```python
model.compile("adam", lr=1e-3, loss_weights=[1, 100])  # [pde, bc]
```

### 3.3 硬约束：把 BC 内嵌到网络输出

硬约束让网络输出自动满足 BC，训练更稳。常用套路 `u(x) = D(x) * NN(x) + g(x)`：

```python
# 例：u(-1) = u(1) = 0
def output_transform(x, y):
    return (1 + x) * (1 - x) * y

net.apply_output_transform(output_transform)
# 此时可以不传 bc，或者保留作为一致性检查
```

对复杂几何，DeepXDE 提供 `geom.boundary_constraint_factor(x, smoothness)` 近似距离函数：
- `"C0"`：连续但不可导，测度零点不可导。
- `"C0+"`（默认）：连续且几乎处处可导。
- `"Cinf"`：任意阶可导（代价是高阶多项式）。

### 3.4 点集 BC：PointSetBC

在反问题 / 带观测数据的任务中，离散观测点用 `PointSetBC`：

```python
obs = dde.icbc.PointSetBC(X_obs, y_obs, component=0)
```

### 3.5 小练习

- 把第 1 章的 Dirichlet BC 改为硬约束，对比训练速度与最终误差。
- 用 `OperatorBC` 给 `u'(1) + u(1) = 0` 这样的混合条件建模。

---

## 第 4 章：时间相关 PDE —— 求解 Burgers 方程

目标：`u_t + u u_x - 0.01/π u_xx = 0`，`x ∈ [-1, 1]`，`t ∈ [0, 0.99]`，`u(x, 0) = -sin(πx)`，`u(±1, t) = 0`。

完整脚本见 `examples/pinn_forward/Burgers.py`：

```python
import deepxde as dde
import numpy as np

def pde(x, y):
    dy_x  = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t  = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

geom     = dde.geometry.Interval(-1, 1)
timedom  = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedom)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_b: on_b)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]),
                 lambda _, on_init: on_init)

data = dde.data.TimePDE(geomtime, pde, [bc, ic],
                        num_domain=2540, num_boundary=80, num_initial=160)
net  = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3); model.train(iterations=15000)
model.compile("L-BFGS");        loss, state = model.train()
dde.saveplot(loss, state, issave=True, isplot=True)
```

### 4.1 时空输入格式

输入 `x` 是 `(N, dim_space + 1)` 的张量，最后一列是时间：
- `x[:, 0:1]`：空间坐标（1D 问题）。
- `x[:, 1:2]`：时间坐标。

所以 `dy_t = dde.grad.jacobian(y, x, i=0, j=1)`（对第 1 列求导）。

### 4.2 标准工作流：Adam 预热 + L-BFGS 精修

几乎所有 PINN 任务都遵循这个模板：
1. `model.compile("adam", lr=1e-3)` → `model.train(iterations=N)` 做粗调。
2. `model.compile("L-BFGS")` → `model.train()` 二阶精修（内部根据 `set_LBFGS_options` 跑满 maxiter 或收敛）。

对 PyTorch 还可再追加一轮 **NNCG** 精修：

```python
dde.optimizers.set_NNCG_options(rank=50, mu=1e-1)
model.compile("NNCG")
model.train(iterations=1000, display_every=100)
```

### 4.3 评估误差

```python
data_ref = np.load("../dataset/Burgers.npz")
t, x, exact = data_ref["t"], data_ref["x"], data_ref["usol"].T
xx, tt = np.meshgrid(x, t)
X = np.vstack((xx.ravel(), tt.ravel())).T
y_true = exact.flatten()[:, None]

y_pred = model.predict(X)
resid  = model.predict(X, operator=pde)    # PDE 残差
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
print("Mean residual:",     np.mean(np.abs(resid)))
```

---

## 第 5 章：自适应采样与优化器组合

### 5.1 残差自适应采样（RAR）

思想：在 PDE 残差大的地方追加新点。DeepXDE 提供 `PDEPointResampler` 实现定期整体重采样；对“只补点”的 RAR，可手写：

```python
resampler = dde.callbacks.PDEPointResampler(period=100)
model.train(iterations=10000, callbacks=[resampler])
```

完整示例：`examples/pinn_forward/diffusion_1d_resample.py`、`Burgers_RAR.py`、`heat_resample.py`。

### 5.2 损失权重调度

多个 loss 之间（PDE / BC / 观测）量级差异大时，使用 `loss_weights` 手动平衡：

```python
model.compile("adam", lr=1e-3, loss_weights=[1, 100, 10])
```

更进阶地，可以写回调动态调整（见下文第 10 章“自定义回调”）。

### 5.3 学习率衰减

```python
model.compile("adam", lr=1e-3, decay=("inverse time", 2000, 0.5))
# PyTorch 下还可：("step", 1000, 0.9)、("cosine", T_max, eta_min)、("exponential", gamma) 等
```

### 5.4 L-BFGS 微调

```python
dde.optimizers.set_LBFGS_options(maxcor=100, gtol=1e-10, maxiter=50000)
model.compile("L-BFGS")
model.train()
```

当 L-BFGS 提前停止，尝试：
- `dde.config.set_default_float("float64")`（强烈推荐）。
- 调大 `maxiter` 与 `maxls`。

### 5.5 EarlyStopping

训练预算有限或 loss 早已饱和时：

```python
model.train(iterations=50000,
            callbacks=[dde.callbacks.EarlyStopping(patience=5000, monitor="loss_train")])
```

---

## 第 6 章：反问题 —— 从数据中恢复物理参数

目标：已知 Lorenz 系统的若干观测点，反推参数 `σ, ρ, β`。

完整脚本：`examples/pinn_inverse/Lorenz_inverse.py`。

```python
import deepxde as dde
import numpy as np

# 待识别的三个参数
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)

def lorenz(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + C3 * y3,
    ]

geom = dde.geometry.TimeDomain(0, 3)
ic1 = dde.icbc.IC(geom, lambda X: -8, lambda _, on_init: on_init, component=0)
ic2 = dde.icbc.IC(geom, lambda X:  7, lambda _, on_init: on_init, component=1)
ic3 = dde.icbc.IC(geom, lambda X: 27, lambda _, on_init: on_init, component=2)

# 观测数据
t_obs, y_obs = np.load("../dataset/Lorenz.npz").values()
obs0 = dde.icbc.PointSetBC(t_obs, y_obs[:, 0:1], component=0)
obs1 = dde.icbc.PointSetBC(t_obs, y_obs[:, 1:2], component=1)
obs2 = dde.icbc.PointSetBC(t_obs, y_obs[:, 2:3], component=2)

data = dde.data.PDE(geom, lorenz, [ic1, ic2, ic3, obs0, obs1, obs2],
                    num_domain=400, num_boundary=2, anchors=t_obs)

net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)

variable = dde.callbacks.VariableValue([C1, C2, C3], period=600,
                                       filename="variables.dat")

model.compile("adam", lr=1e-3,
              external_trainable_variables=[C1, C2, C3])
model.train(iterations=20000, callbacks=[variable])

model.compile("L-BFGS",
              external_trainable_variables=[C1, C2, C3])
model.train(callbacks=[variable])
```

### 6.1 反问题五要素

1. `dde.Variable(init_value)` 声明未知量。
2. `compile(..., external_trainable_variables=[...])` 告诉框架它们也参与训练。
3. `PointSetBC` 用观测点驱动网络。
4. `anchors=t_obs` 让训练样本必含这些观测时刻。
5. `VariableValue` 回调周期性导出参数值（便于监控 / 后期分析）。

### 6.2 其他反问题示例

| 场景 | 脚本 |
| --- | --- |
| 扩散方程未知反应率 | `examples/pinn_inverse/diffusion_reaction_rate.py` |
| 椭圆方程未知系数场 | `examples/pinn_inverse/elliptic_inverse_field.py` |
| 分数阶 Poisson 反演 | `examples/pinn_inverse/fractional_Poisson_2d_inverse.py` |
| Navier-Stokes 反问题 | `examples/pinn_inverse/Navier_Stokes_inverse.py` |
| 拟多孔介质 Brinkman-Forchheimer | `examples/pinn_inverse/brinkman_forchheimer.py` |

---

## 第 7 章：复杂几何与高维、多尺度、分数阶

### 7.1 CSG：Poisson on L-shape / Disk

`examples/pinn_forward/Poisson_Lshape.py`、`Laplace_disk.py` 展示了如何用 `Polygon / Rectangle / Disk` 直接定义复杂域并求解。

### 7.2 多尺度问题

`examples/pinn_forward/Poisson_multiscale_1d.py` 演示 **多尺度 Fourier 特征网络** 的用法（仅 TF 后端内置）：

```python
net = dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10, 20])
```

### 7.3 高维 Poisson

`fractional_Poisson_3d.py` 展示 3D 问题；更高维可用 `Hypercube`：

```python
geom = dde.geometry.Hypercube([0]*10, [1]*10)   # 10 维单位立方体
```

高维下推荐：
- `num_domain` 随维度指数增长，但要配合 GPU 显存。
- `train_distribution="Sobol"` 或 `"Hammersley"` 低偏差序列。
- 预热阶段先 `num_test` 小一些，最后再评估。

### 7.4 分数阶 PDE（fPINN）

```python
alpha = 1.8
data = dde.data.FPDE(geom, pde_fn, alpha, bcs,
                     resolution=[100], meshtype="dynamic",
                     num_domain=64, solution=ref)
```

更多参考：`fractional_Poisson_1d.py` / `fractional_diffusion_1d.py`。

### 7.5 积分-微分方程

```python
def kernel(x, s, y, unused): ...
data = dde.data.IDE(geom, pde_fn, [bc, ic], quad_deg=20, kernel=kernel,
                    num_domain=32, num_boundary=2, num_initial=0)
```

示例：`examples/pinn_forward/Volterra_IDE.py`、`ide.py`。

---

## 第 8 章：算子学习 —— DeepONet 与 PI-DeepONet

算子学习的目标是学一个 **函数 → 函数** 的映射。DeepONet 结构：`G(v)(y) = <branch(v), trunk(y)>`。

### 8.1 纯数据驱动：反导算子

完整示例：`examples/operator/antiderivative_aligned.py`。

```python
import deepxde as dde, numpy as np

d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train, y_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32)), d["y"].astype(np.float32)
d = np.load("antiderivative_aligned_test.npz",  allow_pickle=True)
X_test,  y_test  = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32)), d["y"].astype(np.float32)

data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train,
                                    X_test=X_test,   y_test=y_test)

m = 100; dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],       # branch：输入函数在 100 个点处的值
    [dim_x, 40, 40],   # trunk：评估点 y
    "relu", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["mean l2 relative error"])
model.train(iterations=10000)
```

要点：
- 输入是 `tuple(branch_input, trunk_input)`。
- `TripleCartesianProd` 意味着 branch × trunk 的笛卡尔积输出，常见于网格数据。
- 评估：`model.predict((v, y))`。

### 8.2 物理信息 DeepONet（PI-DeepONet）

完整示例：`examples/operator/antiderivative_aligned_pideeponet.py`。

```python
geom = dde.geometry.TimeDomain(0, 1)

def pde(x, u, v):
    return dde.grad.jacobian(u, x) - v        # u'(x) = v(x)

ic    = dde.icbc.IC(geom, lambda _: 0, lambda _, on_init: on_init)
pde_d = dde.data.PDE(geom, pde, ic, num_domain=20, num_boundary=2, num_test=40)
space = dde.data.GRF(length_scale=0.2)        # 高斯随机场采样 v
evalp = np.linspace(0, 1, 50)[:, None]

data  = dde.data.PDEOperatorCartesianProd(pde_d, space, evalp, 1000,
                                          num_test=100, batch_size=100)
net   = dde.nn.DeepONetCartesianProd([50, 128, 128, 128],
                                     [1, 128, 128, 128], "tanh", "Glorot normal")

# 硬约束：IC 为 0，网络输出乘以 t 自动满足
def zero_ic(inputs, outputs):
    return outputs * transpose(inputs[1], [1, 0])
net.apply_output_transform(zero_ic)

model = dde.Model(data, net)
model.compile("adam", lr=5e-4)
model.train(iterations=40000)
```

思路：
- 用 `dde.data.GRF` / `PowerSeries` / `Chebyshev` 定义输入函数空间。
- `PDEOperatorCartesianProd(pde, space, eval_points, n_funcs)` 生成训练样本。
- 网络输出做硬约束，把 IC 吸收到结构里。

### 8.3 MIONet：多输入算子

```python
net = dde.nn.MIONetCartesianProd([m, 40, 40], [m, 40, 40], [dim_y, 40, 40],
                                 "tanh", "Glorot normal")
data = dde.data.QuadrupleCartesianProd(X_train=..., y_train=..., X_test=..., y_test=...)
```

见 `examples/operator` 中 `advection_*`、`stokes_*`、`diff_rec_*`。

### 8.4 POD-DeepONet

在高精度数据下，POD-DeepONet 通过预先 POD 分解 trunk 提升精度：

```python
net = dde.nn.PODDeepONet(pod_basis, [m, 40, 40], "tanh", "Glorot normal")
```

---

## 第 9 章：进阶加速

### 9.1 混合精度

```python
dde.config.set_default_float("mixed")
```

TF 后端自动启用 `mixed_float16`；PyTorch 下通过 `torch.autocast` 在前后向用 FP16，权重仍存 FP32。可显著减小显存与加速。

### 9.2 前向模式自动微分

在输入维度小（例如 1D/2D）、输出维度大的问题中，前向 AD 可能更快：

```python
dde.config.set_default_autodiff("forward")
```

与 `dde.grad.jacobian/hessian` 完全兼容，无需改业务代码。

### 9.3 Zero Coordinate Shift（ZCS）

当 PI-DeepONet 涉及大 branch 样本 + 高阶导数时，ZCS 把空间坐标替换为基准点 + 零偏移，使多样本共享 AD 结果：

```python
from deepxde.zcs import LazyLossModel, jacobian as zjacobian, hessian as zhessian

def pde(x, u, v):
    return zjacobian(u, x) - v

model = LazyLossModel(data, net)
```

完整示例：`examples/operator/diff_rec_aligned_zcs_pideeponet.py`、`stokes_aligned_zcs_pideeponet.py`。

### 9.4 多 GPU（Horovod，仅 TF1）

```bash
DDE_BACKEND=tensorflow.compat.v1 mpirun -np 4 python solve.py
```

```python
dde.config.set_parallel_scaling("weak")   # 每 rank num_domain
# train_distribution 必须为 "pseudo"
```

### 9.5 `Timer` 回调做时间预算

```python
model.train(iterations=10**7,
            callbacks=[dde.callbacks.Timer(available_time=30)])   # 30 分钟
```

---

## 第 10 章：精通之路 —— 自定义回调 / 网络 / 后端

### 10.1 自定义回调：动态调整 loss 权重

```python
from deepxde.callbacks import Callback

class AdaptiveLossWeights(Callback):
    def __init__(self, period=1000, alpha=0.9):
        super().__init__()
        self.period = period
        self.alpha  = alpha
        self.count  = 0

    def on_epoch_end(self):
        self.count += 1
        if self.count % self.period:
            return
        losses = self.model.train_state.loss_train
        # 让 BC loss 权重随其大小动态增加 / 减小
        w = self.model.loss_weights or [1] * len(losses)
        w = [self.alpha * w[i] + (1 - self.alpha) * (losses[0] / (l + 1e-8))
             for i, l in enumerate(losses)]
        self.model.loss_weights = w
```

使用：

```python
model.compile("adam", lr=1e-3, loss_weights=[1]*3)
model.train(iterations=20000, callbacks=[AdaptiveLossWeights()])
```

### 10.2 自定义网络（PyTorch 后端）

```python
import torch
from deepxde.nn.pytorch import NN
from deepxde import config

class SirenFNN(NN):
    """使用 SIREN 初始化 & sin 激活的 FNN。"""

    def __init__(self, layer_sizes, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            lin = torch.nn.Linear(layer_sizes[i-1], layer_sizes[i],
                                  dtype=config.real(torch))
            # SIREN 初始化
            with torch.no_grad():
                bound = (6 / layer_sizes[i-1]) ** 0.5 / omega_0
                lin.weight.uniform_(-bound, bound)
            self.linears.append(lin)
        self.regularizer = None

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for lin in self.linears[:-1]:
            x = torch.sin(self.omega_0 * lin(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
```

注册到命名空间（可选）：

```python
import deepxde as dde
dde.nn.SirenFNN = SirenFNN
```

### 10.3 自定义 BC（适合特殊物理场景）

```python
from deepxde.icbc.boundary_conditions import BC

class FluxBC(BC):
    """给定沿面法向的通量：(∇y · n) * k(x) = g(x)"""
    def __init__(self, geom, k_fn, g_fn, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.k_fn, self.g_fn = k_fn, g_fn

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        flux = self.normal_derivative(X, inputs, outputs, beg, end)
        k    = self.k_fn(X[beg:end])
        g    = self.g_fn(X[beg:end])
        return k * flux - g
```

### 10.4 自定义损失 / 指标

```python
from deepxde import losses, metrics
import deepxde.backend as bkd

def huber(delta=1.0):
    def _loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = bkd.abs(err)
        mask = bkd.cast(abs_err < delta, err.dtype)
        return bkd.reduce_mean(mask * 0.5 * err * err
                               + (1 - mask) * delta * (abs_err - 0.5 * delta))
    return _loss

model.compile("adam", lr=1e-3, loss=huber(delta=0.5))
```

### 10.5 自定义后端（高阶）

当你想接入一个全新的深度学习框架：
1. 实现 `deepxde/backend/<name>/{__init__.py, tensor.py}`（符合 `backend.py` 的 API 规约）。
2. 在 `deepxde/nn/<name>/` 下移植 `NN / FNN / DeepONet` 等网络。
3. 在 `deepxde/optimizers/<name>/` 下提供 `get / is_external_optimizer`。
4. 在 `deepxde/gradients/<name>/` 下实现 `jacobian/hessian/clear`。
5. 在 `deepxde/model.py` 增加 `_compile_<name>` / `_outputs` / `_outputs_losses` / `_train_step` 分支。
6. 更新 `deepxde/config.py` 的随机种子 / 默认 dtype / 混合精度逻辑。

测试：至少跑通 `examples/pinn_forward/Poisson_Dirichlet_1d.py`、`Burgers.py`、`examples/operator/antiderivative_aligned.py`。

### 10.6 调试技巧

- **可视化 PDE 残差**：`model.predict(X, operator=pde)` 后直接 `plt.imshow`；热图里最亮的区域就是该重点采样的地方。
- **loss 分解**：`losshistory.loss_train` 是 `List[ndarray]`，可拆分每条 loss 单独画图。
- **离线推理**：
  ```python
  model.save("ckpt")
  # ...
  model = dde.Model(data, net); model.compile("adam"); model.restore("ckpt-20000.pt")
  ```
- **对照实验**：固定随机种子 `dde.config.set_random_seed(42)` 后切换超参，减少方差干扰。

---

## 结语

走完这十章，你应该能：
- 用 DeepXDE 求解常规 / 复杂 / 时间相关 / 分数阶 / 反问题类型的 (I)PDE。
- 搭建 DeepONet 和 PI-DeepONet 做算子学习。
- 通过 RAR、硬约束、混合精度、ZCS 等手段将精度 / 速度推到极致。
- 扩展框架（自定义回调、网络、BC、损失、乃至后端）。

**推荐进一步阅读**：
- `developer_guide.md`：深入源码的结构化参考。
- `user_guide.md`：按 API 组织的查阅手册。
- `docs/demos/`：所有示例的 RST 展示。
- `docs/user/faq.rst`：常见问题。
- `docs/user/research.rst`：使用 DeepXDE 的科研论文列表（寻找灵感）。

祝你科学计算愉快！
