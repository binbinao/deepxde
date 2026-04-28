# DeepXDE 开发者文档

> 版本基线：基于仓库 `master` 分支源码挖掘整理。
> 适用读者：希望阅读源码、二次开发、扩展后端 / 网络 / 数据类型，或向上游提交 PR 的开发者。

---

## 1. 项目定位与范围

DeepXDE 是一个面向 **科学机器学习（SciML）与物理信息学习（Physics-Informed Learning）** 的 Python 库，核心能力包括：

- **PINN**：求解正问题 / 反问题的 ODE、PDE、IDE、分数阶 PDE（fPDE）、随机 PDE（sPDE）。
- **DeepONet 家族**：DeepONet、POD-DeepONet、MIONet、Fourier-DeepONet、Physics-Informed DeepONet、多精度 DeepONet、DeepM&Mnet。
- **多精度神经网络 MFNN**：从不同精度的数据中学习。
- **通用工程能力**：CSG 几何、6 种采样、多类 BC / IC、3 种自动微分模式、回调系统、模型持久化、Dropout 不确定性量化、float16/32/64 与混合精度、Horovod 数据并行等。

支持 5 种深度学习后端：`tensorflow.compat.v1`、`tensorflow`、`pytorch`、`jax`、`paddle`。

元信息（来自 `pyproject.toml`）：

- 包名：`DeepXDE`  ·  许可证：LGPL-2.1  ·  Python：`>=3.9`
- 运行时依赖：`matplotlib`, `numpy`, `scikit-learn`, `scikit-optimize>=0.10.2`, `scipy`
- 版本号：通过 `setuptools_scm` 动态生成到 `deepxde/_version.py`

---

## 2. 仓库目录结构

```
deepxde/                          # 源码根包
├── __init__.py                  # 顶层命名空间，暴露 backend/data/geometry/...
├── _version.py                  # setuptools_scm 自动写入
├── backend/                     # 后端抽象：统一张量接口 + 5 种后端适配
│   ├── backend.py               # 抽象 API 规约（dtype、tensor ops、元素级数学、规约/规范/正则化）
│   ├── __init__.py              # 运行时装载机制（DDE_BACKEND / config.json / 自动探测）
│   ├── set_default_backend.py
│   ├── utils.py
│   └── {jax,paddle,pytorch,tensorflow,tensorflow_compat_v1}/
├── config.py                    # 全局运行配置：精度、XLA、autodiff、随机种子、Horovod
├── real.py                      # 浮点精度类（float16/32/64）
├── callbacks.py                 # 回调体系：EarlyStopping / ModelCheckpoint / PDEPointResampler / VariableValue / ...
├── display.py                   # 训练日志打印
├── losses.py                    # 通用损失函数注册表（MSE/MAE/MAPE/mean L2 relative error/...）
├── metrics.py                   # 指标函数注册表（基于 numpy / sklearn）
├── model.py                     # 训练主流程：Model / TrainState / LossHistory
├── data/                        # 数据与约束容器（PDE/TimePDE/IDE/FPDE/PDEOperator/Triple/Quadruple/DataSet...）
├── geometry/                    # 几何原语 + CSG 组合 + 时空域 + PointCloud + QMC 采样
├── gradients/                   # 延迟求值的 Jacobian/Hessian（reverse / forward 两种 AD）
├── icbc/                        # 初值条件与边界条件（Dirichlet/Neumann/Robin/Periodic/Operator/PointSet/Interface2D）
├── nn/                          # 神经网络模块（按后端分子包装 FNN/PFNN/DeepONet/MIONet/...）
│   ├── activations.py           # 激活函数注册表
│   ├── deeponet_strategy.py     # DeepONet 训练策略
│   ├── initializers.py          # 初始化器
│   ├── regularizers.py          # 正则化
│   └── {jax,paddle,pytorch,tensorflow,tensorflow_compat_v1}/
├── optimizers/                  # 优化器抽象（Adam/L-BFGS/NNCG）+ 各后端适配
├── utils/                       # 计时 / 绘图 / MPI 辅助 / 后端特化工具
└── zcs/                         # Zero Coordinate Shift：面向 PI-DeepONet 的高效二阶 AD 加速
docs/
├── index.rst, conf.py, Makefile # Sphinx 文档入口
├── demos/                       # 按示例生成的 RST 展示
├── modules/                     # 自动生成的 API 参考
├── user/                        # 安装 / FAQ / 并行 / 团队 / 引用
└── guides/                      # 本文所在目录（开发者 / 用户 / 教程）
examples/
├── function/                    # 函数学习示例
├── operator/                    # DeepONet / PI-DeepONet / MIONet / ZCS 示例
├── pinn_forward/                # 37 个 PINN 正问题示例
└── pinn_inverse/                # 11 个 PINN 反问题示例
docker/Dockerfile                # 基于 Horovod 的 GPU 镜像
pyproject.toml, requirements.txt # 构建与依赖
```

---

## 3. 运行时架构总览

DeepXDE 的工作流被分解为 4 个正交维度：

```
┌──────────────────── Model.train() ────────────────────┐
│                                                      │
│   ┌────────────┐   ┌────────────┐   ┌─────────────┐  │
│   │  Geometry  │──▶│   Data     │──▶│    NN       │  │
│   │  (+ICBC)   │   │ (train/test│   │ (FNN/DeepONet│ │
│   │            │   │  sampler)  │   │  /PFNN/...) │  │
│   └────────────┘   └────────────┘   └─────────────┘  │
│                         │                  │        │
│                         ▼                  ▼        │
│                   ┌───────────────────────────┐      │
│                   │  Model (.compile/.train)  │      │
│                   │  optimizer + loss + AD    │      │
│                   └───────────────────────────┘      │
│                         │                             │
│         ┌───────────────┼────────────────┐            │
│         ▼               ▼                ▼            │
│    Callbacks      LossHistory        TrainState       │
└───────────────────────────────────────────────────────┘
```

`Model` 由 `Data`（包含训练样本 + 约束 / PDE 残差定义）和 `NN`（网络结构）组合而成；`compile()` 阶段根据当前后端装配优化器、求值函数与训练步函数；`train()` 阶段驱动 SGD / L-BFGS / NNCG / TFP 的循环。

### 3.1 顶层命名空间

`deepxde/__init__.py` 暴露以下子包别名：

| 别名 | 实际模块 |
| --- | --- |
| `dde.backend` | `deepxde.backend` |
| `dde.callbacks` | `deepxde.callbacks` |
| `dde.data` | `deepxde.data` |
| `dde.geometry` | `deepxde.geometry` |
| `dde.grad` | `deepxde.gradients` |
| `dde.icbc` | `deepxde.icbc` |
| `dde.nn` / `dde.maps` | `deepxde.nn` |
| `dde.utils` | `deepxde.utils` |
| `dde.zcs` | `deepxde.zcs` |
| `dde.Model` | `deepxde.model.Model` |
| `dde.Variable` | `deepxde.backend.Variable` |
| `dde.saveplot` | `deepxde.utils.saveplot` |

此外为保持向后兼容，直接导出 `DirichletBC / NeumannBC / RobinBC / PeriodicBC / OperatorBC / PointSetBC / PointSetOperatorBC / IC / Interface2DBC`。

---

## 4. 后端抽象层（`deepxde.backend`）

### 4.1 统一接口规约

`backend/backend.py` 定义了所有后端需要实现的统一 API（作为文档签名）：

- 数据类型字典：`data_type_dict()`，至少覆盖 `float16/32/64`、`uint8`、`int8/16/32/64`、`bool`。
- 张量元信息：`is_gpu_available`、`is_tensor`、`shape`、`size`、`ndim`。
- 基本变形：`transpose`、`reshape`、`expand_dims`、`reverse`、`roll`、`concat`、`stack`。
- 构造：`Variable(initial_value, dtype)`、`as_tensor`、`from_numpy`、`to_numpy`、`zeros`、`zeros_like`、`sparse_tensor`、`sparse_dense_matmul`。
- 元素级数学：`sin/cos/exp/square/abs/elu/relu/gelu/selu/sigmoid/silu/tanh/pow/lgamma/minimum/matmul`。
- 规约 / 规范：`mean`、`reduce_mean`、`sum`、`reduce_sum`、`prod`、`reduce_prod`、`min/max`、`reduce_min/max`、`norm`。
- 正则化：`l1_regularization`、`l2_regularization`、`l1_l2_regularization`。

各后端只需实现其中一部分即可，`is_enabled(api)` 用于运行时查询。未实现的 API 会装配一个抛 `ImportError` 的占位函数（见 `_gen_missing_api`）。

### 4.2 后端选择与加载

优先级（`backend/__init__.py::get_preferred_backend`）：
1. 环境变量 `DDE_BACKEND`（向后兼容 `DDEBACKEND`）。
2. `~/.deepxde/config.json` 中 `{"backend": "..."}`。
3. 自动检测 (`get_available_backend`)，命中后写回 `config.json`。
4. 兜底：尝试交互式安装 `paddle`。

合法值：`tensorflow.compat.v1` / `tensorflow` / `pytorch` / `jax` / `paddle`。

设定默认后端（CLI）：
```bash
python -m deepxde.backend.set_default_backend paddle
```

### 4.3 `config` 模块关键开关

`deepxde/config.py`：

- `real`：`Real(32)` 实例，提供 `set_float16/32/64` 以及类型转换；`set_default_float("mixed")` 启用混合精度（TF / PyTorch）。
- `random_seed` / `set_random_seed(seed)`：同时为 Python、NumPy、TensorFlow、PyTorch、JAX、Paddle 设定种子，并启用 `TF_DETERMINISTIC_OPS=1`、`TF_CUDNN_DETERMINISTIC=1`。
- `xla_jit`、`enable_xla_jit` / `disable_xla_jit`：仅在 TF/JAX 下有意义，默认在 GPU 上自动启用，JAX 始终 XLA。
- `autodiff`、`set_default_autodiff("forward" | "reverse")`：切换自动微分模式。
- Horovod：当环境中存在 `OMPI_COMM_WORLD_SIZE` 时自动启用（仅 `tensorflow.compat.v1` 实现）；`parallel_scaling ∈ {"weak", "strong"}` 控制每 rank 的样本数语义。

---

## 5. 数据与约束（`deepxde.data` / `deepxde.geometry` / `deepxde.icbc`）

### 5.1 `Data` 抽象基类

（`deepxde/data/data.py::Data`）

```python
class Data(abc.ABC):
    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None): ...
    def losses_train(self, ...) -> List[Tensor]     # 默认转调 losses()
    def losses_test(self, ...)  -> List[Tensor]     # 可分别实现训练/测试损失
    @abstractmethod
    def train_next_batch(self, batch_size=None): ...
    @abstractmethod
    def test(self): ...
```

所有具体数据类型（`PDE`、`TimePDE`、`IDE`、`FPDE`、`TimeFPDE`、`PDEOperator`、`PDEOperatorCartesianProd`、`Triple`、`TripleCartesianProd`、`Quadruple`、`QuadrupleCartesianProd`、`DataSet`、`MfDataSet`、`MfFunc`、`FuncConstraint`、`Function`、`Constraint`）都继承 `Data`，并在 `data/__init__.py` 中通过 `__all__` 导出。

### 5.2 `PDE` / `TimePDE` 关键属性

（`deepxde/data/pde.py`）

| 属性 | 说明 |
| --- | --- |
| `train_x_all` | 所有 PDE 残差采样点（去重、无序） |
| `train_x_bc` | BC 的采样点（首次构造后默认保持不变） |
| `num_bcs[i]` | 第 `i` 个 BC 的点数（长度与 `bcs` 对齐） |
| `train_x` / `test_x` | 喂给网络的完整输入，顺序为 BC 点 → PDE 点 |
| `train_aux_vars` | 辅助变量（配合 `auxiliary_var_function`） |

采样分布 `train_distribution`：`uniform` / `pseudo` / `LHS` / `Halton` / `Hammersley`（默认）/ `Sobol`。

### 5.3 几何原语

`deepxde.geometry.__all__`：

- 1D / ND：`Interval`、`Hypercube`、`Hypersphere`。
- 2D：`Disk`、`Ellipse`、`Polygon`、`Rectangle`、`StarShaped`、`Triangle`。
- 3D：`Cuboid`、`Sphere`。
- CSG 组合：`CSGDifference`、`CSGIntersection`、`CSGUnion`（通过 `Geometry.union/difference/intersection` 快速构造）。
- 时空：`TimeDomain`、`GeometryXTime`（空间 × 时间）。
- 任意几何：`PointCloud`（基于点云）。

所有几何继承 `Geometry` 基类（`geometry/geometry.py`），关键接口：

- `inside`、`on_boundary`、`random_points`、`uniform_points`、`random_boundary_points`、`uniform_boundary_points`。
- `boundary_normal`：用于 Neumann / Robin BC。
- `periodic_point`：周期 BC。
- `boundary_constraint_factor(x, smoothness)`：**硬约束距离函数**，用于 `u = D(x) * NN(x) + g(x)` 形式，`smoothness ∈ {"C0", "C0+", "Cinf"}`。
- `sample(n, random="pseudo"|"LHS"|"Halton"|"Hammersley"|"Sobol")`：低偏差序列采样入口。

### 5.4 IC / BC 模块

（`deepxde/icbc`）

- `BC` 抽象基类：`filter`、`collocation_points`、`normal_derivative`、`error`（抽象）。
- 内置实现：
  - `DirichletBC`: `y(x) = func(x)`。
  - `NeumannBC`: `dy/dn = func(x)`。
  - `RobinBC`: `dy/dn = func(x, y)`。
  - `PeriodicBC`: 自动在对应周期点对齐。
  - `OperatorBC`: 任意关于 `(x, y, X)` 的算子约束。
  - `PointSetBC` / `PointSetOperatorBC`: 基于离散观测点的约束（常用于反问题 / 观测数据）。
  - `Interface2DBC`: 两域界面条件。
- `IC`: 时间域的初值条件。

所有约束通过 `error(X, inputs, outputs, beg, end, aux_var)` 返回残差张量，`Data.losses_*` 负责把残差组装成 loss 列表。

### 5.5 `PDEOperator` / `PDEOperatorCartesianProd`

用于构造 **物理信息 DeepONet / PI-DeepONet** 的训练数据：将 `function_spaces`（`PowerSeries`、`Chebyshev`、`GRF`、`GRF_KL`、`GRF2D`）采样出来的函数作为输入支，配合底层 PDE 残差构成算子学习任务。

---

## 6. 自动微分（`deepxde.gradients`）

`deepxde.grad` 暴露两个核心函数：

```python
dde.grad.jacobian(ys, xs, i=None, j=None)
dde.grad.hessian(ys, xs, component=0, i=0, j=0)
dde.grad.clear()
```

特性：
- **惰性求值**：只有真正被访问时才计算对应的 `J[i, j]` / `H[i, j]`。
- **缓存复用**：同次前向得到的偏导可被多处复用，极大减少重复反向。
- **双模式**：由 `config.autodiff` 切换 `reverse`（默认）或 `forward`，实现分别在 `gradients_reverse.py` / `gradients_forward.py`。

在新后端接入时，需要在对应后端的 `gradients_*.py` 中实现 `_Jacobian` / `_Hessian` 的取数语义，并保证 `clear()` 能清空缓存。

`Model._outputs` / `_outputs_losses` 在 PyTorch / Paddle 上会在每次前向后调用 `grad.clear()`，避免跨 step 的缓存污染。

### 6.1 ZCS：Zero Coordinate Shift

`deepxde/zcs/` 提供 **面向 PI-DeepONet 的高效二阶 AD**：将空间坐标 `x` 替换为基准点 + 零偏移量 `Δx=0`，使得对 `Δx` 求导等价于对 `x` 求导，但能让多个 branch 样本共享同一个 Hessian 计算，显著降低算子学习的显存与时间开销。参考示例：
- `examples/operator/diff_rec_aligned_zcs_pideeponet.py`
- `examples/operator/stokes_aligned_zcs_pideeponet.py`

---

## 7. 神经网络模块（`deepxde.nn`）

### 7.1 运行时后端分发

`nn/__init__.py` 在被导入时读取 `backend_name` 并执行 `_load_backend(backend_name)`，从 `nn/{pytorch,tensorflow,...}/__init__.py` 中将类提升到 `deepxde.nn` 命名空间。因此用户代码只需：

```python
net = dde.nn.FNN([2, 32, 32, 1], "tanh", "Glorot normal")
```

即可跨后端复用。覆盖度在 `installation.rst` 中已注明：PaddlePaddle ≈ TF1.x > TF2.x ≈ PyTorch > JAX。

### 7.2 内置网络（以 PyTorch 后端为例）

`deepxde/nn/pytorch/__init__.py`：

- `NN`：所有网络的基类（支持 `apply_feature_transform` / `apply_output_transform` 定义输入输出变换与硬约束）。
- `FNN`：全连接网络。
- `PFNN`：并行全连接网络（每个输出一条子网络，或通过嵌套 list 精细控制层间共享）。
- `DeepONet` / `DeepONetCartesianProd` / `PODDeepONet`：算子学习网络。
- `MIONetCartesianProd` / `PODMIONet`：多输入算子学习。

TF 子包额外提供：
- `FourierNet`：多尺度 Fourier 特征全连接网络。
- `STMsFFN`：时空多尺度 Fourier 特征网络（参考 `Comput. Methods Appl. Mech. Eng.` 2021）。
- `ResNet`：残差网络。
- `MsFFN`：多尺度 Fourier 特征网络。
- `MfNN` / `MfONet`：多精度网络。

### 7.3 初始化器 / 激活函数 / 正则化

- `nn/activations.py`：`tanh`、`relu`、`sigmoid`、`sin`、`swish/silu`、`elu`、`selu`、`gelu` 等。
- `nn/initializers.py`：`Glorot uniform/normal`、`He uniform/normal`、`LeCun uniform/normal`、`zeros`、`orthogonal`、`stacked` 等。
- `nn/regularizers.py`：`l1`、`l2`、`l1+l2`，由 `Model._compile_*` 读取并加入损失。

---

## 8. 优化器（`deepxde.optimizers`）

`optimizers/__init__.py` 按后端分发，提供的名称包括：

- 1 阶：`adam`、`sgd`、`rmsprop`、`adamw`（PyTorch）。
- 2 阶：`L-BFGS`（全部后端）、`L-BFGS-B`（TF1）、`NNCG`（Nyström-preconditioned Newton-CG，仅 PyTorch）。
- 外部：TF2 经由 `tfp.optimizer.lbfgs_minimize`；TF1 经由 `scipy.optimize.minimize`；PyTorch 经由 `torch.optim.LBFGS`；Paddle 经由 `paddle.incubate.optimizers.LBFGS`。

学习率 decay（`Model.compile(decay=...)`）跨后端支持：

- TF1：`inverse time`、`cosine`。
- TF2：`InverseTimeDecay`、`CosineDecay`。
- PyTorch：`step`、`cosine`、`inverse time`、`exponential`、`lambda`。
- Paddle：`inverse time`。
- JAX：`linear`、`cosine`、`exponential`（Optax schedules）。

### 8.1 `set_LBFGS_options` / `set_NNCG_options`

`optimizers/config.py` 提供参数化入口：

```python
dde.optimizers.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-8, maxiter=15000, maxls=50)
dde.optimizers.set_NNCG_options(lr=1, rank=50, mu=0.1, updatefreq=20)
```

PyTorch / Paddle 下 L-BFGS 采用分步调用：`iter_per_step = min(1000, maxiter)`，避免单次 `closure` 过长。

---

## 9. 训练主流程：`deepxde.model.Model`

### 9.1 `__init__`

```python
Model(data: Data, net: NN)
```

持有：
- `opt_name` / `batch_size` / `loss_weights` / `external_trainable_variables` / `callbacks` / `metrics`
- 后端相关：`sess`（TF1）、`lr_scheduler`（PT/Paddle）、`opt_state` 和 `params`（JAX）
- 运行状态：`train_state: TrainState`、`losshistory: LossHistory`、`stop_training: bool`

### 9.2 `compile(optimizer, lr=None, loss="MSE", metrics=None, decay=None, loss_weights=None, external_trainable_variables=None, verbose=1)`

核心流程：

1. 解析优化器名、损失函数（`losses_module.get`）。
2. 对 TF1 自动收集全部 `tf.Variable` 作为可训练变量；其它后端尊重 `external_trainable_variables`。
3. 分发到 `_compile_tensorflow_compat_v1 / _compile_tensorflow / _compile_pytorch / _compile_jax / _compile_paddle`：
   - 构建 `outputs(training, inputs)` / `outputs_losses_train / outputs_losses_test` / `train_step`。
   - 对 TF2 使用 `@tf.function(jit_compile=config.xla_jit)`；对 JAX 使用 `@jax.jit`。
   - PyTorch 支持 L1 正则惩罚，L2 通过 `weight_decay` 传给优化器；对 `external_trainable_variables` 的 `weight_decay` 固定为 0（避免对物理参数加正则，L-BFGS 除外）。
   - JAX 在首次 `compile` 时用 `jax.random.PRNGKey(config.jax_random_seed)` 初始化参数，并将 `[net.params, ext_params]` 合并为 `self.params`。
4. 最后以字符串名解析 metrics（`metrics_module.get`）。

### 9.3 `train(iterations=None, batch_size=None, display_every=1000, callbacks=None, model_restore_path=None, model_save_path=None, verbose=1)`

- 向前兼容参数 `epochs`（已废弃，触发 warning）。
- 对非 `PDE/TimePDE`，可传 `batch_size`；对 `PDE/TimePDE` 则推荐使用 `PDEPointResampler` 回调控制样本。
- 对 `TF1 Scipy L-BFGS`：调用 `_train_tensorflow_compat_v1_scipy`（一步到位，通过 `loss_callback` 接管日志）。
- 对 `TF2 TFP L-BFGS`：`_train_tensorflow_tfp` 滚动调用直到 `converged/failed`。
- 对 `PyTorch L-BFGS / Paddle L-BFGS`：使用带 `closure` 的 step，观察 `state_dict()['state'][0]['n_iter']` 判断收敛。
- 对 `PyTorch NNCG`：走 `train_step_nncg`，不在 `closure` 内 `backward`。
- 其它（Adam / SGD / ...）：`_train_sgd` 循环 `iterations` 步，每 `display_every` 触发一次 `_test`。
- 训练完成后打印 `training_display.summary(train_state)`，按需 `save`。

### 9.4 `predict(x, operator=None, callbacks=None)`

- `operator=None`：返回网络输出。
- `operator=(x, y)` 或 `(x, y, aux_vars)`：返回算子值。常见用法为把 `pde` 作为 `operator` 以评估 PDE 残差：
  ```python
  f = model.predict(X, operator=pde)
  ```
- JAX / TF2 下会通过 `@jax.jit / @tf.function` 即时编译。
- PyTorch / Paddle 下每次调用后 `grad.clear()` 清缓存。

### 9.5 持久化

- `save(path, use_iteration_suffix=True, protocol="backend"|"pickle")`：各后端分别写 `.ckpt / .weights.h5 / .pt / .pdparams`；`pickle` 模式仅保存 `state_dict`，**不支持 restore**。
- `restore(path, device=None)`：PyTorch 支持指定 `device`。
- `print_model()`：仅 TF1 已实现。

### 9.6 `TrainState` 与 `LossHistory`

- `TrainState`：记录当前 `iteration / step`、训练/测试输入输出、最优模型快照、metrics。
- `LossHistory.append(step, loss_train, loss_test, metrics_test)`：`None` 字段自动沿用上一次值，避免可视化锯齿。

---

## 10. 回调系统（`deepxde.callbacks`）

基类 `Callback` 提供 `on_{train,epoch,batch,predict}_{begin,end}` 生命周期；`CallbackList` 负责广播。

| 回调 | 用途 |
| --- | --- |
| `ModelCheckpoint` | 定期 / 当 metric 改进时保存模型；`monitor ∈ {"train loss","test loss"}` |
| `EarlyStopping` | 在 `monitor` 无改进 `patience` 次后停止 |
| `Timer` | 预算训练时间（分钟） |
| `DropoutUncertainty` | 每 `period` 步运行 1000 次 dropout 推断，得到 `y_std_test` |
| `VariableValue` | 监视 `dde.Variable` 值，可落盘到文件（PINN 反问题标配） |
| `OperatorPredictor` | 监视任意算子（算子输入为 `(x, y)`）输出；子类 `FirstDerivative` 便捷监视一阶偏导 |
| `MovieDumper` | 沿一条直线录制训练中网络解的动画 / Fourier 谱 |
| `PDEPointResampler` | 定期重采样 PDE / BC 训练点；BC 重采样仅 PyTorch / Paddle 已实现 |

---

## 11. 损失与指标

- `losses.py::LOSS_DICT`：`MAE/MSE/MAPE/mean l2 relative error/softmax cross entropy/zero`。调用 `get(identifier)` 支持字符串与 callable。
- `metrics.py::get`：`accuracy / l2 relative error / nanl2 relative error / mean l2 relative error / MSE / MAPE / max APE / APE SD`。

---

## 12. 并行训练

仅 `tensorflow.compat.v1` 支持 Horovod 数据并行。检测环境变量 `OMPI_COMM_WORLD_SIZE > 1` 后：

1. `hvd.init()`，`world_size = hvd.size()`。
2. `tf.compat.v1.disable_eager_execution()`。
3. `rank 0` 打印 “Parallel training with N processes.”；`train_distribution` 必须为 `"pseudo"`。
4. `parallel_scaling ∈ {"weak", "strong"}` 改变 `num_domain / num_boundary` 的语义（每 rank vs 总量）。
5. 优化器通过 `set_hvd_opt_options(compression, op, backward_passes_per_step, average_aggregated_gradients)` 包装。
6. `Model._train` 中 `sess.run(hvd.broadcast_global_variables(0))` 广播初值。

其它后端如需并行，可自行对接 `torch.distributed` / `jax.pmap`，但非官方支持。

---

## 13. 开发流程与贡献指南

### 13.1 环境准备

```bash
# 1. 克隆
$ git clone https://github.com/lululxvi/deepxde.git
$ cd deepxde

# 2. 建议虚拟环境
$ python -m venv .venv && source .venv/bin/activate

# 3. 安装（editable）
$ pip install -e .

# 4. 至少安装一个深度学习后端
$ pip install torch         # 或 tensorflow / jax flax optax / paddlepaddle
```

如需本地构建文档：

```bash
$ pip install -r docs/requirements.txt
$ cd docs && make html
```

### 13.2 运行示例

```bash
$ cd examples/pinn_forward
$ DDE_BACKEND=pytorch python Poisson_Dirichlet_1d.py
```

每个示例文件顶部注释会标明所支持的后端，如：
```python
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
```

### 13.3 代码规范

项目尚未包含自动化 lint 配置，但从现有代码风格可以推断出以下习惯：

- PEP 8 + 4 空格缩进 + 行宽 ≈ 88（部分 docstring 更宽）。
- 采用 Google 风格 docstring（`Args:` / `Returns:` / `Warning:` / `References:` / `Examples:`）。
- 公共 API 通过 `__all__` 显式声明；跨后端类通过 `_load_backend` 机制提升到 `deepxde.nn`。
- 类型提示局限于新代码（如 `boundary_constraint_factor` 使用 `Literal`）；老代码以 docstring 类型为主。

### 13.4 新增一种 PDE / 约束

1. 如果是新 BC，在 `deepxde/icbc/boundary_conditions.py` 新增 `class XxxBC(BC)`，实现 `error()`；在 `__init__.py` 的 `__all__` 中注册。
2. 若涉及新损失形式，考虑放进 `deepxde/losses.py::LOSS_DICT`。
3. 为示例新增一个 `examples/pinn_forward/<name>.py`，并在 `docs/demos/pinn_forward.rst` 追加条目。

### 13.5 新增后端

以 `pytorch` 为模板：

1. 在 `deepxde/backend/<name>/` 实现 `backend.py` 中规约的所有 API。
2. 在 `deepxde/backend/__init__.py::load_backend` 自动会装配；如需补全 `backend_message` 的提示信息，同步更新。
3. 在 `deepxde/nn/<name>/` 下实现至少 `NN / FNN / DeepONet` 等必需网络类。
4. 在 `deepxde/optimizers/<name>/` 实现 `get()`（返回 optimizer + lr scheduler）与 `is_external_optimizer()`。
5. 在 `deepxde/gradients/<name>/` 实现 `jacobian` / `hessian` / `clear`（与 `gradients_reverse.py` 相似）。
6. 为 `deepxde/model.py` 的 `_compile_<name>` / `_outputs / _outputs_losses / _train_step` 增加分支。
7. 补充 `deepxde/config.py` 中随机种子 / 默认 dtype / 混合精度等分支。

### 13.6 新增网络

1. 在 `deepxde/nn/<backend>/` 新建 `<name>.py`，继承 `NN`，实现 `forward` 与属性（`regularizer`、`_input_transform`、`_output_transform`、`auxiliary_vars`）。
2. 在 `nn/<backend>/__init__.py` 将其加入 `__all__` 并从对应模块 import。
3. 若是跨后端的通用 Net，建议同时在 `pytorch/tensorflow/paddle` 等子包实现。
4. 如暴露给 PI-DeepONet，需要保证 `apply_output_transform` / `apply_feature_transform` 可用。

### 13.7 发起 PR

- Issue：先到 [GitHub Issues](https://github.com/lululxvi/deepxde/issues) 搜索是否已有讨论。
- 分支：基于 `master` 开分支。
- 自检：至少在 1 个后端跑通相关示例；对跨后端改动，逐一验证。
- 文档：涉及公共 API，更新 `docs/modules/` 下的 RST；涉及示例，更新 `docs/demos/`。
- 沟通：复杂特性先在 Issues / Discussions / Slack 中和维护者同步。

详见仓库根 `README.md` 的 *Contributing to DeepXDE* 章节。

---

## 14. 常见调试技巧

| 现象 | 检查方向 |
| --- | --- |
| `AttributeError: module 'deepxde.nn.pytorch' has no attribute 'XXX'` | 该网络在当前后端未实现，可切换后端或自行补齐。 |
| `API "XXX" is not supported by backend "YYY"` | `_gen_missing_api` 抛出，检查 `backend/YYY/*.py` 是否缺失对应函数。 |
| L-BFGS 提前停止 | `dde.config.set_default_float("float64")`，再次尝试；或调大 `ftol/gtol`。 |
| 损失 NaN 后停止训练 | `Model._test` 检测到 NaN 会置 `stop_training=True`；检查 PDE 残差表达式与 BC 定义。 |
| `num_bcs changed` | `PDEPointResampler` 重采样后 BC 点数变化；需要在回调结束后调用 `model.compile()` 重新构图。 |
| 随机结果不可复现 | `dde.config.set_random_seed(seed)`，TF2 还需同一 Python 进程下单次运行。 |

---

## 15. 参考资料

- 主论文：Lu, Meng, Mao, Karniadakis. *DeepXDE: A deep learning library for solving differential equations*. SIAM Review 63(1): 208-228, 2021.（见 `README.md` 引用条目）
- 在线文档：<https://deepxde.readthedocs.io>
- 讨论：GitHub Discussions / Slack（详见 `README.md`）

如需更细粒度的 API 说明，可本地构建 Sphinx 文档：`cd docs && make html`，在 `docs/_build/html/index.html` 浏览 `modules/` 下自动生成的 API 参考。
