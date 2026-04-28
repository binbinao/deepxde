# DeepXDE 指南中心

本目录包含了对 DeepXDE 项目进行深度挖掘后整理的三份系统化文档，面向不同的读者群体：

| 文档 | 适用人群 | 主要内容 |
| --- | --- | --- |
| [`developer_guide.md`](./developer_guide.md) | 框架开发者 / 贡献者 | 项目架构、源码分层、后端抽象、扩展与贡献指南 |
| [`user_guide.md`](./user_guide.md) | 科研用户 / 算法工程师 | 安装配置、核心 API、典型工作流、常见问题 |
| [`tutorial.md`](./tutorial.md) | 所有学习者 | 从零到精通的 10 章递进教程（PINN → DeepONet → 进阶技巧） |

> 项目主页：<https://github.com/lululxvi/deepxde>  ·  官方文档：<https://deepxde.readthedocs.io>

## 文档生成说明

- 基于仓库 `deepxde/`（框架源码 116 个 Python 文件）、`examples/`（84 个示例，含 PINN 前向 / 反向、算子学习、IDE / fPDE）、`docs/` 目录下的 RST 资料综合挖掘。
- 覆盖的源码子包：`backend`、`config`、`data`、`geometry`、`gradients`、`icbc`、`nn`、`optimizers`、`callbacks`、`model`、`zcs`。
- 所有示例代码均来自本仓库 `examples/` 目录，可直接运行。

阅读建议：
1. 新用户优先阅读 `user_guide.md` 并结合 `tutorial.md` 第 1–4 章动手实践。
2. 希望解决特定 PDE / 反问题 / 算子学习任务的用户，直接跳到 `tutorial.md` 对应章节。
3. 希望扩展后端、新增网络结构或贡献代码的开发者，阅读 `developer_guide.md`。
