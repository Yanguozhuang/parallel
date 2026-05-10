]633;E;sed -n '1,4p' README.md;63186d17-1134-4acb-a408-ffead675fdcb]633;C# 矩阵SVD框架（简要说明）

你可以认为这个文档是一个“导航”，
它负责让你快速知道每个文件负责什么，以及它们是怎么串起来工作的。


## 文件说明

### `matrix.h`
基础矩阵类。
- 提供矩阵存储、元素访问（`at`）、加法、乘法。
- 提供随机矩阵生成（`Matrix::random`）。
- 提供打印接口（`print`），方便调试和展示结果。

### `givens.h`
Givens 旋转工具函数。
- 计算旋转参数（`c, s, r`）。
- 提供 2 元素向量的左乘/右乘旋转辅助函数。


### `bidiagonalization.h`
`bidiagonalization.cpp` 的接口声明。
- 对外暴露 `to_bidiagonal(const Matrix&, Matrix&, Matrix&)`。

### `bidiagonalization.cpp`
上二对角化实现。
- 输入：`m x n` 矩阵 `A`（保证 `m >= n`）。
- 通过左右交替 Householder 变换，把 `A` 化为上双对角 `B`。
- 同时累计正交矩阵 `U`、`V`，满足：`A = U * B * V^T`。

这是进行 GKH 迭代前你需要做的。

### `gkh.h`
GKH 迭代模块对外接口声明。
- 主要接口：`gkh_svd_from_bidiagonal(...)`。
- 作用：从“已上二对角化的 `B`”继续迭代，收敛到奇异值结果。

### `gkh.cpp`
GKH（Golub-Kahan）迭代实现。
- 在保持 `A = U * B * V^T` 不变的前提下，对 `B` 做 bulge chasing。
- 包含活动块划分，`Wilkinson`偏移计算，`bulge chasing`以及收敛检测几个环节。
- 收尾时把奇异值整理为“非负且降序”，并同步调整 `U`、`V`。

你可以这样理解：对一个普通矩阵，先经过`bidiagonalization.cpp` 先把问题化简，`gkh.cpp` 再把化简后的矩阵做完。

### `main.cpp`
当前主测试程序（包含计时）。
- 覆盖 5 个样例（固定值、小中规模随机、近秩亏损、大规模随机）。
- 验证重构误差、正交误差、对角结构与排序。
- 分别统计并打印：
  - 上二对角化耗时
  - GKH 迭代耗时
- 随机样例支持命令行种子，便于复现和调参。


## 代码执行（你需要理清顺序）

1. 在 `main.cpp` 构造测试矩阵 `A`。
2. 调 `to_bidiagonal`（`bidiagonalization.cpp`）得到 `U, B, V`。
3. 调 `gkh_svd_from_bidiagonal`（`gkh.cpp`）让 `B` 收敛到对角。
4. 检查 `A` 与 `U * S * V^T` 的重构误差（其中 `S` 就是收敛后的 `B`）。

## 脚本用法

在目录下，使用 `bash test.sh [LAB] [NODES] [CORES] [-O <opt>] [-s <seed>]`，其中`-O`为优化等级，`-s`为矩阵随机初始化的种子号。

请注意：

1. 请勿随意修改main.cpp下的代码，如需修改请务必联系助教，这可能涉及你的分数；

2. 请勿修改test.sh以及qsub.sh文件。

3. 请使用test.sh进行测试和提交，如服务器未收集到你的测试数据，你可能面临再次询问。
改框架test.sh用法：bash test.sh [LAB] [NODES] [CORES] -O O2 -s 20260410

该命令前三个参数必填，LAB为实验编号（SIMD实验编号为1），NODES为申请节点数，CORES为每个节点申请核心数。
-O与-s为可选参数，其中-O为优化等级（缺省默认O2优化），-s为矩阵初始化种子（缺省使用默认种子），运行输出存放至test.o文件。

### 如果你发现哪里有问题，或者对框架有大规模的改进，欢迎飞书联系助教邹博闻（2312251）


