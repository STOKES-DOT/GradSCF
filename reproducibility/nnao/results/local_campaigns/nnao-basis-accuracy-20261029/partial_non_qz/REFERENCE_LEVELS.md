# 数据库参考能量的计算级别

| 子集 | 本次条数 | 参考计算级别 | 原始来源 |
|---|---:|---|---|
| ISO34 | 6 | **W1-F12 复合波函数方法** | Goerigk 等，PCCP 19, 32184–32215 (2017)，§2.2.6；[DOI](https://doi.org/10.1039/C7CP04913G) |
| S22B | 6 | **estimated CCSD(T)/CBS** | Marshall、Burns、Sherrill，JCP 135, 194102 (2011)；[DOI](https://doi.org/10.1063/1.3659142) |

ISO34 采用 GMTKN55 更新后的 W1-F12 理论参考值，替代旧版本的反向修正实验值。
W1-F12 是复合协议，不能简单标成某一个有限基组的 CCSD(T) 单点。

S22B 是修订的高精度相互作用能基准，以 MP2 的 CBS 外推结果加上有限基组的
CCSD(T)−MP2 修正来估计 CCSD(T)/CBS。这里保留“estimated”限定。
本次使用数据库发布的数值；不把我们尚未完成的 aug-cc-pVQZ 计算当作数据库参考。

原始 .res/.bib 来自固定上游 commit `ccabc16c8c4677fc60c19870ae96ac25405844a2`，
保存在 references/；所选 12 个参考值均已逐项核对。
S22 原始 .res 使用正解离能，本次统一取负号为相互作用能。

我们的结果为 frozen-core DF-MP2 或 omegaB97M-V（main/NLC grid level 均为 3）的
电子相对能量，没有追加 CP/BSSE、零点能、热力学或溶剂修正。
散点纵轴因此衡量方法与基组共同造成的相对能量误差，并非纯粹的同方法基组极限误差。
