# qec-certify 工业价值推进计划书

- **日期：** 2026-11-23
- **状态：** 提议执行
- **适用范围：** 解码器研究、circuit-level 基准、噪声鲁棒性、最优性参考、实时部署
- **目标读者：** 项目维护者、量子纠错研究人员、机器学习与解码器工程师、硬件/控制系统合作者

> 本计划依据 Google Quantum AI 和 IBM Quantum 的公开工程方向制定，不代表其内部招聘、采购或合作标准，也不保证获得任何公司的认可或合作。所有性能目标均为本项目建议的验收条件，而不是上述公司的公开录取线。

## 1. 执行摘要

本项目不应以“再训练一个更大的神经网络”为主要目标，而应升级为：

> **面向真实电路噪声的量子纠错解码器评测、优化与实时部署平台，并用精确参考、数值近似或解析界限说明解码器距离最优还有多远。**

项目最有价值的差异化资产是：

1. 小规模精确 ML 参考和 gap-to-optimal 评估；
2. 查表对照、置信区间和负面结果所体现的严格评估方法；
3. 噪声结构描述符与 decoder 表现之间的联系；
4. 将准确性、速度、资源、鲁棒性和最优性参考统一起来的可能性。

未来 90 天的优先级依次是：

1. circuit-level surface-code benchmark；
2. 强传统基线和统一评估；
3. 噪声失配、校准漂移和实时指标；
4. 在严格验收条件下测试一个轻量 recurrent Transformer 或混合 decoder。

Transformer 保留为受约束的研究线，但不是当前工程主线。近期更高的投入产出比预计来自 noise-aware MWPM/BP、混合解码、噪声失配评估和流式实时原型。

## 2. 当前基础与主要差距

### 2.1 已有基础

截至本计划日期，仓库已经具备：

- Python `src` 包结构、YAML 配置、固定随机种子和实验复现脚本；
- repetition code 和 rotated surface code；
- code-capacity depolarizing、biased Pauli 和最近邻 correlated noise；
- MWPM、correlated MWPM、查表 decoder、MLP；
- full-table exact ML 和逐 shot SweepML；
- LER、Wilson 置信区间、gap ratio、gap closed 和部分噪声描述符；
- M1–M3 的部分实验记录，包含有益的负面结果；
- Ruff、pytest 和 GitHub Actions；2026-11-23 本地验证为 281 项测试通过。

现有实验已经说明：

- d=3 上 MLP 接近 ML 很大程度可以由查表记忆解释；
- d=5 上 MLP 在部分噪声和训练数据条件下表现出泛化，但没有达到 exact ML；
- 低噪声和较大代码距离下仍存在数据量与训练量问题；
- MWPM 与最优 decoder 的差距会随代码距离变化，小代码结果不能直接外推。

### 2.2 第一阶段首个已实现增量（2026-11-23）

- 新增 Stim 标准 rotated-surface circuit-level memory，统一覆盖 Clifford、idle、measurement 和 reset 噪声；d=3/5/7 的 detector 数量和 graphlike distance 均有回归测试。
- 新增 `configs/m4_circuit_level_d3.yaml`，通过现有 baseline CLI 运行 MWPM、correlated MWPM、BP+OSD 和 Union-Find。
- BP+OSD/Union-Find 直接消费任意 DEM 的校验矩阵和错误机制，逻辑 observable 由同一 DEM 重建；`ldpc>=2.4` 已成为显式项目依赖。
- 结果新增 `experiment`、`rounds` 和带 `independent_equal_hazard` 假设标记的等效每轮 LER。
- 可选 `latency_shots` 报告 offline batch=1 的 p50/p95/p99 延迟和吞吐，不改变未启用性能测量时的旧实验复现行为。
- 后续增量已加入通道独立强度、seeded lognormal qubit 异质性和首轮到末轮的线性漂移；`configs/m4_circuit_level_scaling.yaml` 覆盖 d=3/5/7 的均匀/异质 × 无漂移/有漂移对照。
- 已加入因果 replay streaming 参考接口：逐轮输入、未来置零、最终预测与离线一致，并报告逐轮延迟、吞吐和固定到达速率下的 backlog 估计。该适配器每轮重放历史，总工作随轮数二次增长，是正确性/测量基准而不是优化实时 decoder；Union-Find 因前缀可实现性限制明确标记为不支持。
- 已加入固定窗口 MWPM/correlated MWPM：每轮使用最近 W 轮的局部 DEM，matching 规模不再随完整历史增长；全窗口必须与离线一致，部分窗口单独报告 `streaming_ler`。首次 W=3 对照显示部分窗口的 `projected_dem_boundary` 会使 LER 严重退化，因此该近似只保留为测量/负面基线，不能用于生产。下一步需要边界锚定或精确增量 matching，而不是继续优化朴素投影的速度。
- 新增 `delayed_commit` 边界锚定对照：利用 matched edges，仅提交窗口最老轮中完全内部的分量，较新和跨界分量留在尾部；全窗口仍要求与离线一致。预声明 W 扫描和逐 shot McNemar 对照显示所有部分窗口均显著退化，d=7 即使 W=5 仍有约 0.30 LER，因此停止扩展这条启发式作为生产候选，仅保留为测量/负面基线。后续需要更强边界状态、精确增量 matching 或不同算法。
- qLDPC 首个增量已接入：固定 commit 和 SHA-256 的公开 BB [[18,4,3]] 与 gross [[144,12,12]] Stim 电路、通用 DEM BP+OSD，以及公开 Rust Relay-BP。小码配对比较未显示显著准确率差异（p=0.281）；gross Relay-BP 为 0/1000，但默认参数整 shot p99 约 1.79 s，不能声称零 LER 或实时能力。上游许可证和“非受支持 IBM 产品”声明保留在 `codes/data`，结果由独立 `eval.qldpc` runner 生成。
- 已加入 validation-only Relay-BP 参数选择：候选、选择规则、validation/test seeds 和最终 test 全部保存在 tuning JSON 中，避免 test-set 调参。预注册实验选中 `paper_like`，独立 test 为 Relay-BP 0/300、BP+OSD-0 27/300，配对 p=1.49e-08；这只证明优于当前 OSD-0，不证明优于论文高成本 CS-10。Gross BP+OSD 预算探测显示 OSD-1 单 shot 约 10.7 s 且峰值内存约 4.4 GB，整批 OSD-0 也有明显重尾，实时性仍未建立。
- 固定 `paper_like` 的跨 p/seed 扫描已完成：p=0.001 两者零分歧；p=0.003 和 p=0.005 的两个 seed 中 Relay-BP 均显著优于 BP+LSD-0，支持参数的初步鲁棒性。长时间实验已加入逐 cell checkpoint/resume。该结果仍不覆盖高成本 BP+OSD-CS-10、真实硬件数据或实时部署；整 shot 延迟和重尾仍显著。
- 已加入预注册 validation p99 延迟预算选择：达标候选先比准确率，全不达标时选最低 p99 并显式 `met: false`，避免把低延迟候选误称为实时。100 ms 实验中仅 `single` 达标，但独立 test LER=0.67，BP+LSD=0.04，说明小预算 Relay 配置速度合格而准确率不可用。后续必须同时预声明准确率约束或报告 Pareto frontier，不能只优化延迟。
- 已加入 Pareto runner：validation 按错误数和 p99 做非支配筛选，所有 frontier 候选共享同一 test shots 和同一 BP baseline。结果显示准确率断崖：single/five LER=0.67/0.385，paper_like=0 但 test p99=789 ms，没有 100 ms 级低 LER 甜点。下一步应评估固定点/低精度 Relay，而非继续减少 ensemble。
- 已接入 F32/I32/I64 Relay 实现和同 shots 精度对照，失败候选回退 no-flip 并记录成功率，避免把不满足 syndrome 的候选当成有效纠正。gross p=0.003 实验中三者均 0/200、成功率 100%、两两预测零分歧；I32 mean/p99 为 156/291 ms，F32 为 234/799 ms，I64 为 170/316 ms，显示整数实现加速潜力。该结果仅限当前电路和参数，整数缩放不冒充 FPGA 定点证明。
- 已加入跨 p/seed 精度扫描 runner：F32/I32 在每单元共享 shots，直接比较 logical decisions、成功率、LER 和延迟，并逐 cell checkpoint。p=0.001/0.003 保持零分歧，但 p=0.005 出现 20/25 个分歧且 I32 更慢；其较低错误数不显著（p=0.064/0.307），因此整数化不能视为普遍等价或普遍加速。下一步应做失败感知回退或 validation-only 量化调参。
- 已加入失败感知 `relay_bp_fallback`：仅 Relay `success=False` 的 shots 调用 BP+LSD。p=0.005 实验中回退纠正 7 个错误且不引入新错误（配对 p=0.015625），但 18% 回退率使混合 p99 达 14.2 s；准确率改善并未解决实时性，下一步需要更便宜的回退或超时/并行策略。
- 新增短 Relay 独立 seed 重试作为低成本回退：p=0.005 时纠正 5 个错误且不引入新错误，但 p=0.0625 不显著，31/36 次重试仍失败，延迟继续增加。它比 BP+LSD 回退便宜但证据和 LER 更弱，不作为生产推荐；后续需要更好的算法、validation-only 调参或系统级测量。
- Relay 原生串行 batch 存在状态残留，已改为 fresh native decoder per shot；并行通过公开 `ObservableDecoderRunner` 的独立 clone 实现。修正后串行/并行预测、成功率和迭代数完全一致；整批耗时在 p=0.003/0.005 分别加速约 1.48x/1.34x，但 batch=1 尾延迟无实质改善。此前 Relay batch 数字需要按修正后实现复核。
- Fresh-state 核心复核已完成并写入独立 `_fresh` 输出：gross 调参最终配对结果不变；小 BB Relay 36/1000，仍无显著优势；F32/I32 在 p=0.003 零分歧，在 p=0.005 有 19 个分歧且 I32 更慢。旧 stateful-batch 数字仅保留为历史，后续以 fresh 输出为准。
- 逐 shot 剖析已定位瓶颈：原生 BP/Relay 消息传递主导成本；p=0.005 失败 shots 全部跑满 3680 次迭代，平均约 6.6 s，成功 shots 也有重尾。串行初始化约 37 ms、Python 开销不足 1 ms，并行 batch=1 无实质改善。后续优化应针对原生迭代成本、非收敛和成功 shot 尾部，而非 Python 包装。
- 受控提前接管实验完成：三种 Relay+BP+LSD 混合方法最终失败均为 0，且显著优于 BP+LSD；但 10/20/60 组候选的回退率为 58%/46%/19%，p99 仍为 6.5/7.5/11.9 s。提前接管没有解决尾延迟，停止继续增加此类预算组合，转向原生迭代或根本不同的非收敛策略。
- Relay alpha 对照完成：alpha=0.90 将 p=0.005 错误数从 37 降至 18、成功率从 81.5% 提至 91%、平均迭代从 1239 降至 806，平均延迟从 1642 ms 降至 1046 ms；但失败 shots 仍跑满预算，p99 仍约 6.3 s。alpha 改善平均收敛而非尾延迟，需要跨 p/seed 复核和独立尾部策略。
- `alpha=0.90` 跨 p/seed 复核完成：低 p 下零错误、零分歧；高 p 两个 seed 中错误数分别从 37/50 降至 18/20，成功率从 81.5%/75.5% 提至 91%/90%，配对 p=2.10e-05/2.27e-07。平均收敛收益稳定，但失败 shots 仍跑满预算，p99 仍为 5.7--6.3 s，尾部问题未解决。
- 固定 alpha=0.90 的尾部预算扫描完成：680/1280/2480 上限虽将 p99 降至约 1.2/2.1/4.0 s，但所有截断配置在两个 seed 上均显著损害准确率，包括 2480 上限。当前实现没有免费尾部截断空间，停止该参数路线，后续需要原生迭代加速或不同解码机制。
- 真实硬件数据、优化增量 streaming decoder/实时原型仍未完成。异质性和漂移目前是可复现模拟参数，尚未由硬件校准数据标定；固定到达速率是模拟参数，不是硬件实测。
- `ldpc 2.4.1` 的 Union-Find 对不可实现 syndrome 可能不返回，当前接口明确要求输入可由 DEM 实现；BP+OSD 的 OSD 阶数已在包装层限制到有效机制数，避免原生库越界崩溃。

### 2.3 工业落地的关键缺口

- 当前主要结果仍是 code-capacity，而不是 circuit-level；
- 缺少真实硬件 syndrome 和校准数据验证；
- 缺少 BP+OSD、Relay-BP、低延迟 decoder 等强基线；
- 缺少统一的 streaming、p99 延迟、吞吐、backlog 和硬件资源评估；
- 缺少跨设备、跨校准时间和噪声失配测试；
- 没有实时 FPGA/ASIC/优化 CPU 原型；
- qLDPC 支持尚未形成；
- 解析或可扩展的最优性界限仍未实现；
- 精确 ML 的指数复杂度不能直接扩展到工业规模。

因此，当前项目是一个可信的小规模研究基准，而不是生产 decoder 或硬件可靠性证明工具。

## 3. 北极星目标与产品定位

### 3.1 北极星问题

项目每次重要实验都应帮助回答：

> **在明确的 code、circuit、噪声和计算预算下，哪个 decoder 最好；它比最强基线好多少；距离已知最优还有多远；结论能否在未见过的噪声和设备上复现？**

### 3.2 产品形态

最终交付应由四层组成：

1. **Benchmark 层：** 固定版本 circuit、噪声、数据划分、baseline 和指标；
2. **Reference 层：** exact ML、近似最优参考、上下界及适用假设；
3. **Decoder 层：** noise-aware 传统方法、轻量学习方法和混合 decoder；
4. **Deployment 层：** 流式 API、性能测量、回退机制和至少一种实时原型。

### 3.3 两条公开方向对应路线

| 路线 | 重点 | 必须展示的证据 |
|---|---|---|
| Google Quantum AI 方向 | Surface code、circuit-level、距离扩展、实时 decoding、设备噪声 | LER per round、随距离的错误抑制、噪声失配鲁棒性、p99 延迟和 backlog |
| IBM Quantum 方向 | qLDPC/BB code、BP 系列、紧凑实时 decoder、FPGA/ASIC | accuracy、speed、compactness、flexibility，以及与 BP+OSD/Relay-BP 的比较 |

两条路线共享 benchmark、reference 和评估基础设施。工程顺序上先做 surface code，因为现有基础最完整；qLDPC 在其后扩展，避免同时全面铺开。

## 4. 总体原则

1. **Circuit-level 优先。** 不再用 code-capacity 结果支持真实硬件结论。
2. **强基线优先。** 新模型不能只与普通 MWPM 比较。
3. **准确率不是唯一指标。** 同时报告速度、资源、样本效率和鲁棒性。
4. **先 profile 再优化。** 不在没有瓶颈数据时重写整个项目。
5. **保留 Transformer，但设置停止条件。** 不预先假设更大模型更有效。
6. **区分参考与部署。** Exact ML 是离线标准答案，不是可扩展在线 decoder。
7. **不隐藏负面结果。** 如果学习方法不胜出，转向 noise-aware 传统方法或混合方法。
8. **不夸大证书。** 小代码上的精确最优不等于大代码或真实硬件上的全局最优。

## 5. 工作流与任务清单

### WS1：Circuit-level Surface Code

- [ ] 用 Stim 实现多轮 surface-code memory experiment。
- [ ] 覆盖 measurement、reset、单双量子比特 gate 和 idle 错误。
- [ ] 首先支持 d=3/5/7，验证后扩展到 d=9/11。
- [ ] 支持不同 qubit、gate、measurement 的异质错误率。
- [ ] 支持空间相关、时间相关和校准漂移配置。
- [ ] 明确标记 code-capacity、phenomenological、circuit-level 和 hardware data。
- [ ] 为 circuit、DEM、detector、observable 和采样添加回归测试。
- [ ] 固定数据生成配置和版本，防止训练/验证/测试泄漏。

**交付物：** 固定版本、可复现的 circuit-level benchmark suite。
**验收条件：** 所有 decoder 在完全相同的 circuits、shots 和噪声信息下比较，结果带采样置信区间和训练种子变化。

### WS2：强基线体系

- [ ] 保留普通 MWPM 作为基础参照。
- [ ] 完善 correlated/noise-aware MWPM，使用 DEM 和校准噪声权重。
- [ ] 引入经过验证的 BP+OSD 实现，作为高准确率参考。
- [ ] 引入 Union-Find 或 local clustering，作为低延迟参照。
- [ ] 保留查表 decoder，用于识别记忆而不是泛化。
- [ ] 保留 exact ML/SweepML，用于小规模标准答案和一致性测试。
- [ ] 为大规模离线参考增加清楚标注的高成本近似方法。
- [ ] qLDPC 阶段加入 Relay-BP；使用公开实现前确认版本、许可证和正确性。
- [ ] 所有方法使用统一 decoder 接口，并区分离线与 streaming 能力。

**交付物：** 可一键运行的 baseline matrix。
**验收条件：** 新结果至少与当前最强且适用于该 code/noise 的基线比较；同时保留低延迟基线，避免只对准确率有利的比较。

### WS3：统一工业指标与实验协议

- [ ] LER per round、logical error per logical qubit 和置信区间。
- [ ] 多训练种子的均值、标准差、范围和最差结果。
- [ ] LER 随代码距离、物理错误率和 syndrome 轮数的变化。
- [ ] batch=1 streaming 吞吐和 p50/p95/p99 延迟。
- [ ] 长时间运行时的 syndrome backlog。
- [ ] 训练样本量、微调样本量、训练时间和推理成本。
- [ ] 模型大小、内存和 CPU/GPU/FPGA 资源。
- [ ] 噪声失配、跨时间和跨设备性能。
- [ ] 分离 shot sampling 不确定性与训练随机种子不确定性。
- [ ] 结果表由脚本生成，禁止手工填写 benchmark 数字。

**交付物：** 统一结果 schema、报告生成器和性能测量工具。
**验收条件：** 每份重要报告同时包含准确率、速度、资源、统计不确定性和适用范围。

### WS4：噪声失配与设备适配

- [ ] 在 `p_train` 训练，在多个未见过的 `p_test` 测试。
- [ ] 训练使用平均噪声，测试使用异质设备噪声。
- [ ] 加入未见过的时间和空间相关模式。
- [ ] 模拟校准漂移和错误率突变。
- [ ] 比较零样本迁移、少量微调和完整重训。
- [ ] 建立不参与模型选择的固定 robustness test set。
- [ ] 提供真实硬件 syndrome/calibration 数据导入接口。
- [ ] 有公开数据时，报告跨日期和跨设备结果；没有数据时明确标注缺口。

**交付物：** 固定 robustness benchmark 和设备数据适配器。
**验收条件：** 工业结论至少包含一个完全独立、未参与模型选择的外部测试集；若无法获得硬件数据，不得声称完成硬件验证。

### WS5：Noise-aware 传统 Decoder

- [ ] 从校准参数估计 DEM 和 decoder 权重。
- [ ] 动态更新 MWPM/BP 的错误概率。
- [ ] 研究在线噪声估计和漂移适应。
- [ ] 量化不同校准数据量带来的收益。
- [ ] 比较 noise-aware MWPM/BP 与神经网络。
- [ ] 测试噪声估计错误时的稳定性和回退策略。

这是近期优先级最高的算法方向之一，因为它可能比大型神经网络更容易解释、验证和部署。

### WS6：受约束的 AI 与混合 Decoder

- [ ] 保留一个轻量因果 recurrent/local Transformer 候选。
- [ ] 使用逐轮处理、固定大小状态和局部或滑窗 attention。
- [ ] 不读取未来 syndrome，支持 streaming。
- [ ] 可输入校准参数、DEM 权重和噪声漂移特征。
- [ ] 输出校正结果及不确定度。
- [ ] 不同时开发多个大型 GNN/Transformer 架构。
- [ ] 实现“便宜 decoder 为主、高成本 decoder 兜底”的混合路径。
- [ ] 设置超时和传统 decoder 回退。
- [ ] 通过蒸馏、剪枝和量化评估紧凑部署。
- [ ] 做消融实验，区分模型结构、噪声特征、数据量和训练策略的贡献。

**优化目标：** 固定平均与 p99 延迟预算下的最低 LER，而不是不计成本地降低 LER。

### WS7：最优性参考与证书

- [ ] 小代码继续使用 exact ML 给出无采样误差的 exact gap。
- [ ] 将 full-table ML 与 SweepML 的一致性纳入持续测试。
- [ ] 大代码使用明确标注的高成本近似参考或界限。
- [ ] 研究至少一种受限噪声族上的解析下界。
- [ ] 每份证书注明 code、circuit、噪声假设、decoder 输入和统计置信度。
- [ ] 区分 exact、numerical、sampled 和 analytic 结果。
- [ ] 不将小规模最优性外推为大规模或真实硬件的全局最优证明。

**交付物：** 与实验报告自动关联的 certificate manifest。

### WS8：实时部署与系统工程

- [ ] 定义逐轮 syndrome 输入、内部状态和 logical correction 输出的 streaming API。
- [ ] 分离离线正确性测试与在线性能测试。
- [ ] 先 profile，再决定是否用 C++/Rust 优化热点。
- [ ] 测量延迟随代码距离和轮数的增长。
- [ ] 根据目标硬件周期预先定义延迟和吞吐预算。
- [ ] 至少完成一种优化 CPU、FPGA 或其他实时原型。
- [ ] FPGA/ASIC 路线报告资源、时钟、吞吐和量化后的 LER 损失。
- [ ] 测试超时、异常输入、回退和长时间运行稳定性。

**验收条件：** 在约定负载下不会持续增长 backlog，p99 延迟满足预先定义的预算，精度损失和资源成本可量化。

### WS9：qLDPC 扩展

- [ ] 将 decoder 接口从 surface code 抽象到通用 DEM/qLDPC。
- [ ] 接入一种公开 qLDPC/BB code，并验证 circuit 和 logical observable。
- [ ] 比较 BP+OSD 与 Relay-BP。
- [ ] 同时报告 accuracy、speed、compactness 和 flexibility。
- [ ] 评估量化、剪枝和蒸馏对 LER 的影响。
- [ ] 给出 FPGA 资源估算或原型。
- [ ] 明确区分 quantum memory 与 logical operations 的实验范围。

### WS10：可复现交付与外部采用

- [ ] 一条命令复现数据、训练、评估和图表。
- [ ] 发布固定版本配置、数据 manifest、模型和结果文件。
- [ ] 提供 CPU quickstart，GPU 只作为大规模训练选项。
- [ ] 使用锁定环境或容器，避免路径和依赖失效。
- [ ] CI 覆盖单元测试、circuit-level 回归、结果 schema 和 streaming API。
- [ ] 保留负面结果和失败配置。
- [ ] 编写技术报告，包含最强基线、统计方法、噪声失配、延迟、资源和局限性。
- [ ] 为新 code、noise 和 decoder 编写接入指南。

## 6. Transformer 的继续与停止条件

### 6.1 当前决定

保留 Transformer 作为研究候选，但不把它作为前 90 天的主要工程投入。现有 MLP 结果不足以证明 Transformer 无效，也不足以支持直接训练大型模型。

### 6.2 继续扩大投入的条件

至少满足一项，并且相对最强适用基线在独立测试集上可复现：

- [ ] 相同延迟预算下显著降低 LER；
- [ ] LER 相当时显著降低延迟或资源；
- [ ] 噪声失配和校准漂移下显著更稳定；
- [ ] 新设备所需微调数据显著更少。

### 6.3 暂停或停止的条件

- 只在 d=3 或 code-capacity 上获胜；
- 只赢普通 MWPM，无法赢过 noise-aware MWPM/BP；
- 优势无法跨训练种子复现；
- 延迟随规模增长到无法满足 streaming；
- 充分训练后仍只相当于查表或简单噪声拟合；
- 资源成本超过其准确率或鲁棒性收益。

停止后保留配置和负面结果，将主线转向 noise-aware 传统方法或混合 decoder。

## 7. 时间表与阶段门

时间按小型 2–4 人团队粗略估计；个人执行时以阶段门为准，不为了日期跳过验证。

| 阶段 | 建议时间 | 主要工作 | 阶段门 |
|---|---|---|---|
| Phase 0：指标冻结 | 第 1–2 周 | 明确 circuit、噪声、baseline、结果 schema 和延迟预算定义 | 实验协议评审通过，所有结论范围可机器读取 |
| Phase 1：可信基准 | 第 3–6 周 | Circuit-level d=3/5/7、noise-aware MWPM、BP+OSD、低延迟基线 | 同 circuit/shots 对比，测试和报告自动生成 |
| Phase 2：鲁棒性与轻量模型 | 第 7–12 周 | 噪声失配、漂移、streaming 指标、一个轻量或混合模型 | 在独立测试集上回答是否优于最强基线 |
| Phase 3：设备适配 | 第 4–6 月 | 校准驱动权重、公开硬件数据、跨时间/设备评估 | 外部数据结果可复现，或明确记录硬件数据阻塞 |
| Phase 4：qLDPC | 第 4–8 月，可并行 | qLDPC/BB、BP+OSD、Relay-BP、资源评估 | 四项工业指标完整，不只报告 LER |
| Phase 5：实时原型 | 第 6–9 月 | Streaming API、profile、优化 CPU/FPGA、回退 | 无持续 backlog，满足预定 p99 和吞吐预算 |
| Phase 6：证书与发布 | 第 9–12 月 | 可扩展参考/界限、技术报告、一键复现和外部文档 | 第三方能够从干净环境复现核心表格 |

## 8. 未来 90 天执行清单

### 第 1–2 周：冻结实验协议

- [ ] 确定 d=3/5/7 circuit-level memory experiments。
- [ ] 确定噪声网格、shots、训练种子和独立测试集。
- [ ] 确定 baseline matrix 和结果 schema。
- [ ] 确定 streaming、延迟和资源的测量方法。

### 第 3–6 周：完成可信基准

- [ ] 实现 circuit-level 数据生成及测试。
- [ ] 完成 noise-aware correlated MWPM。
- [ ] 接入 BP+OSD 和一个低延迟 baseline。
- [ ] 自动生成 LER、CI、距离扩展和资源报告。

### 第 7–9 周：完成鲁棒性评估

- [ ] 完成噪声失配矩阵。
- [ ] 完成校准漂移和相关噪声测试。
- [ ] 完成 streaming 吞吐、p99 延迟和 backlog 测量。
- [ ] 确认最强传统基线及其适用条件。

### 第 10–12 周：检验一个学习或混合方法

- [ ] 只选择一个轻量 recurrent Transformer、noise-aware 混合方法或局部神经模型。
- [ ] 在相同 circuit、shots 和噪声信息下与最强基线比较。
- [ ] 完成独立测试和消融实验。
- [ ] 按第 6 节条件决定继续、调整或停止。

### 90 天必须交付的报告

> 在 circuit-level 和噪声失配条件下，候选方法是否比最强传统 decoder 更准确、更快、更省资源或更省校准数据？结论的统计不确定性和适用范围是什么？

如果答案都是否定的，应如实报告并转向传统方法，而不是通过挑选实验单元制造正面结论。

## 9. 资源与算力策略

- 当前测试、Stim 采样、MWPM、查表和精确 ML 不以 GPU 为前提。
- Circuit-level baseline 和 streaming 原型优先使用多核 CPU，并记录线程数。
- 只有进入大规模神经训练、多随机种子和超参数实验后，GPU 才成为主要效率工具。
- Exact ML 的主要瓶颈是指数状态空间和内存，GPU 不能从根本上解决。
- 实时部署不能使用“训练有 GPU”替代“在线延迟满足预算”的证明。
- 所有性能报告注明硬件、线程、batch、精度、软件版本和是否包含数据传输。

建议团队能力覆盖量子纠错、decoder 算法、机器学习和系统/FPGA；个人推进时按上述顺序完成，不并行铺开所有方向。

## 10. 风险与应对

| 风险 | 应对 |
|---|---|
| 神经网络无法超过强基线 | 转向 noise-aware MWPM/BP、混合 decoder 和 benchmark/reference 平台 |
| 真实硬件数据不可得 | 提供标准适配器并使用公开数据；明确标注硬件验证未完成 |
| Exact ML 无法扩展 | 限定为离线参考，发展高成本近似和受限解析界限 |
| 实时延迟不达标 | 局部/循环模型、早退、混合路径、量化和硬件实现 |
| 同时做 surface/qLDPC 导致分散 | Surface code 先行，qLDPC 在共享接口稳定后扩展 |
| 实验结果不可复现 | 固定配置和 manifest，自动报告，多种子，CI 回归，禁止手填数字 |
| 对“最优”或“实时”过度宣传 | 所有结论携带 noise/code/circuit/统计/延迟适用范围 |

## 11. 非目标

在上述阶段门完成前，本项目不把以下事项作为优先目标：

- 大型全序列 Transformer 或多个相似神经架构的竞赛；
- 更多 code-capacity MLP 参数扫描；
- 只针对普通 MWPM 的“SOTA”声明；
- 没有延迟和吞吐数据的“实时”声明；
- 没有硬件数据的“适用于真实设备”声明；
- 将小代码 exact ML 外推为大代码全局最优证明；
- 为追求架构新颖而牺牲可复现性和基线公平性。

## 12. 最终完成定义

项目具备可信工业价值时，至少应交付：

- [ ] 可复现的 circuit-level surface-code benchmark，并有一种 qLDPC 扩展；
- [ ] 包含 noise-aware MWPM、BP+OSD、低延迟方法及适用 qLDPC 强基线的对比；
- [ ] 在独立测试集上，相对最强基线至少有一项可复现的准确率、速度、资源或校准数据效率优势；
- [ ] 噪声失配、跨时间或跨设备的鲁棒性报告；
- [ ] 无持续 backlog 的 streaming 原型及 p99 延迟、吞吐和资源报告；
- [ ] 小规模精确最优参考和大规模清楚标注的近似参考或界限；
- [ ] 从干净环境一键复现核心结果的配置、代码、模型、数据 manifest 和文档；
- [ ] 包含负面结果和明确局限性的技术报告。

最终价值不在于“使用了 AI”，而在于：**在接近真实硬件的条件下，可复现地解决现有 decoder 的明确工程痛点，并准确说明收益的代价和边界。**

## 13. 公开方向参考

- Google Quantum AI, [Making quantum error correction work](https://research.google/blog/making-quantum-error-correction-work/)：surface-code 距离扩展、below-threshold 错误抑制和实时 decoder 延迟问题。
- Google Quantum AI and Collaborators, [Quantum error correction below the surface code threshold](https://www.nature.com/articles/s41586-024-08449-y), Nature 638, 920–926 (2025)。
- IBM Quantum, [Introducing Relay-BP](https://www.ibm.com/quantum/blog/relay-bp-error-correction-decoder)：decoder 的准确性、速度、紧凑性、灵活性和 FPGA 实现方向。
- IBM Quantum, [IBM lays out clear path to fault-tolerant quantum computing](https://www.ibm.com/quantum/blog/large-scale-ftqc)：bivariate bicycle qLDPC、模块化容错架构和实时 decoder 路线。

项目内部相关文档：

- [研究问题与里程碑](../README.md)
- [现有实验与负面结果](experiment_log.md)
- [证书定义与适用范围](certificates.md)
- [噪声结构描述符](noise_descriptors.md)
