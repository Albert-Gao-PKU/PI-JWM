# 三项编码器改进：实现与训练说明

## 这版究竟改了什么

这是同学 PI-JWM 的**独立研究版本**，不是重新写一个替代世界模型。
项目仍在 WSL：`/home/gao/projects/pi-jwm-upstream-20260929`。
分支为 `experiment/geo-temporal-count-20261002`，基于已有源码 `f95bf2e`。
此次 GitHub fetch 遇到 TLS 中断，所以不声称已同步到今天的远端最新提交。
没有推送到同学的 GitHub，也没有改原始数据包或旧 checkpoint。

| 修改 | 实际做法 | 保持不变 |
|---|---|---|
| P2C 几何分支 | 当前通信端点的 Δx/Δy/Δz、距离、标量速度差、标量加速度差，附六维 mask，经 MLP 成为 16 维向量，与原 P2C 上下文拼接 | 无线关系才启用；不要求存在对应物理边；不增加第三张图 |
| CSI 历史编码 | 复用原 CSI 特征 MLP，增加独立 GRU，再融合当前编码与历史隐藏态 | 每 RB 特征顺序、历史长度 H=2、最终通信表示维度 16、有线当前编码 |
| 数量感知聚合 | 各信息关系家族和计算方向分别聚合 `[masked_mean, log(1+有效边数)]`，经 MLP 回到 16 维 | 物理图聚合、DAG 单向、P2A 对齐聚合及残差 LayerNorm |

后续的 Structured RSSM、原预测头、显式规则递推、动作、Loss/KL、Motion/CSI 指标和训练/验证划分都沿用原实现。
不增加任务进度预测头，不修改 Flow 的层间行为，不加入 HGT/完整 PNA。
六维几何特征使用原来的固定 TRAIN 物理关系归一化尺度，不利用验证集拟合。
速度差是标量差，**不是径向相对速度**。count 是关系条数，**不是唯一任务数、资源负载或队列长度**。

## 最重要的数据接口修正

真实正式数据中，通信行是各时刻分别排列的。前一帧第 10 行不一定是当前第 10 行的同一条链路。
因此新模块先按窗口内的 `(源实体, 目标实体, 关系类型)` 对齐到当前链路，再编码历史。
缺失时刻不更新该链路隐藏态；重复身份拒绝；RB 非标准顺序拒绝。
这个对齐只改变读取方式，原始压缩包、npz、训练标签均未修改。

## 本次实际验证结果

最终 `code/artifacts/research/geo_temporal_count_20261002/final_readiness/readiness.json` 的 `passed=true`。

- 60 个正式分片、5520 个窗口的身份/CSI 历史对齐审核通过；最终复用了同编码器和同数据哈希的审核结果，同时重新校验所有包文件哈希。
- 5 步有界短训覆盖 H1/H2/H4 与两个训练阶段；损失、梯度均有限，参数确实更新。
- 两个不同 TRAIN 轨迹的 H4 批次通过；所有新增分支收到非零、有限梯度。
- batch=8、H4、FP32 的优化器步骤通过，实测约69.60秒；这是本机耗时，不是云端速度承诺。
- 固定两样本 H1 的 4 步学习检查，预测损失 `0.844377 → 0.843350`，仅是小幅下降，不能称为充分过拟合或正式模型改进。
- 两个验证窗口的 prior-only H1–H4 接口通过，无参数更新；没有执行全1104窗口验证。
- 新模型实例 checkpoint 重载后的参数完全一致，恢复的优化器能继续更新编码器。
- 本机 RTX3060 Laptop 上 torch 峰值 allocated 约150.92MiB、reserved约164MiB（不包括驱动/其他进程显存；不是全数据最大批次保证）。
- focused regression 70/70 通过；没有宣称全仓回归通过，缺失历史开发夹具的问题保留在记录中。
- 最终 verification 从完整 Trainer 工厂重建并重载 checkpoint，在真实 TRAIN 当前状态上改变两架 UAV 的控制方向，位置与循环隐藏态均发生变化，输出有限且不使用未来标签；只是动作敏感性检查，不是 Planner 性能实验。

知识索引生成器另因历史 P4 实验的 `config.json` 缺失无法重建，未伪造历史文件或改写旧索引。新的研究文档和 AI_CONTEXT 路由已手动更新；该旧资料缺口不影响本次正式分片训练接口。

失败的初版通信行对齐检查和中间恢复夹具检查也保留在 research 目录中；最终结论以 `final_readiness` 为准，不能把失败收据当作可训练证据。

## 六种可直接选用的实验

| `--variant` | 几何 P2C | CSI 历史 | 数量聚合 |
|---|---|---|---|
| `baseline` | 关闭 | 关闭 | 关闭 |
| `geometry` | 开启 | 关闭 | 关闭 |
| `history` | 关闭 | 开启 | 关闭 |
| `count` | 关闭 | 关闭 | 开启 |
| `geometry_history` | 开启 | 开启 | 关闭 |
| `full` | 开启 | 开启 | 开启 |

三项全关时，参数初始化和数值输出与原编码器逐位一致；研究版会额外标记配置/输出合同，不能假称满足原来的“无 CSI 历史 GRU”验收条款。
完整改进编码器有 47,368 个参数，连同 RSSM/目标编码器/后验模型共 113,430 个训练参数。
不同参数量也应在正式比较中报告，不能仅凭指标变化断言是图结构带来的收益。

## 如何重新做准备验证

在 WSL 终端运行（下面是检查命令，不是正式训练）：

```bash
cd /home/gao/projects/pi-jwm-upstream-20260929
PYTHONPATH=code/src:code/scripts .venv/bin/python \
  code/scripts/run_research_geo_temporal_count_v1.py \
  --mode readiness --variant full --device cuda \
  --output code/artifacts/research/my_new_readiness
```

输出目录必须是新的空目录，防止覆盖已有实验。
结果在 `readiness.json`；只有 `passed=true` 才表示该次准备检查通过。
检查含 60 分片/5520 窗口的身份与包哈希、H1/H2/H4 短程训练、多轨迹 H4、固定训练小批次可学习性、batch=8 的 H4、验证集接口、新实例 checkpoint 加载和继续更新。
默认是有界 smoke，不会启动 5520 步正式训练。
固定小批次下降不代表泛化提升；这里只抽取两个验证窗口检查接口，**不是 1104 窗口正式验证**。

## 如何开始正式研究训练

这一步尚未自动执行。建议在云 GPU 上做，上传代码和既有 Training Bundle，保留其相对目录结构。
先确保 Python 3.10–3.13、numpy 和可用的 CUDA PyTorch；离线训练不需要重新启动 SUMO/AirFogSim。
运行 `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'` 检查 GPU。
不要上传本地 `.venv`，在服务器使用匹配的环境。

```bash
cd /path/to/pi-jwm-project
PYTHONPATH=code/src:code/scripts python \
  code/scripts/run_research_geo_temporal_count_v1.py \
  --mode train --variant full --device cuda --batch-size 8 \
  --output code/artifacts/research/full_seed5601
```

保留基线的数值协议：seed=5601、AdamW lr=3e-4、FP32、batch=8、TRAIN=4416、VAL=1104、最多5520步。
Stage1=552步；H1/H2/H4从0/1104/2208步开始；KL warmup=1104；每552步存 latest、每1104步做全量 prior-only 验证；按原 `L_Val` 选择 best，patience=3。
云端路径不同可用 `--manifest /absolute/path/formal_dataset_manifest.json` 指定入口；包路径由原 manifest 解析。
这不是读取同学已训练权重继续训练：不同结构应从同样的初始化协议重新训练，准备检查 checkpoint 也不能充当正式权重。

断点恢复时，其他参数和输出目录必须与第一次完全相同，再加：

```bash
--resume code/artifacts/research/full_seed5601/latest.pt
```

先完成 `full` 的正式训练和验证；若表现有变化，再用同数据、同预算分别跑 `geometry`、`history`、`count` 等消融。
同学旧成绩只有在源码、数据 manifest、归一化、随机种子、训练配置和评估协议一致时才可直接比较。
不同快照下的已训练成绩不可直接当作本实验的公平对照。

## 文件入口

- 新编码器：`code/src/pi_jwm/research_geo_temporal_count_v1.py`
- 新训练/检查入口：`code/scripts/run_research_geo_temporal_count_v1.py`
- 新测试：`code/tests/test_research_geo_temporal_count_v1.py`
- 原编码器只新增三个受保护的扩展钩子，默认路径不变；原 Trainer 只新增编码器类的工厂钩子。
- 原 Step5.2 历史开发夹具 `unified_flow_tensor.npz` 在本机缺失，相关旧测试不能启动；这是已记录的历史资料缺口。正式分片接口的实际验证单独提供证据。

本说明中的“可开始训练”仅指工程准备，不代表已验证创新性、泛化优势或下游规划性能。
