# 当前独立研究版本

更新：2026-10-13。明确新仓库所有者、代码来源和当前证据边界；不是新的方法决定或训练授权。

## 仓库与来源

- GitHub：Albert-Gao-PKU/PI-JWM，公开；2026-10-13初建私有，随后按用户要求改为公开。
- 独立WSL工作区：/home/gao/projects/PI-JWM。
- 原上游工作区：/home/gao/projects/pi-jwm-upstream-20260929，其origin和暂存/未提交内容保留不变。
- baseline：yys806/network-state-world-model，提交f95bf2e5b9430666f325af3e6d6846d68b97e6e8。
- 本仓库是当前工作树的过滤快照，不包含原Git历史、完整实验数据或权重。

## 方法

GeoTemporalCountEncoder继承PIJointGraphEncoder。完整版本开启enable_geometry_branch、enable_csi_history、enable_count_aggregation。

Physical–Information定义不变；Task/DAG在异构Information图中。Encoder输出每对象/关系16维表示，Structured RSSM、动作、规则、头、target teacher及loss沿用baseline。

几何速度差是标量差而非径向速度；count是有效关系条数而非唯一任务数。通信历史按窗口内稳定实体端点+类型匹配；H=2、RB=50。

继承字段csi_history_encoder=false、aggregator=masked_mean不能单独解释研究版行为，新增功能由三个独立开关控制。

## 配置与来源核验

编码器37080→47368参数，总训练模型103142→113430参数。与原部署绑定的SHA：

- code/src/pi_jwm/research_geo_temporal_count_v1.py：cebdecf4e50834f7dc074265f9f31958e3476006d228ca9c8e8a552791794177。
- code/src/pi_jwm/step4_3b_dual_graph_encoder_v1.py：80926907e5fa26b1bca76f4707dc6d1075525d107895db38d5516d2117ff7a4a。

原正式配置：seed5601、AdamW lr3e-4、weight_decay0、batch8、FP32、5520步上限、H1/H2/H4课程。

[完整原始provenance](../../code/artifacts/manifests/research_training_20261003/formal_provenance.json)。

## 结果边界

已有full记录：3948步保护停止，3312步最佳完整验证L_Val=0.05927223645。未因迁移重新训练，未进行本次同条件baseline/消融、locked_test或Planner性能实验。

仓库迁移只验证代码及小型证据保存，不证明创新性、泛化优势或云镜像恢复已验收。

## 阅读顺序

README → CURRENT_RESEARCH → 数学说明/原始provenance → 实际源码 → 对应实验结果。

保留上游AI_CONTEXT、计划和AGENTS作为来源/背景；不得依据旧文档自动训练、推送同学仓库或把其历史实验写成本版成绩。

后续修改科研假设、结构、loss、数据、对比协议或公开结论，需要先与研究者确认。
