# 云端代码上传准备（2026-10-02）

## 当前结论与边界

已修复代码包解压到普通目录、没有 `.git` 时无法记录源码来源的启动问题。新的云端代码包是独立文件 `research_code_bundle_cloud_20261002.tar.gz`，没有覆盖旧代码包、旧 release manifest、readiness 或 checkpoint。

本记录只证明本地部署准备。云端环境尚未检查，代码/数据尚未上传，正式训练未启动，也未获本轮授权；服务器租赁与上传状态应以实际操作记录为准。本次未访问 GPU 或 `locked_test`，不宣称性能改进。

## 来源记录的修复

`run_research_geo_temporal_count_v1.py` 新增 `deployment_git_commit()`：有真实 Git 仓库时记录其 HEAD；Git 命令不存在或解压目录不是仓库时返回 `None`，不会把已修改的研究代码假称为某次提交。

`baseline_source_commit` 独立记录同学源码起点 `f95bf2e5b9430666f325af3e6d6846d68b97e6e8`；原逐文件 `research_source_sha256` 仍保留。模型、优化器、loss、训练日程、数据与评估协议均未改变。

本地 focused tests 6/6 通过：包含 Git 成功、非 Git 目录、没有 Git 可执行文件三个模拟案例，以及原训练配置、早停、恢复步数检查。这不是云端 readiness 或新的 GPU 训练凭证。

## 上传文件

- 新代码包：`code/artifacts/research/geo_temporal_count_20261002/research_code_bundle_cloud_20261002.tar.gz`。
- 来源/包哈希清单：同目录 `cloud_transfer_manifest.json`；上传后校验 SHA256，应与清单完全一致。
- 数据：另传 `PIJWM_FormalDataset_v1_H2L4_training_part1.zip` 和 `PIJWM_FormalDataset_v1_H2L4_training_part2.zip`。它们是互补的两个 Training ZIP，不是两个独立实验集；必须解压到同一个数据目录。

代码包含 `code/src`、`code/scripts`、`code/tests`、核心 requirements、pyproject、README、AGENTS、AI_CONTEXT 当前状态/变更和两份研究说明；不含原始/正式数据、虚拟环境、Git 历史、checkpoint、浏览器凭据或私密笔记。RAW ZIP 暂时不用上传用于离线模型训练。

## 云端部署顺序

1. 用户确认实例及费用后再租赁。选用已提供 CUDA PyTorch、Python 3.10–3.13、NumPy，且与主机驱动兼容的镜像；镜像版本仍待现场确认，不固定为某个 PyTorch/CUDA 组合。显卡名称或网站“最高 CUDA 版本”不能替代实际环境检查。
2. 在云数据盘的新项目目录解压代码包。两个 Training ZIP 都解压到下面同一个目录，保留 `formal_dataset_manifest.json`、`packages/` 和相关审计文件的相对结构：
   `code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/`。
3. 不复制本地 `.venv`。用云镜像的 Python 验证 NumPy/PyTorch 导入及 `torch.cuda.is_available()`，查看实际 GPU、驱动、PyTorch 版本。不要为满足某个版本号盲目升级驱动或 PyTorch。
4. 在云端使用新的空输出目录重新做 readiness，不使用旧 smoke checkpoint 来启动正式训练。以下只进行有界准备检查，不是 5520 步正式训练：

```bash
cd /path/to/new/project
PYTHONPATH=code/src:code/scripts python \
  code/scripts/run_research_geo_temporal_count_v1.py \
  --mode readiness --variant full --device cuda \
  --output code/artifacts/research/cloud_readiness_new
```

只有新的云端 `readiness.json` 显示 `passed=true`，才能说该云环境训练准备通过。它会重新检查数据和 H1/H2/H4、batch8、保存重载等接口，不等于正式泛化实验。

5. 云端准备通过后，请用户另行确认正式训练。不同实验版本使用独立输出目录，训练从同等初始化开始；不要上传或复用旧 readiness 的训练权重作为正式结果。

模型设计和原验证范围参见 `docs/research/geo_temporal_count_20261002.md`。旧说明的本机 readiness 仍属于旧源码记录，不自动升级为本次云端执行证据。
