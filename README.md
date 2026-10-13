# PI-JWM

Physical–Information Joint World Model（物理—信息联合世界模型）的独立研究仓库。

仓库所有者：Albert-Gao-PKU。仓库地址：https://github.com/Albert-Gao-PKU/PI-JWM 。可见性：私有。

本版保留既有双图、Structured RSSM、动作、规则递推、预测头及训练协议，增加三项可独立关闭的编码器扩展：

- P2C显式几何分支：通信端点位置差、距离、标量速度差和加速度差。
- CSI历史编码：按源实体、目标实体及关系类型对齐链路后，用GRU编码两个历史时刻。
- 数量感知聚合：各信息关系家族/方向融合消息均值与log(1+有效关系条数)。

这些是待通过公平比较验证的研究扩展，不是已经证明的性能提升。Task/DAG仍属于信息图，没有新增第三个独立图编码器。

## 研究入口

- [当前研究版本与来源](docs/research/CURRENT_RESEARCH.md)
- [数学定义、逐层公式和全部参数对照](docs/research/双图表征模型数学说明与Baseline对比-20261012.md)
- [阶段训练结果分析](docs/research/模型训练结果分析-20261005.md)
- [参数核验JSON](docs/research/双图表征参数核验-20261012.json)
- [原始小型训练证据](code/artifacts/manifests/research_training_20261003)
- [原仓库README存档](docs/upstream/README.original.md)

## 来源与独立仓库边界

源自yys806的[network-state-world-model](https://github.com/yys806/network-state-world-model)，baseline提交f95bf2e5b9430666f325af3e6d6846d68b97e6e8。保留原作者、源码注释、项目记录和已有许可信息，不重新授予上游代码许可。

本仓库以2026-10-13的当前研究源码快照建立独立Git历史，包含原工作树中尚未提交的研究修改。没有修改同学的仓库或向其origin推送；原Git历史留在原工作区，没有导入本仓库。

历史记录、AI_CONTEXT和AGENTS可能保留上游路径、GitHub地址和旧阶段状态；它们不是本仓库新实验成绩。当前研究范围以CURRENT_RESEARCH及源码/配置/原始证据为准。

## 环境

正式云端训练使用Python3.11.17、NumPy1.26.4、PyTorch2.7.1+cu118、FP32。项目要求Python >=3.10、<3.14；复现应核对实际安装版本，不能把默认依赖安装结果视为原实验环境。

源码位于code/src，运行脚本位于code/scripts。安装及只读导入：

~~~bash
cd /home/gao/projects/PI-JWM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=code/src:code/scripts python -c "from pi_jwm.research_geo_temporal_count_v1 import GeoTemporalCountEncoder; print('encoder import OK')"
~~~

## 数据和权重不在Git仓库内

未上传数据ZIP/NPZ、真实轨迹、模型权重、虚拟环境、上传残片、账号密码或云端登录工具。大型历史协议样本也没有复制。

完整正式数据需另行取得，保持Training Bundle结构，通过其formal_dataset_manifest.json指定入口。仓库中的小型manifest/结果JSON是来源证据，不是完整可训练数据。部分历史测试依赖未包含的本地协议夹具，不能宣称新clone能不带数据运行全部历史测试。

## 检查入口

取得数据并核对环境后，可显式执行检查：

~~~bash
PYTHONPATH=code/src:code/scripts python code/scripts/run_research_geo_temporal_count_v1.py \
  --mode readiness --variant full --device cpu \
  --manifest /absolute/path/formal_dataset_manifest.json \
  --output code/artifacts/research/new_readiness
~~~

输出必须为新空目录。检查包含有界优化/验证，不是正式5520步训练；此次仓库迁移没有执行这个命令。可选版本：baseline、geometry、history、count、geometry_history、full。

## 已保存的阶段训练

2026-10-03至10-05的full训练到3948/5520步，因48小时任务保护窗口停止。3312步最低已验证L_Val为0.05927223645，尚无本次同条件baseline/消融成绩。

恢复检查点和最佳验证模型用途不同。续训需恢复模型、优化器、随机数和采样状态，不能直接重跑旧云端从头训练入口或覆盖原结果。

本次上传没有启动训练、访问locked_test、修改云实例计费或关机状态。
