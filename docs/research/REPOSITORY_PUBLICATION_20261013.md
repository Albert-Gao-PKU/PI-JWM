# 私有仓库发布完成

日期：2026-10-13。

后续同日更新：用户明确要求改为公开，已完成GitHub可见性切换。未登录GitHub REST API核验full_name=Albert-Gao-PKU/PI-JWM、private=false、visibility=public。以下私有发布描述是创建当时的历史事实；当前以公开状态为准。没有新增数据、权重、模型修改或训练。

## 创建和推送

- 用户指定名称PI-JWM、可见性私有。
- 仓库：https://github.com/Albert-Gao-PKU/PI-JWM 。
- 网页实际确认所有者Albert-Gao-PKU和Private标记。
- 首次提交：61f3c858307d9e5c22d10b54c56fee3f8ca7e69e，feat: initialize independent PI-JWM research repository。
- git push --set-upstream origin main成功，git ls-remote --heads origin main核验同一SHA。
- 首次提交后新工作区干净、main跟踪origin/main。本收口文件会作为后续文档提交，不修改任何模型代码。

## 工作区

新目录/home/gao/projects/PI-JWM。原/home/gao/projects/pi-jwm-upstream-20260929的origin仍是同学仓库，原暂存/未提交状态保持不变。没有把原Git历史复制进新仓库。

后续个人开发建议打开新目录，避免在原工作区修改后误认为已同步到本人GitHub。真实数据和训练权重仍须使用外部存储，不在本Git快照中。

## 内容和验证

从原工作树复制1101个经选择源文件、约28.40MB，逐文件SHA核验通过；过滤314个原清单中的载荷/本地状态文件。新增4份科学说明/参数记录及7份小型训练证据，原README另行存档。

发布预检时1114个暂存文件、805个源码/脚本/测试文件与原工作树字节一致、774个Python源码编译通过；元数据和收口文档后续增加文件数。新编码器CPU导入与37080/47368参数核验通过，命令行--help正常。

已识别的GitHub token/AWS key/private-key格式扫描无命中；未上传数据ZIP/NPZ、权重、虚拟环境、云登录工具或账号状态。该格式扫描不是对所有可能秘密的绝对保证。

上游历史文档/LaTeX模板有尾部空白和EOF空行，整体首次快照diff-check存在继承警告；不改其原字节。新增README/.gitignore/CURRENT_RESEARCH/数学说明的范围检查通过。历史夹具不在仓库内，未宣称全仓测试通过。

## 研究范围不变

没有新训练、GPU、locked_test、baseline/消融或Planner实验；没有修改模型、数据、归一化或训练协议。full阶段成绩仍以已保存原始证据为准，发布成功不等于方法优势已经验证。
