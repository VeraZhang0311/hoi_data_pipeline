# G1 HOI offline data pipeline

本仓库只生成 **reference 的几何接触意图**。不训练 policy，不修改 SuperMimic 的观测、奖励、真实力接触标签或已有数据。

## 数据与几何

- ULTRA：冻结 train2651 与 canonical HOI80，源时间按既有 60 Hz 协议解释；不宣称重建了采集时钟。
- HOI-Retarget / OMOMO：30 Hz G1；按关节名读取，保留同源 HOI80 隔离标记。流式读取 Parquet，不解码 video。
- GRAIL sitting：发布 G1 轨迹；利用独立 `pose_aa` 验证根四元数顺序。物体 USD 相对刚体根提取，按照作者发布数据回放器应用 `objects.pkl` 的 motion-lib scale；非默认 scene_scale 需要独立证据。
- OmniContact：90 Hz G1；验证发布版本后重排关节。原 carry 箱体路径保留。新增显式 SHA 绑定的正式训练资产目录，可解锁 1,608 条 push / slide / soccer 片段；没有绑定的片段仍保留 review。
- HiPHI 本轮不纳入。

机器人绑定 v1.2 验收 bundle 中的 G1 USD、URDF 和依赖 SHA。使用已启用的碰撞表面，包括橡胶手、原五 hull 足部；禁止重新 cook 足部、启用旧脚球或以关节中心替代表面。导出时核对 USD 与 URDF 的身体坐标系。

39 个 canonical link 各对应一个 `body__object` 通道，外加 `object__ground`。没有碰撞表面的 link 保持 unknown、active mask=false。这里的 39 是 link 命名空间，不是 29 个控制关节。

## 标注合同

1. 保留完整原始片段，不按时长切段；源数据只读。
2. 真实时间插值到 50 Hz：位置/关节线性插值，四元数 SLERP；不拉伸片段。末尾不足 20 ms 的余量显式记录。
3. 同时计算原始时间网格和 50 Hz 网格的几何标签；短事件通过原始采样的时间支持区间保留。区间覆盖比例不是接触概率，也不是无法观测的亚帧真值。
4. 距离 ≤2 cm 为 proximity=1，≥4 cm 为 separated=0，中间、穿透、无几何为 unknown=-1。这是待校准的初始阈值。距离、有效性、相交标记均保存；宽相位下界明确标为非精确距离。
5. 全身距离不等于力接触。凹物体保持原表面，不取覆盖椅子空隙的整体 convex hull；只有闭合几何能进行实体包含检测。坐下/倚靠的奖励和通过标准应另行定义。
6. T+1 个状态与 T 个控制区间分别保存；不插值源二值 contact。源 contact 独立保留，NaN 和缺失不变成负标签。

场景桌面等支撑物保留为独立 partner 元数据；当前主标签只覆盖机器人—目标物体和物体—地面，不把桌面并入地面。

## 运行

Python ≥3.10，几何批处理以 Linux CPU 为目标。依赖见 `pyproject.toml`，ULTRA 另需兼容的 CPU/GPU PyTorch 来读取权重安全 tensor，不进行 GPU 计算。

```bash
python -m unittest discover -s tests -v
python -m hoi_pipeline.cli freeze --config configs/cluster_v1.json --output /new/frozen
python -m hoi_pipeline.cli run --frozen /new/frozen --output /new/derived --source ultra --shard 0 --shards 8
```

`freeze` 绑定代码、输入、资产、配置及 runtime 版本。未完成上传的数据只能使用带预期 SHA 的 inventory，并在作业前经过就绪检查。每个 shard 用独占 claim 防重复 producer；成功片段有 `SUCCESS.json`，失败逐条记录原因。重跑使用新 output 或显式审查旧 claim，不自动覆盖。

输出组织为 `derived/<recipe>/<source>/<clip-hash>/`。`labels.npz` 无 pickle，`metadata.json` 含数组引用与来源，物体几何按 hash 复用。`training_ready` 固定 false：离线接触标注成功不代表 G1 已可物理执行；后续需 privileged tracking rollout 筛选。

当前工程状态和集群回执保存在 ignored `receipts/`，不提交原 motion、资产、权重或结果。

## OmniContact 正式资产绑定

`python -m hoi_pipeline.omnicontact_catalog --asset-root /audited/assets --author-code-root /pinned/training --output /new/catalog.json`

目录构建重开正式 USD，并与独立 URDF 交叉验证，绑定审查文件、全部资产依赖及作者的文件名解析、物体生成、MotionCommand 源码 SHA。数据 revision 为 `068b83049df50653327b06ee41bbbaf1b9bd593e`，训练代码 revision 为 `19cde38364a5b48e47348b270c895eabe2ba3ad9`。每条 inventory 显式增加 `author_geometry_binding`，同时锁定 clip ID、原 NPZ SHA、资产名和目录 SHA；freeze 将依赖闭包纳入输入。

55 种箱体使用真实 USD Cube 尺寸。`soccer_20` 和 `soccer_22` 的正式资产相同：约 21.6 cm、382 顶点的偏心网格，中心相对刚体根约 `[-0.00906, -0.01449, -0.00310] m`。保留原顶点和三角表面，验证闭合、绕序及凸性；不按名称生成球、不重新居中。其 USD 使用合并碰撞子树和 `convexHull`，这里不宣称重建 PhysX cooked 顶点缩减或 contact/rest offsets。

作者 MotionCommand 直接使用记录的 object position / quaternion，spawner 的 scale 为 1。历史 scale、center offset、geom quaternion 不再次应用；源尺寸元数据若与正式资产不同，仍写入 review flags，不能据此自动修正轨迹。几何绑定成立不保证物理可执行、动作意图正确或数据 split 已确认。

### 2026-10-30 验收与补标

生产代码冻结在 `5b3da10`。57 个真实资产均通过 USD / URDF / SHA 交叉验证；集群 CPU suite 为 73 passed、1 optional skipped。验收作业 `1482189` 已 COMPLETED / 0:0，9 条完整 reference 标注、保存及回读全部通过，其中包含两种足球别名、推/滑箱体、尺寸冲突片段及原 carry 回归。

原批次已完成 623 条 carry，另有 1,608 条仅因几何绑定待审查。新批次逐条核对这 1,608 个 ID 和原 NPZ SHA：push 453、slide 190、足球 965，共约 7.75 小时；没有截断或改位姿。adapter 审查标记保留：82 条 grounding 尺寸冲突、1 条 object_size 冲突、5 条跳变候选、1 条分组未确认（后者与 object_size 冲突重叠，共 88 个 clip）。标注阶段的速度/几何诊断还可能新增标记。

- 补标作业：`1482213`，32 shards、最多 8 个 CPU 作业并行；提交成功不代表全部完成。
- 冻结 recipe：`378c47897b83440505e3915094eb32923f9968b4a37f19bb30a057b1e68e5eb6`。
- 集群运行目录：`/mnt/projects/ll/tianzizh/hoi_data_pipeline_20261029_01/runs/omni_author_geometry_20261030_01`；含 `batch_frozen`、`VALIDATION_PASSED.json`、`BINDING_AUDIT.json`、`batch_submission.json` 和日志。
- 独立输出：`/mnt/projects/ll/tianzizh/hoi_corpus/derived/g1_expected_contact_omni_author_geometry_20261030_01`。
- 本地 ignored 回执：`receipts/omni_author_stage/`；原始作者资产及独立审查在 `receipts/omni_author_assets/`。实际进度以 Slurm、各 shard summary 和 `SUCCESS.json` 为准。
