# G1 HOI offline data pipeline

本仓库只生成 **reference 的几何接触意图**。不训练 policy，不修改 SuperMimic 的观测、奖励、真实力接触标签或已有数据。

## 数据与几何

- ULTRA：冻结 train2651 与 canonical HOI80，源时间按既有 60 Hz 协议解释；不宣称重建了采集时钟。
- HOI-Retarget / OMOMO：30 Hz G1；按关节名读取，保留同源 HOI80 隔离标记。流式读取 Parquet，不解码 video。
- GRAIL sitting：发布 G1 轨迹；利用独立 `pose_aa` 验证根四元数顺序。物体 USD 相对刚体根提取，按照作者发布数据回放器应用 `objects.pkl` 的 motion-lib scale；非默认 scene_scale 需要独立证据。
- OmniContact：90 Hz G1；验证发布版本后重排关节。已明确绑定的 carry 箱体按作者最终尺寸和重居中记录构造；其余有歧义资产保留 review，不猜变换。
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
