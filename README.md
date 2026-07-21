# SSTG-Nav ObjectNav benchmark

`sstg-nav-bench` 是 **SSTG-Nav 算法的公开 benchmark 实现**，用于在 Habitat/HM3D 中复现和评测 SSTG-Nav 的预先建图、语义定位、候选检索及导航指标。它不是 SSTG-Nav 的完整实机部署仓库。

SSTG-Nav 的实机部署、机器人系统集成及相关工程代码属于与华为合作项目的保密内容，受合作保密约束，不在本仓库的开源范围内，也不会对外开源。本仓库仅提供可公开复现的 benchmark、评测工具、实验配置和结果校验代码。

本 benchmark 与交付仓库 `../sstg-nav` 隔离。它复现 SSTG-Nav 的预先建图范式：先在与任务无关的拓扑点采集带位姿 RGB-D，多模态模型离线检测目标，深度将图像框转换为三维物体位置与可导航停止点；查询时只需检索候选、规划路径并验证是否到达。

交付仓库未被改动。主实验使用 PeterAI 作为 API 传输端点、模型身份严格记为 `GPT-5.4`；`MiMo-v2.5` 仅作为 30-episode 模型消融。

实现入口、地图契约、分层 Top-K 排序和协议边界见 [`docs/core-pipeline.md`](docs/core-pipeline.md)。首次使用建议先阅读其中的“环境边界”和“评测协议”两节。

## 主结果

完整实验使用 HM3D ObjectNav-v2 validation 的 1,000 个 episode、36 个 scene。独立拓扑由 12,000 个 navmesh 随机候选做最远点采样，覆盖半径 0.8 m，共 6,642 个节点、21,845 条边；建图过程不读取 ObjectNav goal。每个节点采集四张 640×640、HFOV/VFOV 120° 的 RGB-D，共 26,568 个视角。

| 方法 | 候选/反馈 | SR | SPL | DTG |
|---|---|---:|---:|---:|
| 独立几何覆盖上限 | evaluator 仅用于事后赋义 | **0.994** | **0.992** | 0.622 |
| GPT-5.4 camera node | 将识别结果贴在相机点 | 0.835 | 0.560 | 1.039 |
| GPT-5.4 raw RGB-D，single | 框中心深度反投影 + 可导航 standoff | 0.920 | 0.603 | 0.879 |
| GPT-5.4 raw RGB-D，Top-3 | raw 候选，2 m 多样性 | **0.975** | **0.616** | -- |
| GPT-5.4 3D soft fusion，single | 多视角、三维与可达性融合 | 0.926 | 0.586 | **0.747** |
| GPT-5.4 fusion-aware Top-3 | fused 首选 + 两个 2 m 分离 residual standoff | **0.975** | 0.601 | -- |

关键结论：

- 独立 0.8 m 拓扑本身能覆盖 99.4% 的成功区域，因此 `Target-view coverage` 的高成功率主要来自它包含真实成功视点；它只能是 privileged coverage control，不是可部署主结果。
- 在完全相同的 6,642 次 GPT-5.4 节点响应下，depth grounding 将 SR 从 0.835 提升到 0.920。paired gain/loss 为 93/8，exact McNemar `p=1.74e-19`；SPL 增加 0.042，95% CI `[0.028, 0.057]`。
- 3D fusion 将 20,107 个 raw 候选压缩到 1,329 个并降低 DTG，但只带来 +0.006 SR；SPL 下降 0.017。因此它主要是地图压缩与假设合并，不应宣称是主要性能来源。
- 原实现把 fusion 后的每个证据簇压缩成唯一代表点，又用另一套 raw 候选做 verified Top-3，因此 Table 2 中的 0.912 并不是“fused 失败后再试两个 fused 备选”。现已修复为分层候选：Top-1 使用稳定 fused representative；失败后从建图阶段保留的 residual RGB-D standoff 中选两个高置信、至少相距 2 m 的备选。Success@1/2/3 为 0.928/0.965/0.975，SPL@3 为 0.601。
- 公平的 2 m GPT-5.4 恢复对照也已补齐：raw Top-3 为 S@1/2/3 0.920/0.963/0.975，collapsed fused representatives 为 0.926/0.952/0.964，fusion-aware 为 0.928/0.965/0.975。最终策略在首次和第二次访问最强，并在三次预算内保持最高成功率；raw 保留 0.616 SPL 的路径效率优势。Qwen 仅作为辅助模型消融。
- 每个候选的四张 120° 到达 RGB-D、参考图和姿态全部保存在 `gpt54_arrival_capture_fusion_aware_top3/`。新鲜 RGB-D/VLM STOP 分类仍保留为独立校准实验和可视化来源，不再与 candidate-list Success@3 混成同一行。
- 更宽视场不是无条件增益。GPT-5.4 minival 在 120° 下优于 90°，尤其补回低处 toilet；MiMo-v2.5 在 120° 下反而退化，说明 FoV、畸变和 VLM 定位能力必须联合标定。

### 全量统计

| 对比（同一批 GPT-5.4 响应） | success gain/loss | ΔSR | ΔSPL [95% CI] |
|---|---:|---:|---:|
| camera → raw RGB-D | 93 / 8 | +0.085 | +0.042 `[0.028, 0.057]` |
| raw RGB-D → fusion | 28 / 22 | +0.006 | −0.017 `[−0.032, −0.002]` |
| camera → fusion | 113 / 22 | +0.091 | +0.026 `[0.005, 0.047]` |

| 类别（n） | Camera SR/SPL | Raw SR/SPL | Fused SR/SPL |
|---|---:|---:|---:|
| chair (195) | 0.841/0.415 | 0.974/0.488 | 0.995/0.498 |
| bed (165) | 0.891/0.673 | 0.988/0.716 | 0.976/0.631 |
| plant (152) | 0.862/0.473 | 0.868/0.481 | 0.941/0.532 |
| toilet (166) | 0.934/0.771 | 0.940/0.746 | 0.940/0.673 |
| TV/monitor (135) | 0.644/0.455 | 0.800/0.578 | 0.778/0.571 |
| sofa (187) | 0.807/0.572 | 0.914/0.611 | 0.893/0.613 |

所有置信区间、per-episode 行和 paired changes 均保存在 `outputs/hm3d_val_uniform/`，不是从论文表格反推。

## 方法实现

1. `uniform_map.py` 从 navmesh 随机池做 goal-independent farthest-point sampling，并建立可达拓扑边。
2. `rgbd_capture.py` 在每个节点保存四个 cardinal RGB-D、相机内外参和节点位姿。
3. `peterai_rgbd_semantics.py` 每个节点一次请求四张图，输出 `view_index/category/bbox_norm/confidence`；原始响应逐节点原子缓存，可断点续跑。
4. `rgbd_fusion.py` 在框内取有效深度中位数，反投影物体表面点，沿观察方向构造 0.8 m standoff，snap 到 navmesh；随后按三维距离、楼层高度、独立来源和可达性做 soft fusion，同时输出 `raw_maps/`、`clustered_maps/` 和 `multi_standoff_maps/` 三种候选表示。
5. `experiments.py` 从 episode 起点对同类候选做 geodesic 路径规划，以 Habitat 的 1 m success region 计算 SR、SPL、DTG。
6. `topk.py` 支持 fused primary、可选 secondary 与 fallback 三层候选，并在全局多样性约束下排序（主实验为 2 m）；它在每次访问后读取官方成功区域，因此只作为 candidate-list oracle-feedback ceiling。
7. `arrival_capture.py` 在不读取 goal 的前提下按同一查询规则选择 primary + fallback Top-3，并在每个唯一候选处采四张目标定向的新鲜 RGB-D，同时保存带框建图参考图。当前 capture CLI 尚未接入 `topk.py` 的可选 secondary tier。
8. `arrival_vlm.py` 用 GPT-5.4 判断真实目标可见性、停止侧几何、到达视图框与置信度；对齐深度在框中央区域给出实测距离，严格策略要求两种几何判断同时通过。
9. `arrival_evaluate.py` 让 verifier 的 accept/reject 真正控制 STOP/继续，并在决定完成后才读取官方 goal 计算 SR/SPL、precision/recall 与错误类型。

success 判定是终点到任一官方同类 goal viewpoint 的 navmesh 距离不超过 1 m；`SPL=S·L/max(L,P)`。主 benchmark 使用连续 navmesh 路径，不模拟离散动作碰撞、控制噪声或定位漂移。

### 候选地图与恢复策略

同一次 fusion 会写出三套 scene-local 地图。当前全量缓存分别包含 20,107 个 raw 候选、1,329 个 fused representatives 和 18,786 个 retained multi-standoff 候选：

| 目录 | 一个节点表示什么 | 分数与用途 |
|---|---|---|
| `raw_maps/` | 一个有效框深度反投影得到的独立 standoff | 原始检测置信度；保留全部恢复假设 |
| `clustered_maps/` | 一个保留证据簇的最高置信 representative | 独立拓扑来源的 noisy-OR 分数；适合稳定 Top-1 和压缩地图 |
| `multi_standoff_maps/` | 保留证据簇内的每个可达 standoff | 共享 cluster score/support；用于研究同一融合目标的替代停止侧 |

论文主策略是两层的 `clustered primary (K=1) → raw fallback`。`topk.py` 还支持 `primary → secondary → fallback` 三层诊断；三层接口、精确排序键和示例命令见 [`docs/core-pipeline.md`](docs/core-pipeline.md#分层-top-k-候选)。所有地图中的 `id` 只在“单 scene、单目录”内唯一，跨层关联应优先使用实际位姿，以及存在时的 `fusion_cluster_id` 或 `source_candidate_id`，不能直接拼接本地 `id`。

## 环境与数据

已有环境：Python 3.9、Habitat-Sim 0.3.3、NVIDIA RTX 3080 Ti。推荐直接使用：

```bash
export PYTHONPATH=/home/daojie/SSTG_Nav/sstg-nav-bench
export HABITAT_PYTHON=/home/daojie/anaconda3/envs/habitat/bin/python
```

也可以从环境文件创建：

```bash
conda env create -f environment.yml
conda env create -f environment-vlm.yml
```

`environment.yml` 负责 Habitat/navmesh、RGB-D 和指标计算；`environment-vlm.yml` 负责本地 Qwen 推理。Hosted API 阶段只需要通用 Python 环境中的 `requests`。不要在 VLM 环境中强行安装 Habitat，也不要让 Habitat 环境升级到 NumPy 2。

原始下载位于 `/data/Database/Nav_scence_datasets`，当前 HM3D 与 ObjectNav-v2 已整理到 `data/hm3d/` 和 `data/datasets/objectnav_hm3d_v2/`。`.env` 只用于本地读取 API key，任何输出、日志、论文和清单都不包含 key。

## 一键复现全量实验

```bash
cd /home/daojie/SSTG_Nav/sstg-nav-bench
export PYTHONPATH=$PWD
./tools/run_full_rgbd_benchmark.sh
```

脚本按顺序建立独立拓扑、采集 RGB-D、调用 GPT-5.4、构造 camera/raw/fused maps、评测三种表示、运行 diverse Top-3、采集并验证到达 RGB-D、生成闭环可视化和计算几何上限。已成功的 VLM 响应不会重复计费。可通过 `HABITAT_PYTHON`、`GENERAL_PYTHON` 和 `WORKERS` 覆盖默认环境。

该脚本复现论文采用的两层 fusion-aware Top-3。若要运行代码新增的三层候选诊断，请使用 [`docs/core-pipeline.md`](docs/core-pipeline.md#三层诊断示例) 中的显式命令；三层 `topk.py` 结果不能直接当作自主 STOP 结果，因为它仍在每次候选后使用 evaluator success feedback。

只生成全量可视化：

```bash
$HABITAT_PYTHON -m sstg_bench.rgbd_visuals \
  --split val --scene-dir val \
  --semantics outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120 \
  --fusion outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals

$HABITAT_PYTHON -m sstg_bench.visualize_results \
  --split val --scene-dir val \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps \
  --episodes outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/episodes_vlm_all_confidence.csv \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals/navigation \
  --successes 4 --failures 6
```

MiMo-v2.5 minival 消融：

```bash
./run_rgbd_minival.sh mimo wide120
```

Qwen2.5-VL-3B 的同响应 camera/raw/fusion 补充实验：

```bash
./run_qwen_rgbd_minival.sh
```

该脚本复用 339 个独立拓扑节点的四视角 90° RGB-D。Qwen 首先进行带框定位；camera-node 基线丢弃框和深度，raw/fusion 则使用完全相同的检测，因此可以形成严格的辅助表示消融，而不会把旧 panorama 分类响应与新定位响应混作一个 triplet。

本次完整缓存为 339/339 节点、1,064 个定位检测和 959 个有效深度候选；camera/raw/fusion 分别得到 `0.400/0.233`、`0.333/0.188`、`0.500/0.218` SR/SPL。结果说明 Qwen 的单框深度投影并不稳定；raw→fusion 有 8 个 gain 和 3 个 loss，使成功数从 10 净增到 15，配对 SPL 95% 区间 `[-0.103,0.167]` 跨零。旧 `0.400/0.344` 是只输出类别的 panorama 响应，作为独立 coverage baseline 保留，不能拿来填 raw/fusion 两列。

## 结果与可视化入口

- `outputs/hm3d_val_uniform/VISUAL_INDEX.md`：最便于人工核验的总入口，直接链接总览图、四个成功与六类失败的第一/第三视角视频、逐 episode 记录和论文图。
- `outputs/paper_core/`：论文直接相关的核心结果/地图元数据、逐 episode CSV、SHA-256 清单和大型媒体索引；可独立复算所有主表数字。
- `outputs/sstg_nav_paper_core.tar.gz`：上述核心包的便携压缩版，适合 Supplementary 提交；原始图像、深度和视频不重复打包。
- `outputs/hm3d_val_uniform/oracle_geometry/`：独立几何上限、1,000 条 episode 记录和密度图。
- `outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120/`：6,642/6,642 完成的 GPT-5.4 缓存、模型身份审计和 semantic maps。
- `outputs/hm3d_val_uniform/gpt54_camera_node_analysis/`：camera-node 同响应基线。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/`：raw RGB-D 主结果。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/`：3D-fusion 主结果。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/{raw_maps,clustered_maps,multi_standoff_maps}/`：同一批 RGB-D 检测派生的三类候选地图。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/`：与最终策略匹配 2 m 多样性的 GPT-5.4 raw Top-3 核心对照。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/`：与最终策略匹配 2 m 多样性的 collapsed fused-representative Top-3 对照。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/`：最终 fusion-aware Success@1/2/3 与逐 episode 路径。
- `outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/`：最终 720 个唯一候选、2,880 张到达 RGB-D、参考图和姿态。
- `outputs/hm3d_val_uniform/gpt54_arrival_capture_top3/`：656 个唯一候选、2,624 张新鲜到达 RGB-D 与全部位姿/参考图。
- `outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict/`：656/656 GPT-5.4 原始响应、双重几何判定与 token/latency 报告。
- `outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/`：1,000 episode 自主 STOP/继续结果、逐次 attempt 与 confusion。
- `outputs/hm3d_val_uniform/gpt54_arrival_verified_visuals/`：656 个到达判定 overlay、接受/拒绝总览和真实恢复路线。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_3m/`：0.975 SR 的 Top-3 诊断。
- `outputs/hm3d_val_uniform/paired_*.json`：三组 paired tests。
- `outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals/`：全量检测框、RGB-depth 对照、36-scene semantic maps、第一视角与俯视视频。
- `outputs/hm3d_minival_uniform/`：GPT-5.4 与 MiMo-v2.5 的 90°/120° 控制实验及可视化。
- `outputs/hm3d_minival_uniform/qwen_rgbd_semantics_90/`、`qwen_rgbd_fusion_90/` 与 `qwen_rgbd_90_analysis_{camera,raw,fused}/`：补齐后的 Qwen 339/339 同响应表示消融。
- `../SSTGNavPaperAAAI/ARTIFACT_INDEX.md`：论文声明到证据路径的总索引。
- `../SSTGNavPaperAAAI/figures/generated/`：从真实 benchmark 与 ROS 2 记录生成的论文 PDF/PNG 图。

## 校验

```bash
python tools/validate_release.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. $HABITAT_PYTHON -m pytest -q
```

这里禁用环境外的 pytest 自动插件，避免系统 ROS 2 的
`launch_testing` 插件被注入 Habitat 的 Python 3.9 环境；项目测试本身不依赖该插件。

校验器重新从 per-episode 结果计算 SR/SPL，检查唯一 episode key、API 完整性、goal-independent capture 声明、RGB-D 数量、模型身份、Top-K 结果、可视化数量和文献键。最终报告写入 `outputs/release_audit.json`。

整理并独立验证论文核心数据包：

```bash
python tools/package_paper_artifacts.py
python outputs/paper_core/verify_paper_bundle.py outputs/paper_core
```

清理模式只会移除未被引用的旧 `ral_results_table.tex` 派生片段，不触碰 JSON/CSV 评测结果或任何图片、深度和视频；操作记录保存在 `outputs/output_cleanup_report.json`。

## 协议边界

unknown-scene online ObjectNav 在 episode 内探索并把探索成本计入 SPL；本方法在任务前扫描场景，并将一次 survey 成本摊销到重复请求。两种信息预算不同，不能直接把本项目的 0.926 SR 宣称为对在线方法的同协议 SOTA。Target-view map 又进一步读取了目标实例附近视点，只能用于隔离 coverage/semantics。论文主表已按 unknown / target-derived / goal-independent pre-map 分组。

当前剩余科学限制是：只评测一个 HM3D-v2 validation split、静态模拟环境和一个全量主 VLM；未评测地图老化、动态物体、定位漂移与真实机器人 SR/SPL。自主 Top-3 已不使用 evaluator arrival feedback，但它仍基于模拟器的新鲜 RGB-D，真实相机噪声与停止几何需要后续物理评测。真实机器人图只证明 ROS 2 数据流和运动接口已经实现，不作为量化 ObjectNav 结果。
