# HomographyFirst-RunwayDetector 实施方案

## Context

机载实时跑道视觉特征检测器，输出 4 角点 + 4 条边线（左/右/入口/出口）+ 1 条中线，运行在 ≤30ms/帧（Jetson Orin NX 级硬件）。**高精度 + 高可靠性 + 实时**三者并重。

可用先验远比上版方案丰富：
1. **图像先验**：上一帧的检测结果、基于 GPS/IMU + PnP 的 4 角点投影
2. **几何先验**（关键洞见）：
   - 跑道是平面矩形，**宽度已知**（首都机场 60m）
   - 4 角点共面 → 与图像平面之间存在唯一**单应矩阵 H**
   - 左/右边线与中线在 3D 中平行 → 图像中三条线共**长向消影点 VP_L**
   - 入口/出口边线在 3D 中平行 → 图像中两条线共**横向消影点 VP_S**
   - VP_L 方向由（跑道航向 − 飞机航向）决定，可从 IMU 直接给出强先验
   - 中线 = 左右边线的中位线（无需独立标注，可解析推导）

**核心设计思想**：检测器只学**一件事**——4 角点。所有其它几何要素（5 条线、2 个 VP、中线）通过 4 角点拟合的单应矩阵 H **解析推导**。这把 18 维输出（4×2 角点 + 5×2 线段 + 2×2 VP）压缩成 8 维输出（H 的 8 个 DOF），用几何硬约束代替学习负担，**结构性地**保证所有平行、消影、共面关系成立。

最接近的成熟范式是**体育场地标定（sports field registration）**：球场也是已知尺寸的平面矩形，从单视角推导 H，PnLCalib (CVPR 2024) / TVCalib 等是 SOTA 参考。

---

## 总体架构（三段式，全部可微）

```
Frame t
  │
  ├─ [Crop] 以 EKF 预测中心 + 3× 边长（首帧用 GPS 投影做种子）
  │
  ├─ [Prior Rendering] 渲染两路 4 通道高斯热图
  │      • prev_prior  = 上一帧 H × runway 3D 模板 → 4 角点
  │      • gps_prior   = GPS/IMU + PnP 投影 → 4 角点
  │      • VP_L 先验   = 从 IMU yaw 推出，渲染成方向向量通道（2ch）
  │      → 共 8 + 2 = 10 通道先验
  │
  ├─ [Network] 11 ch RGB + 10 ch prior = 13 ch 输入
  │      Backbone : MobileNetV3-Small (in_ch=13, 新通道 zero-init)
  │      Neck     : BiFPN-lite (3 层)
  │      Heads    :
  │        A. Corner Head    — DSNT 或 RLE-μ（无 σ），输出 4×(x,y)
  │        B. Line AFM Head  — Attraction Field Map（辅助监督，不进推理）
  │        C. VP Head        — 直接回归 VP_L、VP_S 在球面坐标（处理无穷远）
  │
  ├─ [Geometric Refinement Block] 全可微，纯矩阵运算 (~0.3ms)
  │      1. DLT 从 4 角点拟合单应矩阵 H_raw（4×2 → 3×3）
  │      2. 投影 runway 3D 模板（5 线 + 中线 + 4 角）→ "理论位置"
  │      3. 加权融合 raw 与理论位置 → 输出最终 H
  │      4. 从 H 解析出所有输出：4 角点、5 边线、1 中线、2 VP
  │
  └─ [Output Smoothing] α-β 滤波器（不是 EKF，因为用户跳过不确定性）
         在 H 的 8 个参数上做指数加权平滑，下一帧 prior 用平滑后的 H 投影
```

---

## 关键设计决策

| 维度 | 选择 | 理由 |
|---|---|---|
| 主输出 | **单应矩阵 H (8 DOF)** 通过 4 角点参数化 | 一次性硬编码所有平行/共面/消影约束；输出维度最低；不可能产生几何矛盾 |
| 输入先验 | RGB + 8ch 角点热图（prev+gps）+ 2ch VP_L 方向场 | 显式注入图像/几何/IMU 三类先验，零初始化新通道保留预训练 |
| 主干 | MobileNetV3-Small @ 512²，13ch in | Jetson Orin NX TensorRT FP16 ~6ms 实测；ConvNeXt-Atto / RepVGG-A0 备选 |
| 角点头 | DSNT（不接 RLE 的 σ 输出） | 用户跳过不确定性；DSNT 子像素精度足够；输出 4×(μx, μy) 共 8 数 |
| 线段头 | **Attraction Field Map (AFM)** — 仅训练时辅助监督 | 推理时所有线段从 H 解析；AFM 让 backbone 学到强边线特征，**但不进入推理时延** |
| VP 头 | 球面坐标（θ, φ）参数化，避免无穷远点数值问题 | 跑道航向几乎平行光轴时 VP 会跑到图像外甚至无穷远，欧式坐标会爆 |
| 几何精化 | 可微 DLT + 模板重投影融合 | DLT 闭式解 ~0.1ms；融合权重 α 可学，让网络自己决定信原始角点还是信几何模板 |
| 时序滤波 | **α-β 滤波器在 H 参数上**（不是 EKF） | 用户跳过不确定性 → 不需要协方差；α-β 足够，~10μs CPU |
| 中线处理 | 几何推导，不独立学习 | 中线 = 左右边线中位线，标注成本零，监督信号无歧义 |

---

## 几何一致性 Loss（核心）

总 loss 由 6 项组成，前 2 项是数据 loss，后 4 项是**几何先验 loss**（无需任何标注）：

```
L_total = L_corner + L_line_afm
        + λ₁ L_homography_consistency
        + λ₂ L_vp_consistency
        + λ₃ L_cross_ratio
        + λ₄ L_parallel
```

| Loss 项 | 公式概要 | 强制的几何约束 |
|---|---|---|
| **L_corner** | Wing loss on 4 (x,y) | 数据 driven |
| **L_line_afm** | L1 on attraction field map | 数据 driven，让 backbone 学边线特征 |
| **L_homography_consistency** | ‖H · template_pts − detected_pts‖² | 4 角点必须对应**某一个**合法 H |
| **L_vp_consistency** | (VP_pred − VP_from_H)² + (VP_pred − VP_from_IMU)² | 网络 VP 头、H 推出的 VP、IMU 推出的 VP 三方互锁 |
| **L_cross_ratio** | (cr(4 detected) − cr(4 template))² | 利用已知矩形交比不变 |
| **L_parallel** | 左/右/中线方向与 VP_L 夹角 → 0 | 平行性约束（H 已隐含，显式加快收敛） |

**关键洞见**：除 L_corner 外，所有约束都不需要额外标注。中线、VP、平行、共面——全部从 4 角点 GT 解析推导。这意味着可以用现有 `dataset/meta.json` 直接训练，**零额外标注成本**。

---

## 新工程目录

```
runway_detector/
├── configs/default.yaml
├── geometry/
│   ├── runway_geometry.py        # 从旧 correct_projection.py 精简移植
│   ├── homography.py             # 可微 DLT、H 分解、VP 提取
│   └── vanishing_point.py        # 球面参数化、VP-from-IMU 推导
├── data/
│   ├── dataset.py
│   ├── prior_renderer.py
│   ├── prior_simulator.py
│   └── augmentation.py
├── models/
│   ├── backbone.py
│   ├── neck.py
│   ├── heads/{corner_head.py, afm_head.py, vp_head.py}
│   ├── refine.py
│   └── runway_net.py
├── losses/
│   ├── corner_loss.py
│   ├── afm_loss.py
│   └── geometric_loss.py
├── filters/alpha_beta_h.py
├── train.py
├── eval.py
├── inference/{pipeline.py, export_trt.py}
└── README.md
```

---

## 关键工程点

1. **可微 DLT 数值稳定性**：8×8 SVD 必须 fp32（即使其余 fp16）；坐标必须归一化到 [-1,1]。
2. **VP 球面参数化**：用单位向量 (sin θ cos φ, sin θ sin φ, cos θ)，回归 (θ, φ)。
3. **先验通道零初始化**：13ch 第一层 conv 前 3 通道保留 ImageNet 权重，后 10 通道置零。
4. **训练-推理裁剪一致性**：训练用 GT 角点 bbox + 随机增广，推理用 α-β 平滑 H 推出的角点，首帧用 GPS 种子。
5. **几何 loss ramp-up**：前 5 epoch 只用数据 loss，之后线性 ramp 几何 loss 权重。
6. **AFM 不进推理图**：ONNX export 时剪除 AFM head，TensorRT 自动优化。
7. **中线 GT 解析推导**：bot_mid = (BL+BR)/2，top_mid = (TL+TR)/2，centerline = (bot_mid, top_mid)。
8. **可靠性测试**：训练时随机让一个先验通道失效，强迫网络学到优雅降级。

---

## 复用清单（不重写）

| 文件 | 复用内容 |
|---|---|
| `D:\projects\approach\correct_projection.py` | `CoordinateTransformer`、`RunwayProjector.__init__`、`project_corners()` |
| `D:\projects\approach\generate_dataset.py` | `get_square_crop()` SCALE=3.0 裁剪逻辑 |
| `D:\projects\approach\dataset\meta.json` | 已有 4 角点 GT |
| `D:\projects\approach\dataset\{train,test}\*.jpg` | 已有 crop 图像 |

---

## 验证计划

**单元级**
1. `homography.py` DLT 反投影误差 < 0.01px
2. `vanishing_point.py` IMU-VP 与角点拟合-VP 角度差 < 0.1°
3. `refine.py` idempotency 检查
4. `prior_renderer.py` 可视化

**精度指标（dataset/test/）**
5. 4 角点平均 px 误差 **< 2px**
6. 5 边线端点平均 px 误差 **< 3px**
7. 中线端点平均 px 误差 **< 3px**
8. VP_L 角度误差 **< 0.5°**
9. **几何一致性**：H from 角点 vs. H from AFM 解码线段，重投影差 < 1px

**可靠性指标**
10. 遮挡 1 角点，剩余精度退化 < 50%
11. GPS 先验注入 50px 偏置，输出退化 < 30%
12. 跨视频泛化（训 V1+V2，测 V3）角点误差 < 4px

**时延（Jetson Orin NX）**
13. TensorRT FP16 单帧 < 15ms
14. 完整 pipeline（裁剪+先验+推理+精化+α-β+解码）< 25ms

---

## 前沿算法参考

| 模块 | 来源 | 借鉴 |
|---|---|---|
| Sports field calibration | PnLCalib (CVPR 2024), TVCalib | H 作为主输出范式 |
| Line representation | AFM (CVPR 2019), F-Clip (CVPR 2022) | Attraction Field Map |
| Vanishing Point | NeurVPS (NeurIPS 2019), CONSAC (CVPR 2020) | 球面参数化、IMU prior |
| Differentiable DLT | DSAC++, DeTone 2016 | 可微 H 估计数值技巧 |
| Backbone | MobileNetV3-Small / RepVGG-A0 / EfficientNet-Lite0 | Jetson 实时主干 |

故意**不选**的：SAM2 / GroundingDINO（>100ms 实时不可行）、DETR 系（query 解码慢）、DeepLSD（精度高但延迟超预算）、Transformer 主干（Jetson INT8 弱）、EKF（无 σ 时 α-β 足够）。

---

---

# 附录 A：加入不确定性 + 位姿生成后的架构差异

主方案刻意省略不确定性（UQ）和位姿头。本附录列出**重新引入这两项**时需要做的所有架构层面改动，可作为后续阶段升级路线图。

## A.1 总览：影响范围

| 模块 | 当前（无 UQ/位姿） | 加入 UQ/位姿后 |
|---|---|---|
| Corner head | DSNT 出 (μx, μy) ×4 | **RLE** 出 (μx, μy, σx, σy, ρ) ×4 |
| AFM head | scalar attraction field | scalar mean + log σ² 双通道 |
| VP head | (θ, φ) 球面 ×2 | (θ, φ, κ) 球面 + von Mises-Fisher 集中度 ×2 |
| Refinement | DLT (均匀权重) | **Weighted DLT** (σ⁻² 加权) + Jacobian 协方差传递 |
| New head | — | **Pose head: EPro-PnP** 输出 (R, t, Σ₆ₓ₆) |
| Temporal filter | α-β on H | **EKF** on [H_params; pose] with R = network σ |
| Prior rendering | 固定 σ 高斯 | EKF 协方差 → **自适应 σ** 高斯 |
| Losses | Wing / L1 | **NLL 形式**（aleatoric uncertainty learning） |
| 新增 loss | — | 位姿重投影 NLL + 位姿-GT NLL |
| Training GT | corner only | corner + **pose**（.txt 中已有，零额外标注） |
| 推理时延增量 | baseline ~10ms | **+4ms**（仍在 30ms 预算内） |

## A.2 Corner Head: DSNT → RLE

**RLE (Residual Log-likelihood Estimation, Li et al. ICCV 2021)** 把关键点回归视为概率密度估计：

- 输出 5 参数/角点：μx, μy, σx, σy, ρ（双变量正态分布）
- Loss = -log p(y_gt | μ, Σ)，σ 直接学到数据噪声
- 推理时只取 μ（精度与 DSNT 相当），σ 给下游协方差链使用

替代方案：保留 DSNT，额外加 σ-head（轻量 conv 输出 4 个 σ 值）。两者效果接近，RLE 更端到端。

## A.3 Refinement: DLT → Weighted DLT + 协方差传递

**Weighted DLT**：标准 DLT 求 H 时所有角点等权，加权 DLT 用 σᵢ⁻² 作为行权重。被遮挡/远端的不确定角点对 H 影响自动减小。

**协方差传递链**：
```
Σ_corners (8×8 块对角)  ──[∂H/∂corners Jacobian]──>  Σ_H (8×8)
Σ_H                     ──[EPro-PnP 内部 Jacobian]──>  Σ_pose (6×6)
```
全闭式可微，~0.5ms 增量。

## A.4 New Head: Pose via EPro-PnP

**EPro-PnP (Chen et al. CVPR 2022 oral)** = End-to-End Probabilistic PnP：

- 输入：N 个 2D 点 (μ, Σ) + 对应 3D 点（跑道 4 角 ECEF 已知）
- 输出：6-DoF 位姿 (R, t) + 6×6 协方差 Σ_pose
- 完全可微，5-10 次 Gauss-Newton，~2ms

**为什么独立做位姿而不从 H 分解**：H = K[r₁ r₂ t] 的分解在平面退化情形（俯视）下数值病态，且不能直接给协方差。让 EPro-PnP 单独跑，与 H 解耦更稳健。

## A.5 时序滤波: α-β → EKF

**状态向量扩展**：
```
x = [H_params (8); pose_se3 (6)] ∈ R¹⁴
```

**观测**：
- z_corners = network corners + RLE σ → R_obs from σ
- z_pose = EPro-PnP output + Σ_pose → R_obs from Σ_pose

**运动模型**：用 IMU 角速度 + 比力做 SE(3) 上的预积分。这构成视觉-惯性松耦合（VIO-lite），是航电级输出的标配。

## A.6 Prior Rendering 自适应 σ

当前用固定 σ 渲染。加 UQ 后：

- **prev_prior σᵢ** = EKF 上一帧后验协方差对角线开方
- **gps_prior σᵢ** = GPS HDOP × focal_length / distance（距离越远投影越不准，物理正确）

网络看到 σ 大的先验自动学到降低信任。这是几乎"免费"的可靠性收益——只要协方差链接通，整个 pipeline 自适应。

## A.7 Loss 全部改 NLL

| 当前 | 加 UQ 后 |
|---|---|
| L_corner = WingLoss(pred, gt) | L_corner = -log 𝒩(gt; μ_RLE, Σ_RLE) |
| L_afm = ‖AFM_pred - AFM_gt‖₁ | L_afm = ½(AFM_pred - AFM_gt)² / σ²_pred + log σ_pred |
| L_vp = 1 - cos(VP_pred, VP_gt) | L_vp = -log vMF(VP_gt; μ_pred, κ_pred) |
| — | L_pose_reproj = Σᵢ ‖π(R·Xᵢ + t) - cornerᵢ‖²_{Σ_corners⁻¹} |
| — | L_pose_gt = ‖log(R_gt⁻¹·R_pred)‖² + ‖t_gt - t_pred‖² |

NLL 形式让网络自动学每个样本难度——困难样本 σ 增大、loss 不爆，模型自动 robust。

## A.8 新增训练数据：位姿 GT 是免费的

`.txt` 文件每帧已记录 (lat, lon, alt, yaw, pitch, roll)，加相机安装外参可直接算相机相对跑道坐标系的位姿 GT。**零额外标注**。

## A.9 新增评估指标

| 指标 | 含义 | 目标 |
|---|---|---|
| **ATE** (Absolute Trajectory Error) | 位姿位置 RMSE | < 2m @ 跑道 1km 外 |
| **RPE** (Relative Pose Error) | 帧间位姿误差 | < 0.5° @ 0.1Hz |
| **NLL** | 不确定性校准（似然） | < 1.0 |
| **ECE** (Expected Calibration Error) | σ 与实际误差的 reliability | < 0.05 |
| **Sharpness** | 平均 σ 大小 | 在 NLL 约束下尽可能小 |

## A.10 计算预算对比

| 组件 | 加 UQ/位姿前 | 加 UQ/位姿后 | Δ |
|---|---|---|---|
| Backbone | 6.0ms | 6.0ms | 0 |
| Heads | 3.0ms | 3.5ms (RLE + σ-AFM) | +0.5 |
| Refinement (DLT) | 0.3ms | 0.8ms (加权 + Jacobian) | +0.5 |
| Pose (EPro-PnP) | — | 2.0ms | +2.0 |
| Temporal (α-β / EKF) | 0.05ms | 0.5ms | +0.5 |
| Prior render | 1.0ms | 1.0ms | 0 |
| **总计** | **~10.4ms** | **~13.8ms** | **+3.4ms** |

仍远低于 30ms 预算。完整 pipeline（含裁剪/解码/I/O）应在 20-25ms。

## A.11 分阶段实施路线图

如果未来要把 UQ/位姿加回来，**不要一次性引入**——按阶段独立验证收益：

| 阶段 | 改动 | 期望收益 | 独立验证方法 |
|---|---|---|---|
| 1 | Corner head: DSNT → RLE，loss → NLL | NLL 改善 ~30%，精度不退 | 测 ECE 是否 < 0.1 |
| 2 | Refinement: 加权 DLT + σ → Σ_H 传递 | Σ_H 与实际 H 误差相关 > 0.5 | 蒙特卡洛仿真验证协方差传递正确性 |
| 3 | 接入 EPro-PnP 输出 (R, t, Σ_pose)，加位姿 NLL loss | ATE < 5m，RPE < 1° | 与 .txt GT 直接对比 |
| 4 | α-β → EKF；EKF cov 反馈到下一帧 prior σ | 闭环可靠性大幅提升；遮挡/丢失场景从崩溃变为优雅降级 | 故意注入故障，看输出 σ 是否真的爆开 |

每阶段独立 ablation，避免一次性引入多变量调试困难。

## A.12 风险与权衡

| 风险 | 后果 | 缓解 |
|---|---|---|
| RLE σ 训练不稳定（σ → 0 或 ∞） | 协方差链失效 | softplus + 1e-3 下界，warm-up 期固定 σ |
| EPro-PnP 在共线/退化情形发散 | 位姿协方差爆炸 | cheirality + condition number 检查；退化时切回 H 分解 |
| EKF 运动模型与真实不符 | 滤波偏置 | 用 IMU 预积分而非恒速；做 outlier rejection (Mahalanobis 门限) |
| 多任务 loss 权重失衡 | 某任务 dominate | Kendall 不确定性加权（每任务一个可学 log σ²） |
| GPS prior 在着陆瞬间不可用（GNSS 拒止） | 单一先验源失败 | 训练时强制 random drop GPS 通道，让 prev-frame + IMU 顶替 |

---

# 附录 B：讨论归档

本方案随对话演进，关键节点：

1. **初始问题**：用户提出跑道关键点 + 边线检测，两路先验（GPS/IMU 投影 + 前一帧）。期望前沿算法综述与可行方案。
2. **首轮综述**：列出先验注入（多通道拼接 / Query 初始化 / Promptable / Deformable attention）、时序融合（Kalman / CoTracker3）、几何约束（可微 PnP / VP / cross-ratio）、边线（DeepLSD / LETR / HAWP）四个轴向，给出 A 路径（增量）和 B 路径（重构）。
3. **进入 Plan 模式**：用户要求重新打开文件夹开发，给出具体方案。
4. **第一版 plan**：PriorFusion-RunwayNet——MobileNetV3 + BiFPN + RLE 关键点 + F-Clip 线段 + EPro-PnP + EKF + 多任务不确定性加权。机载实时（确认 ≤30ms）+ 全套航电输出（角点+边线+位姿+不确定性）。
5. **范围收缩**：用户要求暂时**去掉不确定性和位姿**，专注检测器；同时**扩展先验集合**——新增跑道宽度已知、4 角共面、平行性、消影点、入口/出口边线平行等几何先验；新增中线检测目标。
6. **第二版 plan（当前主体）**：HomographyFirst-RunwayDetector——以单应矩阵 H 为主输出原语，4 角点 + AFM 辅助 + VP 三头；几何精化块用可微 DLT 把所有先验编码进结构；α-β 滤波器替代 EKF（无 σ 时足够）；中线 GT 解析推导（零额外标注）。
7. **本附录 A 起源**：用户问"加入 UQ 与位姿会有哪些差异"——故有以上 12 小节的差异分析与分阶段升级路线图。

**两版方案的本质区别**：

- **v1 PriorFusion** ：先验作为输入通道 + 多任务输出（含 UQ/位姿）+ 端到端 NLL。强调"什么都学"。
- **v2 HomographyFirst**：用几何关系把 18 维输出压成 8 维 H + 解析推导其它要素。强调"只学最少必需的，剩下交给几何"。

v2 在精度上限可能不如 v1（因为没用全部信号），但在**可靠性**和**实时性**上有结构性优势——所有输出永远几何自洽，且推理时延更低。附录 A 描述了如何把 v2 渐进升级到 v1 的能力上限。
