# PriorFusion-RunwayNet 实施方案（v1，完整航电级输出）

> 本方案为对话演进中的**第一版**，目标是**全套航电级输出**：角点 + 边线 + 位姿 + 不确定性。
> 后续因范围收窄演进为 v2 HomographyFirst（仅检测器，见 `runway_detector_design.md`）。

## Context

机载实时跑道视觉特征检测，覆盖整条几何与定位链条。约束与输出：

- **部署目标**：机载实时 ≤ 30ms/帧（Jetson Orin NX 级硬件）
- **输出范围**：4 角点 + 5 条边线 + 6-DoF 位姿 + **逐输出不确定性**
- **可用先验**：上一帧检测结果、基于 GPS/IMU + PnP 的 4 角点投影、IMU 角速度/比力

**设计思想**：端到端多任务学习。所有几何要素都作为网络输出，用 Kendall 多任务不确定性加权统一训练；先验作为输入通道注入，让网络自己学到何时信任先验；EPro-PnP 把位姿求解嵌入网络末端，并把不确定性沿协方差链一路传到 EKF。

---

## 总体架构

```
Frame t  +  上一帧后验状态  +  GPS/IMU/.txt
  │
  ├─ [Crop] 以 EKF 预测中心 + 3× 边长（首帧用 GPS/PnP 投影做种子）
  │
  ├─ [Prior Rendering] 8 通道高斯热图
  │      prev_prior : 上一帧 4 角点，σ = EKF 后验协方差 → 自适应
  │      gps_prior  : GPS+IMU+PnP 投影 4 角点，σ = HDOP × focal / distance
  │
  ├─ [Network] 11 ch 输入（RGB 3 + Prior 8）
  │      Backbone : MobileNetV3-Small (in_ch=11, 新通道 zero-init)
  │      Neck     : BiFPN-lite (3 层)
  │      Heads    :
  │        A. RLE Corner Head   — 5 params/角点 (μx, μy, σx, σy, ρ)
  │        B. Line Head         — F-Clip / HAWP-v3 风格，端点 + σ
  │        C. EPro-PnP Pose Head — (R, t) + Σ₆ₓ₆
  │
  ├─ [Multi-task Loss] Kendall (CVPR 2018) 不确定性加权
  │      4 个任务 log σ²_task 可学，自动平衡，避免手调权重
  │
  └─ [Temporal] EKF on state = [pose_se3 (6) ; corners (8) ; ...]
        网络 σ → R_obs；IMU 预积分 → 运动模型；
        EKF 后验 → 下一帧 prior 的 σ（闭环自适应）
```

---

## 关键设计决策

| 维度 | 选择 | 理由 |
|---|---|---|
| Backbone | MobileNetV3-Small @ 512² | Jetson Orin NX TensorRT FP16 ~6ms |
| 输入通道 | 11ch (RGB + 4ch prev + 4ch GPS prior) | 显式三类先验注入；新通道 zero-init 保留预训练 |
| 角点头 | **RLE** 5-param (μx, μy, σx, σy, ρ) | ICCV 2021，异方差不确定性，端到端 NLL |
| 线段头 | **F-Clip / HAWP-v3** 风格 wireframe + 端点 σ | 实时（<5ms@RTX 级），输出可解码端点 |
| 位姿头 | **EPro-PnP** (CVPR 2022 oral) | 可微 PnP + 6×6 协方差输出；与 RLE σ 链路天然衔接 |
| 多任务 loss | **Kendall 不确定性加权** | 4 个任务 log σ²_task 可学，自动平衡，免去手调 λ |
| 时序滤波 | **EKF**，状态含 pose + 角点 + 线段参数 | RLE/EPro-PnP σ 作为观测协方差 R；IMU 预积分作为运动模型 |
| 部署 | TensorRT FP16 → INT8 校准 | Jetson 标准实时栈 |

---

## 多任务 Loss（Kendall 加权）

```
L_total = Σ_i (½ · L_i / σ²_i  +  log σ_i)
其中 σ_i 是每任务一个可学习参数（log σ²_i 实参数化）
```

四个任务的 NLL：

| L_i | 形式 |
|---|---|
| **L_corner** | RLE NLL：-log p(y_gt \| μ, Σ) 双变量正态 |
| **L_line** | 线段端点 NLL + AFM 监督 |
| **L_pose_reproj** | Σⱼ ‖π(R·Xⱼ + t) − cornerⱼ‖²_{Σ_corner⁻¹}（用 RLE σ 加权） |
| **L_pose_gt** | ‖log(R_gt⁻¹·R)‖² + ‖t_gt − t‖²（GT 从 .txt 直接算） |

Kendall 加权让易学任务（如角点）自动获得小 σ → 大权重；难学任务（如位姿在远距离时）σ 大 → 权重自然下调，**无需手调超参**。

---

## 工程目录

```
runway_detector/
├── configs/default.yaml
├── projection/                   # 精简移植 correct_projection.py
│   ├── coordinate_transform.py
│   └── runway_projector.py
├── data/
│   ├── dataset.py                # 读 meta.json + .txt 位姿 GT
│   ├── prior_renderer.py         # 高斯热图（σ 自适应）
│   ├── prior_simulator.py        # 训练时合成 prev-frame & GPS 先验
│   └── augmentation.py
├── models/
│   ├── backbone.py               # MobileNetV3-Small, in_ch=11
│   ├── neck.py                   # BiFPN-lite
│   ├── heads/
│   │   ├── rle_corner_head.py    # μ + σ + ρ
│   │   ├── line_head.py          # F-Clip style
│   │   └── epropnp_head.py       # EPro-PnP 求解 + Σ_pose
│   └── runway_net.py
├── losses/
│   ├── rle_loss.py
│   ├── line_loss.py
│   ├── pose_loss.py              # 重投影 NLL + 位姿 GT NLL
│   └── kendall_weighting.py      # 多任务自适应加权
├── filters/
│   └── ekf.py                    # state = [pose; corners]，IMU 预积分运动模型
├── train.py
├── eval.py
└── inference/
    ├── pipeline.py               # 在线 pipeline（crop → prior → net → EKF）
    └── export_trt.py             # ONNX → TensorRT FP16 / INT8
```

---

## 关键工程点

1. **新通道 zero-init**：第一层 conv 前 3 通道保留 MobileNetV3 ImageNet 权重，后 8 通道初始化为 0。训练首步等价纯 RGB 模型，先验通道权重渐进增长，避免预训练特征被破坏。

2. **σ 下界裁剪**：softplus(raw) + 1e-3，防 σ → 0 时 NLL 发散；同时给 σ 上界（如 100px）避免训练初期 σ → ∞ 让 loss 退化为常数。

3. **EPro-PnP 集成**：5-10 次 Gauss-Newton 迭代，全可微，Jetson 实测 ~2ms。坐标必须 fp32。退化场景（4 角共线）切回 H 分解兜底。

4. **EKF 与网络解耦**：网络只输出单帧观测 (μ, Σ)，EKF 在后处理跑（CPU 即可，~0.5ms）。这样滤波器可独立调试与评估，也便于把 EKF 升级为 UKF/iEKF 而不动网络。

5. **训练-推理裁剪一致性**：训练用 GT 角点 bbox + 随机偏移/缩放，推理用 EKF 预测 bbox。首帧裁剪用 GPS+IMU+PnP 投影做种子。

6. **Kendall 多任务自动加权**：每任务一个可学 log σ²_task 参数，初始化为 0；训练自然演化出权重，无需手调 4 个 λ。这是 v1 与 v2 在 loss 工程上的核心区别——v2 手调几何先验权重，v1 全部交给数据驱动。

7. **EKF 闭环 prior σ**：EKF 后验协方差对角线 → 下一帧 prev_prior 的 σ。GPS 拒止时 prev_prior σ 自动膨胀，网络自动降低对 prev 的信任。整条链协方差贯通是"航电级"可靠性的核心。

---

## 复用清单

| 文件 | 复用内容 |
|---|---|
| `correct_projection.py` | `CoordinateTransformer` 全部变换链；`RunwayProjector.__init__` 的相机/跑道/外参常量；`project_corners()` 用于生成 GPS prior 与位姿 GT |
| `generate_dataset.py` | `get_square_crop()` SCALE=3.0 裁剪逻辑 |
| `dataset/meta.json` | 已有 4 角点 GT |
| `dataset/{train,test}/*.jpg` | 已有 crop 图像 |
| `.txt` 位姿文件 | (lat, lon, alt, yaw, pitch, roll) → 相机相对跑道坐标系位姿 GT，**零额外标注** |

---

## 训练数据合成

| 数据项 | 来源 |
|---|---|
| 4 角点 GT | meta.json（已有） |
| 边线 GT | 4 角点解析推导（无独立标注） |
| 位姿 GT | .txt + 安装外参解析推导 |
| prev_prior 模拟 | GT 4 角加帧间运动扰动 + 随机失效（模拟跟丢） |
| gps_prior 模拟 | PnP 投影 + HDOP-scaled 高斯噪声 + 偶发偏置（模拟差分丢失） |
| IMU 序列 | .txt 帧间差分得姿态/位置增量 |

**关键**：训练时强制 random-drop 各路先验（每路 10% 概率置零），强迫网络学到优雅降级——任一路先验失效不应导致系统性能崩盘。

---

## 验证计划

**单元**
1. RLE σ 在小数据集上训练 100 步无 collapse（σ 不趋 0 也不趋 ∞）
2. EPro-PnP idempotency：给完美 4 角点 + 完美 (R, t) 模板，输出应一致
3. EKF 与 GT 位姿在仿真序列上 ATE < 1m
4. Kendall log σ²_task 训练收敛到合理范围

**精度（dataset/test/）**
5. 角点平均 px 误差 < **3px**
6. 线段端点 < **4px**
7. 位置 ATE < **5m** @ 距跑道 1km 外
8. 姿态 RPE < **0.5°**
9. NLL < **1.0**
10. ECE < **0.05**
11. σ-vs-实际误差 Pearson > **0.6**（v0 DSNT 当前 < 0.3）

**可靠性**
12. 随机 drop 1 路先验，性能退化 < 30%
13. 注入 50px GPS 偏置，输出不发散
14. 短时（< 5 帧）prev_prior 失效，EKF 应维持

**时延（Jetson Orin NX）**
15. TensorRT FP16 单帧推理 < **15ms**
16. 完整 pipeline（裁剪 + 先验 + 推理 + EPro-PnP + EKF + 解码）< **30ms**
17. INT8 校准后精度退化 < 5%

---

## SOTA 算法引用

| 模块 | 来源 | 借鉴的部分 |
|---|---|---|
| **RLE** | Li et al., ICCV 2021 | 异方差关键点回归头（μ + σ + ρ）+ NLL loss |
| **F-Clip / HAWP** | CVPR 2022 / TPAMI 2023 | 实时 wireframe parsing，提供端点输出 |
| **EPro-PnP** | Chen et al., CVPR 2022 oral | 可微 PnP + 协方差输出 |
| **Kendall multi-task** | Kendall et al., CVPR 2018 | 多任务不确定性加权，免手调 λ |
| **MobileNetV3** | Howard et al., ICCV 2019 | 轻量实时主干 |
| **BiFPN** | EfficientDet, CVPR 2020 | 轻量多尺度特征融合 |
| **EKF + IMU 预积分** | Forster et al., RSS 2015 | 视觉-惯性松耦合 |

故意**不选**的：
- SAM2 / SAM (>100ms，实时不可行)
- CoTracker3 / TAPIR (轨迹跟踪太重)
- DINOv2 / EVA-02 backbone (推理慢、Jetson INT8 支持弱)
- DETR / DINO-DETR (query 解码 4 个点是 overkill)
- DeepLSD (~30ms@RTX3090，Jetson 上挤不进预算)

---

## 与 v2 HomographyFirst 的对比

| 维度 | v1 PriorFusion | v2 HomographyFirst |
|---|---|---|
| 主输出原语 | 多任务直接回归（角点 + 线 + 位姿，各自带 σ） | 单应矩阵 H (8 DOF)，其它要素从 H 解析推导 |
| 几何约束 | 软约束（重投影 loss） | 硬约束（H 结构性保证平行/共面/消影） |
| 不确定性 | 显式（RLE / EPro-PnP / EKF 协方差链） | 暂无（用户阶段性需求） |
| 位姿输出 | 显式（EPro-PnP） | 暂无（可由 H 分解兜底） |
| 时序融合 | EKF（带协方差） | α-β（无协方差时足够） |
| 训练 loss 设计 | Kendall 自动加权 4 任务 | 手调 4 个几何先验权重 |
| 精度上限 | 高（信号利用最充分） | 受 H 8-DOF 表征能力限制 |
| 可靠性 | 高（协方差闭环 + 优雅降级） | 高（几何硬约束 + 结构自洽） |
| 工程复杂度 | 高（RLE + EPro-PnP + EKF + Kendall） | 中（角点 + DLT + α-β） |
| 推理时延 | ~14ms | ~10ms |
| 适用阶段 | 需要完整航电输出 / 上线交付 | 验证检测器骨架 / 内部 baseline |

**v1 与 v2 不是替代关系，而是阶段性关系**：v2 是 v1 的精简骨架，可作为 v1 第一阶段（仅检测器）的实现；待检测器稳定后，按 v2 文件附录 A 的"分阶段路线图"逐步加 RLE σ → 加 EPro-PnP → 加 EKF，最终演化到 v1 的完整能力。
