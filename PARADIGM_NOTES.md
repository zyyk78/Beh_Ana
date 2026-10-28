# 实验范式与分析逻辑备忘（供新会话续接）

## 范式关键事实
- Stage4、2026-01-01+、T7F/T9F：172 sessions、14 鼠、130,205 trials（`trials_all.csv`）
- T7F = 双臂老虎机奖励概率 7:3（最优上限 0.70）；T9F = 9:1（最优上限 0.90）
- **block 长度自适应**：切换后达到 21/30（70%）即切走（`fixing.py` window_size=30, success_threshold=21）；87% 的 block 在检测到达标后 2 trial 内结束
- corr(切换延迟, block长度) = 0.7–0.8；block 最短 30 = 检测架构下限
- 172 sessions 全部有硬件 Mod 事件，无重构 session
- 12/14 鼠从 T9F 起步；仅 C05/C06 从 T7F 起步（T7F 21–22 sessions，其他鼠 3–8）

## Phase 1A 切换延迟分析（v4 已定稿）
- 时间到达标 = 28 + latency（28 = 19 窗预热 + 10 持续窗下限）
- floor-hit（lat=0）= 最早检测点即达标，是最快数据，不能当缺失剔除
- KM 覆盖全部 block（无 room 过滤）
- 检出率：T7F 96.2% / T9F 97.3%；floor-hit 42.6% / 56.4%
- block 级 KM 中位 32 / 29；逐鼠 KM 中位 + Mann-Whitney U=144.5, p=0.0009
- 删失：T7F 46（last-block 19 / mid-block 27），T9F 33（15/18）；<30 trial block 丢弃 157（6.0%）
- 55% 阈值敏感性：结论方向不变
- 归一化延迟（lat_norm）已删除：自适应范式下 maxlat 反映表现而非测量空间，归一化无意义

## KM 图约定（用户要求）
- **对数横轴**，LogLocator 自动刻度（不手动 set_xticks 固定刻度）
- xlim = (28, 1500)，起点标注 "detection floor (28)"
- notebook cell id: `p1a_km`（cell 9）

## 其他 Phase
- Phase 2: Q-learning 4 参数逐鼠逐任务（`qlearning_params.csv`）
- Phase 3: HMM K=3（BIC 全胜 K=2），3 状态占用 low/mid/high≈30/37/33%
- Phase 4: logistic 解码器 5-fold AUC（`decoder_results.csv`）
- Phase 5: 训练顺序效应混合模型：T9F +0.110***、n_same +0.006*、n_other −0.004(n.s.)

## 环境/流程
- 内核 `python3.14-base`（/opt/anaconda3/bin/python）
- nbconvert 沙箱失败 → /tmp 脚本手动 exec cell 并写回 outputs
- 内嵌 PNG dpi=150，落盘 savefig 默认同 dpi，哈希比对必须 MATCH
- git 多轮改动未提交（Phase 1A v4、Phase 3–5），备注需简短中文
