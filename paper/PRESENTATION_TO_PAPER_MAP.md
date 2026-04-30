# Presentation → Paper Mapping

**用途:** 你已经在 presentation 里把所有论点结构好了（`slides.md` / `script.md`）。
这张表告诉你**每一段口播内容应该落到 paper 的哪一节**，以及"slide 形态"和"paper 形态"在篇幅/语气上的差别提示。

**严格说明:** 这只是结构对应表 — paper 段落本身必须由你自己写（GenAI 政策）。
我没有把 slide 内容扩写成段落，也不会在 paper 上代笔。

---

## 总览

| Presentation | Paper section in `main.tex` |
|---|---|
| S1 Title | `\title{}` + `\author{}` |
| S2 Plato's cave | Introduction (motivation 第 1 段) |
| S3 Riverbed + gravity | Introduction (motivation 第 2 段) |
| S4 Three constraints | Introduction (motivation 第 3 段) — 可压缩 |
| S5 The Compression Spectrum | Introduction (问题陈述) + Related Work (压缩视角) |
| S6 "I didn't make this up" | Related Work (引用列表) |
| S7 Pendulum testbed | Method §3.1 Synthetic Pendulum Testbed |
| S8 Train corner / OOD design | Method §3.1 (OOD 切分) + Figure 1 |
| S8b The bottleneck is the dial | Method §3.2 The Bottleneck as a Dial + Figure 2 |
| (隐含) GGR + probe 评测说明 | Method §3.3 Evaluation |
| S9 Money plot | Results §4.1 + Figure 3 |
| S10 ConvNet probe | Results §4.2 + Figure 4 |
| S10b ConvNet vs ViT 双图 | Results §4.3 + Figure 5 (双栏 figure*) |
| (新增 — paper-only) DCT 基线 | Results §4.4 |
| S11 Three regimes of pressure | Discussion §5 |
| S12 Title 复现 | Conclusion §6 |
| — | Statement on Use of Generative AI（强制） |
| — | Division of Work（强制） |
| — | Bibliography |

---

## 详细对应

### 1. Title / Authors → Title block

- 直接对应 S1。
- ⚠️ AAAI 要求 **mixed case**（每个实词首字母大写），副标题用 `\\` 换行。
- Authors 块用 `\textsuperscript{\rm 1}` 标 affiliation，模板里我已写好两人占位。

### 2. Abstract → 独立段落 (slide 没对应)

- Slide 没有 abstract。这块要新写，4-6 句涵盖：
  - 1 句问题（compression spectrum 假设）
  - 1 句 setup（pendulum + bottleneck sweep + 双架构）
  - 1-2 句结果（money plot 趋势 + probe twist）
  - 1 句 ConvNet vs ViT structure tax
  - 1 句 takeaway

### 3. Introduction → S2 + S3 + S4 + S5 + S6

- **AAAI intro 比 slide 密**：slide 上一句话的内容，paper 里通常一段。
- 建议结构：
  - **Para 1 (motivation):** S2 的影子比喻 + 中心问题"什么投射出 intelligence"。Paper 里不一定要保留 Plato's cave 的视觉感，但核心 claim "we should ask what casts intelligence, not what intelligence is" 要写出来。
  - **Para 2 (compression as the answer):** S3 的 gravity-of-intelligence 论点。可以保留 canyon 比喻或直接说 compression。
  - **Para 3 (why compression is forced):** S4 三个约束（pattern/genome/energy）。Paper 里这段可以压缩到 3-4 句，也可以挪到 Related Work 开头。
  - **Para 4 (problem statement):** S5 spectrum 视角 — 把 CBR / prototype / rule 视为同一谱上不同点。这是论文的 thesis。
  - **Para 5 (contributions):** **slide 没有**，paper 必须有的列表式贡献：
    - (a) operationalize compression as a tunable bottleneck on a controlled physics task
    - (b) report what is encoded (probe), not just whether the model generalizes
    - (c) show the same physical rule costs different bottleneck size in different architectures

### 4. Related Work → S5 + S6

- **S5/S6 在 slide 是一句话扫过；paper 必须真的引用**。
- 建议三段：
  - 信息瓶颈 / MDL：Tishby et al., Shannon, Kolmogorov（S6 出现的人名）。
  - KBAI 视角：CBR (Aamodt & Plaza), prototype theory (Rosch), rule-based / EBL — 这是这门课的 framing。
  - Linear probing 方法：Alain & Bengio 2016；representation 分析。

> ⚠️ `references.bib` 是空模板。你需要自己加 BibTeX 条目。

### 5. Method §3.1 Synthetic Pendulum Testbed → S7 + S8

- **S7 + S8 合并为一段（或一节的开头）**：Unity 渲染 / 64×64 / 5 个物理维度 / color encoding (gravity→hue, damping→sat) / IID-Near-Far split。
- Figure 1 = `s8_physics_scatter.png`（已嵌入）。
- ⚠️ S7 的 Unity 视频在 paper 里**没有**，要么用 1-2 帧静图，要么靠文字描述。

### 6. Method §3.2 The Bottleneck as a Dial → S8b

- 直接对应 S8b 架构图 + 7 个 bottleneck widths。
- Figure 2 = `s8b_model_diagram.png`（已嵌入）。
- 要写的细节比 slide 多：
  - 编码器结构（ConvNet 4 层 stride-2 conv / ViT 4 层 transformer, embed=128）
  - dynamics MLP 输入是 (z, action)
  - 优化器、学习率、训练 epochs、early stop（z\_std）
  - DCT baseline 的描述放这里也行，或单独一段

### 7. Method §3.3 Evaluation → slide 隐含内容

- Slide 没有专门一页，但 GGR 公式和 linear probe 在 S9/S10 口播里提了。
- Paper 必须显式写出：
  - GGR 公式：`GGR = (MSE_FarOOD − MSE_IID) / MSE_IID`
  - Linear probe 协议：freeze encoder → ridge regression → R² per GT variable
  - Tier A / B / C 分组定义

### 8. Results §4.1 GGR vs Bottleneck Size → S9

- 1-2 段 + Figure 3 (`s9_money_plot.png`)。
- 要写：观察到 GGR 随 dim 单调上升；ConvNet vs ViT 的两条线对比；最有意思的 grokking 现象（dim=4 GGR 从 3.2% → 9.7%）。
- **悬念铺垫**：slide 在这里有"but wait"停顿；paper 里建议在段尾用一句过渡，比如"this raises a question about *what* the small-bottleneck model has actually learned"。

### 9. Results §4.2 Probe — ConvNet → S10

- 1-2 段 + Figure 4 (`probe_heatmap_conv.png`)。
- 关键发现按口播节奏列出：
  - cam_elevation R²≈1.0 全程
  - 物理变量在 dim≤16 时近 0
  - dim=32 gravity 突然跳到 0.72 (phase transition)
- **结论句**：小 dim 的"好泛化"不是规则提取，是 visual shortcut。

### 10. Results §4.3 ConvNet vs ViT → S10b

- 1 段 + Figure 5（双栏宽 `figure*`，已用 `s10b_dual_heatmap.png`）。
- 主论点：同一规则，ViT 要 4× 容量才学会。
- 架构解释（"structure tax"）：ConvNet 的 4×4×256 spatial map 让 bottleneck 只用编码物理；ViT 的 CLS token 必须同时编码"图像怎么组织"和"物理是什么"。

### 11. Results §4.4 DCT Baseline → 没有对应 slide

- 这块**只在 paper 里**有（presentation 没讲）。
- 1 段：DCT (固定编码器) 在所有 dim 下 GGR≈0% → 证明 memorization 是 jointly trained encoder 的属性，不是 bottleneck size 本身造成的。
- 这是 paper 的一个加分项，因为是 critical ablation。

### 12. Discussion §5 → S11

- 1-2 段。
- Three regimes of pressure（太紧/正好/太松）+ "right pressure depends on what you're squeezing"。
- 加一段 limitations：
  - 单一 domain（pendulum）
  - probe 只用 linear；非线性 probe 可能改变 phase transition 位置
  - 5 维物理空间的 holdout 不一定对应真实 OOD
- 加一段 future work：
  - scale up（更大物理空间 / 真实视频）
  - 不同 inductive bias 的系统对比
  - proto-symbolic readout

### 13. Conclusion §6 → S12

- 3-5 句。
- 复述 thesis："predict well, spend less" 的条件版本：当压力调对的时候。
- 一句对 KBAI 的回响：CBR / prototype / rule 是同一学习曲线上的不同点。

### 14. Statement on Use of Generative AI → 强制

- 模板原话：

  > Our only uses of generative AI were for [specific purposes],
  > using [tool(s)]. No generative AI was used for idea generation,
  > outlining, or drafting content.

- 你需要诚实写：
  - 哪些工具用了（Claude Code? Grammarly? ChatGPT?）
  - 用在什么环节（typo / grammar / LaTeX 表格 / figure script / README 写作）
- 不能把 paper 段落代笔的事写进去（因为政策禁止，所以也不该发生）

### 15. Division of Work → 强制（team only）

- 2-3 句，每人贡献。建议分两条：
  - 谁做 Unity 数据生成 / 训练 pipeline / probe / plots
  - 谁写 paper 哪些 section
- 模板：
  > Author One built the Unity data generation environment and ran the
  > ConvNet/ViT/DCT training sweeps and probe analysis. Author Two built
  > the evaluation pipeline (GGR + linear probes) and the plots.
  > Both authors contributed equally to the paper.

---

## "Slide 上没有但 paper 必须有"清单

写 draft 时容易忘的：

1. **Abstract**（独立段落）
2. **Contribution bullets**（intro 末尾）
3. **Related work** 真正的引用 + bib 条目
4. **Hyperparameters / training details**（method 里）
5. **GGR 数学定义**（公式形式）
6. **Probe protocol 细节**（linear, ridge, train/test split）
7. **DCT baseline 段落**（slide 没讲）
8. **Limitations 段**（discussion 里）
9. **Statement on Use of Generative AI**
10. **Division of Work**
11. **Bibliography 条目**

---

## 长度估算

3-4 页 AAAI 双栏 ≈ 2400-3200 词。粗分配：

| Section | 估计词数 |
|---|---|
| Abstract | 150 |
| Introduction | 500-600 |
| Related Work | 250-350 |
| Method | 600-700 |
| Results | 700-900 |
| Discussion | 300-400 |
| Conclusion | 80-120 |
| Statements + Bib | 不计页数（bib 自动 wrap） |

把 6 张图都放进去会占掉相当篇幅，所以正文要紧凑。如果超 4 页，Method 和 Related Work 是优先压缩对象。

---

## 你接下来该做的

1. 把 `main.tex` 上传到 Overleaf（连同 `aaai25.sty`, `aaai25.bst`, `references.bib`, `figures/` 整个目录）。
2. 按上面对应表，每个 `% TODO` 注释处**自己写**段落。
3. 写完一遍完整 draft 再叫我，我可以做 typo / grammar / LaTeX 排版 / 表格调整。
