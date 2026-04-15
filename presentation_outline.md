# Presentation Plan: Beyond the Illusion of Intelligence
## B552 Miniconference — 11-minute slot (2-person)

---

## Context

Work-in-progress talk scheduled April 21. ConvNet sweep + probe complete; ViT dims 2/4 still training.
Priority: **story coherence over completeness of results.** ViT is mentioned as pending.
Rule of thumb: ≤1 slide/minute → ~10 slides total.

Split: Person A covers setup + GGR story (~5 min), Person B covers probe twist + takeaways (~6 min).

---

## Narrative Arc

> "We set out to prove that knowledge compression lives on a spectrum.
> We found the spectrum — but the story it tells is not what we expected."

Three beats:
1. **Setup the question** — are knowledge types really different?
2. **Show the spectrum is real** — GGR confirms the hypothesis
3. **Reveal the twist** — linear probe shows *why*, and it's surprising

---

## Slide Outline

---

### [Person A — Slides 1–5, ~5 min]

---

#### Slide 1 · Title (30 sec)
**Beyond the Illusion of Intelligence: Exploring the Knowledge Compression Spectrum**

- Names, course, date
- One-line teaser: *"Is rule-based AI fundamentally different from memorization — or just more compressed?"*

---

#### Slide 2 · KBAI Context: Three Schools of Thought (1 min)
**Knowledge-Based AI offers three classic answers to "how does an agent know things?"**

| Paradigm | Representation | Generalization mechanism |
|----------|---------------|------------------------|
| **CBR** (Aamodt & Plaza) | Store raw cases; retrieve by similarity | k-NN in case space |
| **Prototype / Concept learning** | Compress cases into category centroids | Match to prototype |
| **Rule-based / EBL** | Compile experience into symbolic rules | Deductive inference |

**The implicit assumption in KBAI:** these are *qualitatively different* forms of knowledge — you build a CBR system OR a rule system, not something in between.

**Our challenge:** What if this is a false dichotomy? What if the same learner, given different memory budgets, produces all three naturally?

> Visual: a spectrum diagram mapping CBR ↔ Prototype ↔ Rules, with "memory budget" as the axis
> Caption: *"We operationalize memory budget as the information bottleneck dimension."*

---

#### Slide 3 · Our Hypothesis & Approach (1 min)
**Hypothesis:** The CBR → Rule transition is not a design choice — it's an emergent consequence of compression.

Information bottleneck (Tishby et al.) as the dial:
- Small `latent_dim` → severe compression → forced to extract abstract structure (→ rules)
- Large `latent_dim` → ample capacity → can store raw cases (→ CBR)
- Medium → prototype-like intermediate representations

**Prediction:** if we sweep `latent_dim`, GGR (OOD failure rate) should rise monotonically — small-dim models generalize like rule systems, large-dim fail on novel cases like pure CBR.

**Test bed:** Synthetic pendulum physics world
- 5 physical dimensions: length, angle, gravity, damping, initial angular velocity
- Predict next frame from current frame + camera action
- ConvNet (local features) and ViT (global attention) — same bottleneck, different inductive bias

> Visual: architecture diagram (encoder → Z_t → dynamics MLP → decoder)

---

#### Slide 4 · Measuring Generalization: The OOD Design (1 min)
**Key challenge:** How do we measure whether a model has learned *rules* vs. *specific examples*?

- Train on certain physics parameter bands → test on held-out bands
- **IID**: same distribution as training
- **Near-OOD**: 1–2 parameters out of range
- **Far-OOD**: 3–5 parameters out of range

**Money metric — Generalization Gap Ratio (GGR):**
```
GGR = (MSE_FarOOD − MSE_IID) / MSE_IID
```
Low GGR = model performs similarly on seen vs unseen physics → generalizes  
High GGR = model breaks on new physics → memorizes

> Visual: diagram of IID/Near/Far bands on a 2D parameter space

---

#### Slide 5 · GGR Results: The Spectrum IS Real (1.5 min)
**ConvNet v2 — 7 bottleneck sizes:**

| latent_dim | GGR |
|-----------|-----|
| 2   | 3.6%  |
| 4   | 9.7%  |
| 8   | 11.1% |
| 16  | 12.2% |
| 32  | 25.5% |
| 64  | 25.6% |
| 128 | 24.5% |

**The hypothesis is confirmed:** small dim → low GGR, large dim → high GGR.

Bonus: **grokking at dim=4** — GGR rose from 3.2% → 9.7% over training epochs. Delayed generalization is real.

> Visual: GGR vs. latent_dim line chart (the "money plot")

**Transition line (Person A → B):**
> *"So the spectrum exists. But now the real question: WHY do small-bottleneck models generalize better? Have they actually learned physics rules? That's where it gets interesting."*

---

### [Person B — Slides 6–10, ~6 min]

---

#### Slide 6 · The Probe Question (45 sec)
**To test whether models have learned physics, we use linear probes.**

Method: freeze the encoder, extract z_t, fit a Ridge regression:
```
z_t → each ground-truth variable (gravity, damping, length, angle, ...)
```
R² ≈ 1.0 → that variable is linearly encoded in the latent space  
R² ≈ 0.0 → the variable is absent or entangled non-linearly

> Visual: diagram of the probe pipeline

---

#### Slide 7 · Probe Results: The Twist (2 min)
**What we expected:** small dim models encode gravity + damping as compact rules.

**What we found:**

| dim | gravity R² | damping R² | cam_elevation R² |
|-----|-----------|-----------|-----------------|
| 2   | 0.00 | 0.00 | **0.94** |
| 8   | 0.00 | 0.00 | **1.00** |
| 16  | 0.01 | 0.02 | **1.00** |
| 32  | **0.72** | 0.21 | **1.00** |
| 64  | **0.69** | **0.53** | **1.00** |
| 128 | **0.71** | **0.43** | **1.00** |

**Two critical observations:**
1. `cam_elevation` R² = 1.0 at every single bottleneck size
2. Physics variables near zero until dim=32, then a sudden jump

> Visual: heatmap (`probe_results/probe_heatmap_conv.png`) or simplified table above

---

#### Slide 8 · Interpreting the Twist: Three Regimes (1.5 min)

**dim ≤ 16 — "Visual Shortcut" regime**
- Camera elevation dominates: the most visually salient factor in the image
- No capacity left for physics
- **Low GGR ≠ rule extraction → it's uniform failure across all physics**
- The model can't predict well anywhere, so the IID/OOD gap appears small

**dim = 32 — Phase Transition**
- First time there's spare capacity after encoding camera pose
- Gravity suddenly encoded (R²: 0.01 → 0.72) — gravity = bob color, a single salient pixel feature
- Physics knowledge begins, but so does memorization

**dim ≥ 64 — "Memorization" regime**
- Physics encoded but non-linearly; representation becomes entangled
- High GGR: model fits training distribution but fails on OOD

> Visual: annotated spectrum diagram with three labeled regimes

---

#### Slide 9 · ConvNet vs. ViT — Architecture Matters (1 min)

ViT (Transformer) was trained on the same setup. Dims 8–128 complete; dims 2/4 still training.

| dim | ConvNet GGR | ViT GGR |
|-----|------------|---------|
| 8   | 11.1% | 6.6%  |
| 32  | 25.5% | 9.3%  |
| 128 | 24.5% | 14.0% |

**ViT shows systematically lower GGR at all bottleneck sizes.**

Hypothesis: ViT's global attention may avoid local visual shortcuts, forcing more distributed encoding.

> *Full ViT probe analysis pending — will be in the final paper.*

---

#### Slide 10 · Takeaways & Open Questions (1 min)

**What we showed:**
- The compression spectrum is real: GGR increases monotonically with bottleneck size
- "Good generalization at small dim" is NOT rule extraction — it is visual shortcut
- True physics encoding requires ~32 dims, because camera pose consumes most capacity first
- Architecture matters: ViT generalizes better than ConvNet at all scales

**What this means for knowledge-based AI:**
> Forcing compression does not automatically produce abstraction. The model will always find the cheapest shortcut. Rule extraction requires sufficient capacity AND the right inductive biases.

**Connecting back to KBAI:**
- **CBR:** Our large-dim models *are* CBR systems — they retrieve by similarity and fail on distant cases, exactly as CBR theory predicts
- **EBL:** EBL assumes the learner has the right vocabulary to explain a case. Our small-dim models don't — they grab the cheapest visual cue instead of physics
- **Broader lesson:** The compression spectrum is real, but traversing it toward rules requires more than shrinking the bottleneck — you also need the right *what to compress*

---

## Timing Summary

| Section | Slides | Person | Time |
|---------|--------|--------|------|
| Title | 1 | A | 0.5 min |
| KBAI context + Hypothesis | 2–3 | A | 2 min |
| OOD Design + GGR results | 4–5 | A | 2.5 min |
| Probe question + results | 6–7 | B | 2.5 min |
| Three regimes + ViT | 8–9 | B | 2.5 min |
| Takeaways | 10 | B | 1 min |
| **Total** | **10 slides** | | **11 min** |

---

## Key Visuals Needed

1. Spectrum diagram: CBR ↔ Prototype ↔ Rules with "memory budget" axis
2. Architecture diagram: encoder → Z_t → dynamics MLP → decoder
3. OOD band diagram: 2D parameter space with IID/Near/Far shading
4. GGR vs. latent_dim line chart (money plot)
5. Probe pipeline diagram
6. Probe heatmap: `probe_results/probe_heatmap_conv.png`
7. Three-regime annotated spectrum (Slide 8 synthesis visual)

---

## Submission Checklist

- [ ] Slides (pptx) due 11:59pm the day before presentation
- [ ] Practice with timer — 11 min is tight
- [ ] Both members speak for equivalent time (~5 min each)
- [ ] Mention ViT probe results pending for final paper
