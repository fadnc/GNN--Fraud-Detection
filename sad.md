# 🎤 FINAL PRESENTATION SCRIPT — Fadhil Muhammed N C
## Smart ICU Assistant: Real-Time Multi-Task Clinical Risk Prediction Using Ensemble Machine Learning
### 📅 Date: 07 May 2026 (Thursday) | ⏰ Time: 10:00 AM
### 🎯 Duration: EXACTLY 15 Minutes + 5 Minutes Q&A
### Individual Tasks: Sepsis · Vasopressor · Ventilation (+ Shared Infrastructure)

---

> [!IMPORTANT]
> **Timing Guide:** Each slide has a **strict time budget** marked in the header. Total speaking time = **15:00 exactly**. Practice with a timer. Speak at ~140 words/minute (natural conversational pace). Do NOT rush — pauses make you sound confident.

> [!TIP]
> **Delivery Tips:**
> - Make eye contact with the panel, not the screen
> - Point to figures/tables when referencing them
> - Slow down for key numbers (AUROC, dataset size, feature count)
> - When you say a number, pause briefly after — let it land
> - Keep water nearby — you'll be talking for 15 straight minutes

---

## SLIDE 1 — Title & Introduction
### ⏱️ Duration: 0:45 | Running Total: 0:45

> Good morning, respected panel members. I'm **Fadhil Muhammed N C**, Registration Number **243014**, MSc Computer Science — Data Analytics, Digital University Kerala, Semester IV.
>
> My project is titled **"ICU Sentinel — Real-Time Multi-Task Clinical Risk Prediction Using Ensemble Machine Learning."**
>
> This is a collaborative project with my teammate **Raha Billa T P**. We each took three clinical prediction tasks. My individual tasks are **sepsis onset prediction, vasopressor requirement prediction, and mechanical ventilation prediction**. The shared infrastructure — the data pipeline, model architectures, training loop, ensemble framework, and the web dashboard — was built jointly.
>
> I'll walk you through the complete system in the next 15 minutes — from the clinical problem to the deployed dashboard.

---

## SLIDE 2 — Problem Statement & Motivation
### ⏱️ Duration: 1:15 | Running Total: 2:00

> **Why does this project exist?**
>
> An ICU patient generates approximately **1,500 data points per day** — heart rate every few minutes, blood pressures, oxygen saturation, respiratory rate, temperature, glucose, lab results arriving every 4–6 hours. A single nurse manages 2–4 patients simultaneously. No human can process all of this data in real time and catch every early warning sign.
>
> The tools clinicians currently have — **APACHE-II** and **SOFA** — are static scoring systems. APACHE-II is computed **once at admission** and never updated. SOFA refreshes **once daily**. Neither captures non-linear relationships between vitals, and neither can detect the gradual deterioration patterns that develop over hours.
>
> The clinical stakes are real. For **sepsis** specifically — which is my primary task — every hour of delayed antibiotic treatment increases mortality by **4 to 8 percent**. That's not a model metric — that's a life.
>
> Our goal was to build a system that continuously monitors patient data and predicts **six critical clinical events, 6 to 24 hours before they happen** — giving clinicians actionable lead time.

---

## SLIDE 3 — Dataset: MIMIC-III
### ⏱️ Duration: 1:00 | Running Total: 3:00

> We trained on **MIMIC-III** — the largest publicly available ICU database, maintained by MIT and Beth Israel Deaconess Medical Center. It spans **2001 to 2012**, and access requires PhysioNet credentialing with HIPAA-compliant data use agreements.
>
> The key numbers:
> - **43.6 gigabytes** of clinical data
> - **61,532 ICU stays** across **46,520 patients**
> - **17 relational CSV tables** — covering patients, admissions, ICU stays, charted vital signs, laboratory results, prescriptions, diagnoses, procedures, and IV inputs
>
> The single largest table — **CHARTEVENTS** — is **33.6 GB** with approximately **330 million rows** of charted observations. Loading this into memory on our 16 GB machine was our first major engineering challenge.

---

## SLIDE 4 — System Architecture & Pipeline Overview
### ⏱️ Duration: 1:15 | Running Total: 4:15

> Here's the full pipeline — seven layers, from raw CSVs to a live dashboard.
>
> **Layer 1 — Data Ingestion**: We load 17 MIMIC-III tables. CHARTEVENTS is loaded in chunks — 2 million rows at a time — filtering by item ID. This drops memory usage by approximately 85%.
>
> **Layer 2 — Feature Engineering**: From raw vital signs and lab values, we engineer **81 features per hourly timestep**. That includes 15 base signals, 3 derived clinical indices, and rolling window statistics.
>
> **Layer 3 — Label Generation**: Each of our six predictor classes implements a `generate_labels()` method using clinically validated definitions. My three are sepsis, vasopressor, and ventilation.
>
> **Layer 4 — Model Training**: Four architectures per task — BiLSTM-Attention, Pre-LayerNorm Transformer, XGBoost, and LightGBM — all trained with FP16 mixed precision.
>
> **Layer 5 — Ensemble Construction**: AUROC-squared weighted averaging plus a stacking meta-learner.
>
> **Layer 6 — Model Serving**: FastAPI backend with checkpoint loading.
>
> **Layer 7 — Dashboard**: Real-time patient monitoring, risk scores, and clinical alerts.
>
> This modular design means any layer can be swapped or extended independently.

---

## SLIDE 5 — Feature Engineering: The 81 Features
### ⏱️ Duration: 1:15 | Running Total: 5:30

> Let me explain how raw clinical data becomes ML-ready input.
>
> We start with **15 base signals**: 8 vital signs — heart rate, systolic/diastolic/mean arterial pressure, respiratory rate, temperature, SpO₂, and glucose — plus 7 laboratory values — creatinine, lactate, WBC, hemoglobin, platelets, bicarbonate, and chloride.
>
> From these, we compute **3 derived clinical indices**:
> - **Shock Index** = heart rate divided by systolic BP. Above 0.7 indicates shock risk — tachycardia plus hypotension means failing perfusion.
> - **Pulse Pressure** = systolic minus diastolic. Narrow suggests cardiogenic shock; wide suggests septic shock.
> - **SIRS Score** — a 0-to-4 score counting: temperature outside 36–38.3°C, heart rate above 90, respiratory rate above 20, WBC outside 4–12 K/µL.
>
> We then compute **rolling window statistics** — mean, standard deviation, min, max, and linear **trend** — over 2, 6, and 12-step windows. The trend feature is critical: a single heart rate of 110 means little, but heart rate *trending upward for 6 hours* while MAP *trends downward* — that's a deterioration trajectory.
>
> The final matrix per sample: **24 timesteps × 81 features**, created using a sliding window with 75% overlap.

---

## SLIDE 6 — Model Architectures
### ⏱️ Duration: 1:15 | Running Total: 6:45

> We use **four architectures** — two deep learning, two gradient-boosted trees — chosen for **complementary inductive biases**.
>
> **BiLSTM with Temporal Attention**: A 2-layer bidirectional LSTM reads the 24-hour sequence from both directions. The attention layer learns which hours are most informative. For sepsis, it might focus on the hour where the temperature spiked. Hidden size 128 per direction, 256 total.
>
> **Pre-LayerNorm Transformer**: Multi-head self-attention with 8 heads and 3 encoder layers. We use Pre-LayerNorm — normalizing *before* attention — because under FP16, the Q·K-transpose dot products can exceed 65,504 and produce NaN. We also force the attention computation to run in FP32.
>
> **XGBoost**: Gradient-boosted trees on flattened 1,944-dimensional vectors. Trees naturally capture threshold-based clinical rules — for example, "if creatinine exceeds 2.0 AND lactate exceeds 4.0, predict high risk." 300 trees maximum with early stopping.
>
> **LightGBM**: Leaf-wise tree growth provides a different tree structure from XGBoost's level-wise approach. This architectural diversity strengthens the ensemble.

---

## SLIDE 7 — Training Pipeline & Key Decisions
### ⏱️ Duration: 1:00 | Running Total: 7:45

> Five critical training decisions:
>
> **One — Mixed Precision (FP16 AMP)**: Our GPU has only 4 GB VRAM. FP16 halves memory usage and doubles throughput on tensor cores. PyTorch GradScaler prevents gradient underflow.
>
> **Two — BCEWithLogitsLoss**: We use this fused operation instead of separate sigmoid + BCELoss. Under FP16, sigmoid can saturate to exactly 0 or 1, making log(0) = NaN. The fused version is numerically stable.
>
> **Three — Temporal Split**: 70/15/15 chronological by admission time. Random splitting would cause temporal leakage — the model learning from future patients to predict past ones.
>
> **Four — Class Imbalance Handling**: Per-task `pos_weight` = N_negative / N_positive. Sepsis is ~8% positive, mortality at 6h under 2%. Without weighting, models learn to predict "no" every time.
>
> **Five — NaN Detection**: Automatic abort after 10 consecutive NaN validation epochs. This saved hours when our TCN model kept diverging — which is why we ultimately removed TCN.

---

## SLIDE 8 — Ensemble Methods & Threshold Tuning
### ⏱️ Duration: 0:45 | Running Total: 8:30

> After training four models per task, we combine them using two ensemble strategies.
>
> **AUROC-Squared Weighted Averaging**: Each model's weight is its test AUROC squared, normalized. Squaring amplifies the gap between models. The ensemble probability is the weighted sum.
>
> **Stacking Meta-Learner**: A Logistic Regression trained on the four models' probability outputs learns which model to trust for which patterns. To prevent leakage, we split the test set in half — one half trains the meta-learner, the other evaluates.
>
> We also perform **per-task threshold tuning**: instead of the default 0.5, we sweep 100 thresholds and pick the one maximizing F1 on validation. For rare events like mortality, the optimal threshold drops to ~0.25, improving sensitivity from 0.31 to 0.68.

---

## SLIDE 9 — My Task 1: Sepsis Prediction
### ⏱️ Duration: 1:15 | Running Total: 9:45

> Now, my individual tasks. First: **sepsis prediction**.
>
> Sepsis is life-threatening organ dysfunction caused by a dysregulated immune response to infection. It's the **leading cause of ICU death**, and every hour of delayed treatment increases mortality by 4–8%.
>
> **How we define sepsis labels**: We use the clinically established SIRS-plus-infection definition.
>
> **SIRS criteria** — we score 0 to 4 at each timepoint:
> - Temperature > 38.3°C or < 36°C → +1
> - Heart rate > 90 bpm → +1
> - Respiratory rate > 20 → +1
> - WBC > 12 or < 4 K/µL → +1
>
> If SIRS ≥ 2 at ANY point in the prediction window, we look for **infection evidence**: antibiotic prescriptions — searching drug names for keywords like 'cillin', 'mycin', 'floxacin', 'vancomycin', 'meropenem' — **OR** ICD-9 sepsis codes: 038 for septicemia, 995.91 for sepsis, 995.92 for severe sepsis.
>
> Both conditions must be met. This dual requirement dramatically improves specificity over SIRS alone, which triggers false positives in post-surgical and trauma patients.
>
> **Results**: LightGBM achieved the best individual AUROC of **0.76**, with the stacked ensemble reaching **0.77**. Tree models outperformed deep learning here because the SIRS-plus-infection definition is fundamentally threshold-based — trees capture rules like "temperature above 38.3 AND WBC above 12" natively.

---

## SLIDE 10 — My Task 2: Vasopressor Prediction
### ⏱️ Duration: 1:00 | Running Total: 10:45

> **Vasopressors** are emergency IV drugs that constrict blood vessels to raise critically low blood pressure. They're used in septic shock, cardiogenic shock, and hemodynamic emergencies. Starting them requires central line access and one-to-one nursing.
>
> **Label generation from two data sources**:
>
> **Source 1 — PRESCRIPTIONS**: Drug names containing vasopressor keywords — norepinephrine, epinephrine, vasopressin, dopamine, dobutamine, phenylephrine, milrinone.
>
> **Source 2 — INPUTEVENTS_MV**: Specific item IDs for IV vasopressor infusions. Prescriptions capture the doctor's *order*; INPUTEVENTS captures the actual *administration*. Either event confirms vasopressor use.
>
> We predict at **6 and 12 hours only** — not 24. Hemodynamic status changes rapidly, making a 24-hour prediction clinically unactionable.
>
> **Results**: LightGBM achieved the best individual AUROC of **0.80**, with the weighted ensemble reaching **0.81**. Tree models dominated this task because vasopressor need correlates with explicit thresholds: MAP below 65, lactate above 4, shock index above 1.0.

---

## SLIDE 11 — My Task 3: Ventilation Prediction
### ⏱️ Duration: 1:00 | Running Total: 11:45

> **Mechanical ventilation** means a machine breathes for the patient — required for respiratory failure, airway protection, or post-operative recovery. Intubation is invasive and high-risk.
>
> **Label generation uses three data sources** — the most comprehensive detection in our pipeline:
>
> **Source 1 — CHARTEVENTS**: Ventilation-related chart entries — item IDs 225792, 225794, and 226260.
>
> **Source 2 — PROCEDUREEVENTS_MV**: Same item IDs in the procedure events table.
>
> **Source 3 — PROCEDURES_ICD**: ICD-9 codes — 96.70 through 96.72 for continuous mechanical ventilation, 96.04 for intubation, 93.90 for non-invasive ventilation.
>
> No single MIMIC-III table reliably captures ALL ventilation events. Using all three with OR-logic **maximizes sensitivity**.
>
> We predict at **6, 12, and 24 hours**: 6h for equipment pre-positioning, 12h for staffing — ventilated patients need 1:1 nursing, and 24h for bed management.
>
> **Results**: LightGBM achieved **0.85 AUROC**, with the stacked ensemble also at **0.85** — the strongest individual-model result in the project. Tree models excelled because ventilation correlates with threshold-based respiratory indicators — SpO₂ below 90%, respiratory rate above 24.

---

## SLIDE 12 — Engineering Challenges
### ⏱️ Duration: 0:45 | Running Total: 12:30

> Five key engineering challenges we solved:
>
> **Memory**: CHARTEVENTS at 33.6 GB — solved with chunked loading, filtering to ~15% of relevant rows.
>
> **FP16 NaN Loss**: Three separate causes. Unnormalized features exceeding FP16 range — solved by StandardScaler. Transformer attention overflow — solved by Pre-LayerNorm plus FP32 attention. Sigmoid saturation in BCELoss — solved by switching to BCEWithLogitsLoss.
>
> **TCN Collapse**: BatchNorm1d collapsed under FP16 with imbalanced labels — variance in rare-event batches dropped to zero. We removed TCN entirely.
>
> **Windows Compatibility**: PyTorch DataLoader multiprocessing crashes without a 'spawn' guard on Windows. We auto-detect and force `num_workers=0`.
>
> **Temperature Unit Mixing**: MIMIC-III stores temperatures in BOTH Fahrenheit and Celsius. Without detection and conversion, 98.6°F appears as a 98.6°C fever to the model.

---

## SLIDE 13 — Web Dashboard & Clinical Decision Support
### ⏱️ Duration: 0:45 | Running Total: 13:15

> The system is deployed through a **FastAPI dashboard** with five views: Overview with KPI tiles, Patient List filterable by risk, Patient Detail with vital charts and SHAP explanations, New Assessment for manual entry, and Clinical Alerts for critical thresholds.
>
> What makes our dashboard clinically intelligent is **care-unit-adaptive risk scoring**. Not all ICU patients are the same — a **MICU** patient's biggest threat is sepsis, while a **CCU** patient's biggest threat is hemodynamic collapse. So the composite risk score uses **different weights for different care units**:
>
> - **MICU** (Medical ICU): Mortality 30%, Sepsis 30%, AKI 15%, Vasopressor 10%, Ventilation 15%
> - **SICU** (Surgical ICU): Mortality 25%, Sepsis 20%, AKI 20%, Vasopressor 20%, Ventilation 15%
> - **CCU** (Cardiac Care): Mortality 35%, Sepsis 15%, AKI 15%, Vasopressor 25%, Ventilation 10%
> - **CSRU** (Cardiac Surgery Recovery): Mortality 25%, Sepsis 15%, AKI 25%, Vasopressor 15%, Ventilation 20%
> - **TSICU** (Trauma/Surgical): Mortality 25%, Sepsis 25%, AKI 15%, Vasopressor 20%, Ventilation 15%
>
> For **normal ward patients**, the system behaves differently. Vasopressor and ventilation weights are halved — these are ICU-specific interventions unlikely for ward patients. Instead, mortality and sepsis weights are elevated to 35% each, focusing on **early deterioration detection**. Ward patients whose composite score exceeds **0.45** trigger a three-tier ICU transfer escalation: **CONSIDER** (0.45–0.55), **URGENT** (0.55–0.70), and **IMMEDIATE** (above 0.70).
>
> We also integrated **Google Gemini 2.0 Flash** for two GenAI modes: **Risk Interpretation** — vitals plus predictions generate a structured clinical summary with findings and next steps — and **SBAR Handoff** — auto-generated Situation-Background-Assessment-Recommendation notes for shift changes. If no API key is available, a template-based fallback ensures the dashboard always works.

---

## SLIDE 14 — Results Summary
### ⏱️ Duration: 0:45 | Running Total: 14:00

> Here is the results summary for all six tasks:
>
> | Task | Best Model | AUROC | Ensemble AUROC |
> |------|-----------|-------|---------------|
> | **Sepsis (24h)** | LightGBM | 0.76 | 0.77 |
> | **Vasopressor (12h)** | LightGBM | 0.80 | 0.81 |
> | **Ventilation (24h)** | LightGBM | 0.85 | 0.85 |
> | Mortality (24h) | LightGBM | 0.77 | 0.78 |
> | AKI Stage 1 (24h) | LightGBM | 0.85 | 0.85 |
> | LOS Short (<24h) | LightGBM | 0.79 | 0.79 |
>
> Three key takeaways:
> - Tree-based models — particularly **LightGBM** — outperformed deep learning across all tasks. The likely reason: 24 timesteps is insufficient for self-attention to outperform tree-based threshold learning, and clinical definitions are inherently threshold-based.
> - Ensembles improved AUROC by **0.01–0.02** across tasks via both weighted averaging and stacking.
> - We cover **six clinical tasks** with **17 binary labels** using **four model architectures** — a broader scope than the Harutyunyan et al. 2019 benchmark which covered four tasks with one model.

---

## SLIDE 15 — Conclusion & Future Work
### ⏱️ Duration: 1:00 | Running Total: 15:00

> To conclude: we built a **complete end-to-end system** — the **ICU Sentinel** — that:
> 1. Processes **43.6 GB** of real ICU data from **17 relational tables**
> 2. Engineers **81 features per timestep** using clinical domain knowledge
> 3. Generates **17 binary labels** across **6 clinical prediction tasks**
> 4. Trains **4 model architectures** per task with GPU-optimized mixed-precision training
> 5. Combines them via **ensemble methods** for consistently superior performance
> 6. Serves predictions through a **real-time web dashboard** with clinical alerts and care-unit-adaptive risk scoring
>
> **My individual contribution** — sepsis, vasopressor, and ventilation prediction — achieved ensemble AUROCs of **0.77, 0.81, and 0.85** respectively, using clinically validated label definitions with multi-source data verification.
>
> **Future work** includes:
> - **External validation** on eICU, MIMIC-IV, and HiRID datasets
> - **Hyperparameter optimization** with Optuna or Ray Tune
> - **Probability calibration** using Platt scaling
> - **Clinical notes integration** via ClinicalBERT
> - **Live EHR integration** through HL7 FHIR
> - And ultimately — a **prospective clinical trial** to measure whether predictions actually improve patient outcomes
>
> Thank you for your attention. I am now happy to take your questions.

---

---

# 📋 Q&A SESSION — Prepared Answers (5 Minutes)

> [!NOTE]
> **Strategy for Q&A**: Listen to the FULL question. Pause 2–3 seconds before answering (shows you're thinking, not reciting). Keep answers to 30–45 seconds max. If you don't know something, say "That's a great question — based on our current implementation, [what you do know], and [honest limitation]."

---

### Q1: Why did you choose MIMIC-III over MIMIC-IV?

> MIMIC-III has more established benchmarks in the literature — particularly Harutyunyan et al. 2019, which provides direct AUROC comparison points on the same dataset. MIMIC-IV was released more recently and has fewer published baselines for multi-task ICU prediction. However, validating on MIMIC-IV is part of our future work.

---

### Q2: Why 4 models instead of just the best one?

> Different architectures have different **inductive biases**. In practice, tree-based models — LightGBM and XGBoost — dominated all six tasks because clinical definitions are fundamentally threshold-based and 24 timesteps is likely too short for self-attention to outperform trees. However, the deep learning models still contributed to ensemble diversity and improved the final ensemble AUROC by 0.01–0.02.

---

### Q3: Why not use a CNN or TCN?

> We actually tried TCN initially. But BatchNorm1d collapsed under FP16 mixed-precision training with imbalanced labels. The variance in rare-event batches dropped to zero, producing NaN loss from epoch 2 onwards across every task. After extensive debugging, we removed it and found the remaining four models provided sufficient architectural diversity.

---

### Q4: How do you handle the class imbalance?

> Three mechanisms: First, per-task `pos_weight` in BCEWithLogitsLoss — it upweights positive examples by the N_negative/N_positive ratio. Second, per-task threshold tuning on validation F1 instead of the hardcoded 0.5 — for rare events, the optimal threshold drops to around 0.15–0.25. Third, both XGBoost and LightGBM have their own `scale_pos_weight` parameter set per task. Without all three, the model learns to predict "no" every time — 97.6% accuracy but zero sensitivity.

---

### Q5: What is temporal leakage and how did you prevent it?

> Temporal leakage occurs when a model trained on data from a later time period is tested on data from an earlier time period — effectively "knowing the future." We prevent this by sorting all data **chronologically by ICU admission time** and splitting: first 70% for training, next 15% for validation, last 15% for testing. No shuffling whatsoever.

---

### Q6: Why is the sepsis SIRS threshold ≥ 2 and not higher?

> SIRS ≥ 2 is the internationally accepted clinical definition from the American College of Chest Physicians. While SIRS alone has low specificity — post-surgical and trauma patients can trigger it — we combine it with infection evidence: antibiotic prescription OR ICD-9 sepsis codes. This dual requirement dramatically reduces false positives while maintaining clinically meaningful labels.

---

### Q7: Why does the Transformer need a 10× lower learning rate?

> Transformers have more parameters in their attention layers. At the LSTM learning rate of 0.001, the attention weight updates are too aggressive, causing the Q·K-transpose dot products to overflow the FP16 maximum of 65,504 by around epoch 5. The 10× lower rate of 0.0001 keeps these values bounded and training stable.

---

### Q8: What is the clinical significance of shock index?

> Shock index equals heart rate divided by systolic blood pressure. A normal range is 0.5–0.7. Above 0.7 is abnormal, above 1.0 is concerning. It captures the compensatory relationship: when blood pressure drops, heart rate rises to maintain cardiac output. A high shock index means this compensation is failing — the patient is in or approaching shock. It's clinically validated as a predictor of hemodynamic instability.

---

### Q9: How does the attention mechanism help in sepsis prediction?

> Without attention, the LSTM uses only the last hidden state — the most recent hour. But the critical SIRS event — a temperature spike or WBC surge — might have occurred 8 hours before the prediction point. Attention assigns learned weights to each timestep, allowing the model to focus on the clinically relevant hour regardless of when it occurred in the 24-hour sequence. This is why BiLSTM-Attention outperformed the plain architecture for temporal tasks.

---

### Q10: Why StandardScaler instead of MinMaxScaler?

> StandardScaler produces zero-mean, unit-variance features — ideal for neural networks. MinMaxScaler compresses outliers and is distorted by extreme values common in ICU data — a heart rate of 220 during a cardiac arrest, for instance. More critically, without StandardScaler, raw feature values can exceed the FP16 range of ±65,504, causing training to produce NaN losses.

---

### Q11: What are the limitations of your system?

> We're transparent about five limitations: (1) No live validation — this is entirely retrospective. (2) Single hospital data — MIMIC-III is from Beth Israel Deaconess only; generalizability is unknown. (3) MIMIC-III covers 2001–2012 — clinical practice has evolved. (4) We don't use unstructured data — no clinical notes, waveforms, or imaging. (5) No regulatory approval. This is a research prototype — clinical deployment requires prospective validation.

---

### Q12: How does the composite risk score work?

> It's a weighted sum of five task predictions. The weights are **care-unit-adaptive** — different ICU types weight the tasks differently. In MICU, sepsis gets 30% weight because it's the primary concern. In CCU, vasopressor gets 25% because hemodynamic instability is the focus. Patients are classified as HIGH (>0.6), MEDIUM (0.3–0.6), or LOW (≤0.3) risk. Ward patients above 0.45 trigger ICU transfer flags with escalation levels: CONSIDER, URGENT, and IMMEDIATE.

---

### Q13: Why did you use binary classification for LOS instead of regression?

> Because that's how the clinical decision actually gets made. A doctor doesn't ask "will this patient stay 73.2 hours?" — they ask "is this patient going home tomorrow?" or "are we looking at a long stay?" Binary framing at the 24-hour and 72-hour thresholds maps directly to discharge planning rounds and care pathway decisions. Our short-stay predictor achieved F1 of 0.60, which is directly useful.

---

### Q14: How does the stacking meta-learner avoid data leakage?

> We split the test set in half. The first half is used to generate predictions from all four base models, and these predictions become the training features for the Logistic Regression meta-learner. The second half is a completely held-out evaluation set. This ensures the meta-learner never sees data it will be evaluated on.

---

### Q15: What would you do differently if you started over?

> Three things: (1) Start with Optuna for hyperparameter search from day one instead of manual tuning. (2) Implement probability calibration — our probabilities aren't calibrated, so a prediction of 0.7 doesn't necessarily mean 70% true risk. (3) Add clinical notes via ClinicalBERT — notes contain context that vitals and labs simply cannot capture, like "patient appears anxious" or "family requests comfort measures only."

---

> [!TIP]
> **Final Reminders for Fadhil:**
> - Arrive 15 minutes early (by 9:45 AM)
> - Have laptop charged and presentation loaded
> - Keep a backup on USB drive
> - Dress formally
> - Stay calm during Q&A — it's okay to take a moment to think
> - If a question is about Raha's tasks (Mortality, AKI, LOS), you can briefly answer from a system perspective but note "this was my teammate Raha's individual task"
> - **You know this system inside and out. You built it. Be confident.**
