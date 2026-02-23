# Regime-Aware Bayesian Deferral for Seismic Phase Picking in Data-Sparse Regions

## A Technical Reference for ELEC70122: Machine Learning for Safety Critical Decision-Making

---

## 1. Executive Summary

### The Problem

Machine learning models trained in data-rich environments fail silently when deployed in data-sparse environments. In seismology, this is not an abstract concern — it is an equity problem with life-or-death consequences.

PhaseNet (Zhu & Beroza, 2019) is a deep learning model trained on over 700,000 seismograms from Northern California's dense monitoring network (889+ stations, decades of high-quality data, strike-slip tectonic regime). It has become a standard tool for automated seismic phase picking — the task of identifying the exact arrival times of earthquake waves at recording stations. These arrival times are the foundational input to earthquake location algorithms, which in turn feed hazard maps, building codes, and emergency response plans.

When PhaseNet is applied to Kazakhstan — a seismically active region with only 13 monitoring stations, intraplate tectonics, different geological structures, and fundamentally different waveform characteristics — it produces picks that look plausible but may be wrong. The model has no mechanism to signal that it is operating outside its training distribution. The errors cascade: wrong picks produce wrong earthquake locations, wrong locations produce wrong hazard maps, and wrong hazard maps produce building codes that may fail to protect people.

This cascading failure falls disproportionately on developing countries. Regions like California, Japan, and Western Europe have dense networks and abundant expertise to catch model errors. Kazakhstan, Central Asia, and much of the developing world do not. The very regions that would benefit most from automated seismology are the regions where automation is least reliable.

### The Insight

The core insight of this project is that the solution is not to make PhaseNet more accurate in Kazakhstan — that would require retraining on data that does not exist. Instead, the solution is to build a system that **knows when it does not know**. When PhaseNet is confident and likely correct, trust it. When it is uncertain or likely wrong, defer the decision to a human seismologist who can apply domain expertise that the model lacks.

This is a **learning-to-defer** problem: the system must learn a policy for when to use the automated model and when to hand control to a human expert.

### The Complication: Non-Stationarity

A naive deferral approach would set a fixed confidence threshold — defer whenever PhaseNet's output probability is below some cutoff. But seismic environments are non-stationary. After a significant earthquake, the region enters an aftershock sequence where waveforms become more complex, stations may be temporarily affected, and the local stress field changes. PhaseNet's reliability in this post-event regime is different from its reliability during quiet background seismicity. A fixed threshold cannot capture this.

This is where the project connects directly to the work of Prof. Sonali Parbhoo and colleagues. In Joshi, Parbhoo & Doshi-Velez (2022), the Sequential Learning-to-Defer (SLTD) framework addresses exactly this problem in healthcare: a clinical decision support system must decide, at each time step, whether to follow its own recommendation or defer to a clinician — and this decision must account for the fact that the patient's condition (and therefore the model's reliability) changes over time. The environment is modelled as a sequence of MDPs {M_t} with non-stationary dynamics, and the deferral policy is optimised using Bayesian model-based reinforcement learning.

### Our Approach

We adapt the conceptual architecture of SLTD to seismology using tractable Bayesian machinery:

1. **Regime Model**: We model the seismic environment as transitioning between discrete regimes (Quiet, Active, Decaying) using a Hidden Markov Model, mirroring SLTD's sequence of MDPs {M_t}.

2. **Regime-Specific Reliability Tracking**: We maintain separate Beta-Bernoulli posteriors over PhaseNet's reliability in each regime, analogous to SLTD's posterior estimation over dynamics using Dirichlet distributions.

3. **Regime-Aware Deferral**: The deferral decision combines the current regime belief (from the HMM) with the regime-specific reliability posterior, producing a principled, time-varying deferral policy rather than a static threshold.

This framework is a deliberate simplification of the full SLTD approach: we make myopic (one-step) deferral decisions rather than optimising for long-term outcomes via dynamic programming. We acknowledge this limitation explicitly and discuss the extension to full sequential optimisation as future work. The simplification is justified by the MSc project scope and by the fact that implementing the regime-aware Bayesian structure already represents a meaningful advance over static deferral thresholds.

### Connection to the Module

ELEC70122 focuses on machine learning for safety-critical decision-making — applications where incorrect predictions have serious consequences. This project engages with several core module themes:

- **Uncertainty quantification**: We compare multiple UQ methods (native probabilities, MC Dropout, Bootstrap Ensemble) and show that standard UQ breaks down under distribution shift.
- **Bayesian modelling**: The entire deferral framework is built on Bayesian inference — Beta-Bernoulli conjugate models, Hidden Markov Models, and posterior updating.
- **Safety through deferral**: Rather than trying to eliminate model errors, we build a system that manages them by routing uncertain decisions to human experts.
- **Non-stationarity**: Following Prof. Parbhoo's SLTD framework, we explicitly model and account for time-varying dynamics in the deferral policy.
- **Real-world impact**: The equity dimension — automated tools amplifying rather than reducing inequality between data-rich and data-sparse regions — grounds the technical work in genuine societal consequence.

---

## 2. PhaseNet and Seismic Phase Picking

### Background

When an earthquake occurs, it radiates seismic waves outward from the source. Two types of body waves are of primary interest: **P-waves** (primary/compressional waves, faster, arrive first) and **S-waves** (secondary/shear waves, slower, arrive second). The task of **phase picking** is to identify the exact time at which each wave type arrives at a seismometer station. These arrival times, combined across multiple stations, are used to triangulate the earthquake's location through a process called seismic inversion.

Traditionally, phase picking is performed by human analysts who visually inspect seismograms — the time-series recordings of ground motion. An experienced analyst identifies the subtle onset of a P-wave (the first deviation from background noise) and the onset of the S-wave (a change in the wave character, typically with larger amplitude and lower frequency). This is labour-intensive: a single analyst can process perhaps a few hundred events per day, and global seismic networks record thousands of events daily.

### PhaseNet Architecture

PhaseNet (Zhu & Beroza, 2019) is a 1D U-Net — a convolutional neural network with an encoder-decoder architecture and skip connections. Its design is adapted from the U-Net originally developed for biomedical image segmentation (Ronneberger et al., 2015).

**Input**: A 3-channel time series of ground motion — three components (vertical Z, north-south N, east-west E) recorded simultaneously at a single station. Each channel is a sequence of numbers representing ground velocity sampled at a fixed rate (typically 100 Hz). A standard input window is 60 seconds, giving 6000 samples per channel — so the input tensor has shape (3, 6000).

**Output**: Three probability traces of the same length as the input — P(P-wave onset), P(S-wave onset), and P(Noise) for each time sample. At most time steps, the noise probability dominates. At the moments where P and S waves arrive, the corresponding probability spikes.

**Pick extraction**: Discrete picks are extracted from the probability traces by finding peaks above a threshold (default 0.3) with a minimum separation. Each pick has an associated confidence score — the peak probability value.

**Training data**: The "original" pre-trained model was trained on the Northern California Seismic Network (NCSN) dataset — approximately 780,000 waveform examples from Northern California, with analyst-reviewed phase picks as ground truth labels. This training data comes from a specific tectonic environment (transform plate boundary, strike-slip faulting), a dense station network, modern broadband instruments, and relatively low ambient noise conditions.

### Why PhaseNet's Confidence Is Not Calibrated Under Distribution Shift

PhaseNet outputs a probability for each pick, but this probability is not a reliable indicator of correctness when the model encounters waveforms that differ from its training distribution. The model was trained to minimise cross-entropy loss on California data, so its probability outputs are calibrated (approximately) for California-like waveforms. When it encounters waveforms from Kazakhstan — which have different frequency content, different noise characteristics, different source mechanisms, and are recorded on different instruments — the probabilities lose their calibration.

Concretely: a PhaseNet pick with confidence 0.6 in California might be correct 60% of the time. A pick with confidence 0.6 in Kazakhstan might be correct only 30% of the time, or 80% of the time — the model has no basis for knowing, because it has never seen similar waveforms during training.

This is the fundamental problem that motivates the entire project. The model's own uncertainty estimates cannot be trusted under distribution shift, so we need an external framework — the regime-aware Bayesian deferral system — to assess and track reliability.

### Key References for This Section

- Zhu, W. & Beroza, G.C. (2019). PhaseNet: A deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261–273.
- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
- Woollam, J. et al. (2022). SeisBench — A toolbox for machine learning in seismology. *Seismological Research Letters*, 93(3), 1695–1709.

---

## 3. Distribution Shift: California to Kazakhstan

### Formalising the Problem

Let X denote the space of input waveforms and Y the space of output picks (arrival times). PhaseNet has learned a mapping f: X → Y by training on samples drawn from a California distribution P_CA(X, Y). When we deploy it in Kazakhstan, the data is drawn from a different distribution P_KZ(X, Y).

The distribution shift between P_CA and P_KZ has several components:

**Covariate shift** (P(X) changes): The waveforms themselves look different. Kazakhstan's intraplate tectonic setting produces different source mechanisms, different wave propagation paths through different geological structures, and different ambient noise characteristics compared to California's transform boundary.

**Concept shift** (P(Y|X) changes): Even for similar-looking waveforms, the relationship between waveform features and correct pick times may differ, because the velocity structure of the Earth beneath Kazakhstan is different from California.

**Instrument shift**: Kazakhstan's stations use different sensors (STS-5, STS-6 at IRIS stations; various sensors at KZ network stations) with different response characteristics, different noise floors, and different sampling rates (40 Hz at some Kazakhstan stations vs. 100 Hz for much of the California training data).

### Quantifying the Shift

There are several ways to measure the distribution shift, and you should use at least one in your analysis:

**Waveform feature comparison**: Extract summary statistics from waveforms in both regions — signal-to-noise ratio (SNR), dominant frequency, spectral content, amplitude distributions — and compare. This gives an intuitive picture of how different the inputs look.

**PhaseNet confidence distributions**: Compare the distribution of PhaseNet's output confidence scores between California and Kazakhstan. If the model is encountering unfamiliar inputs, confidence scores will typically shift downward and spread out. This is the easiest comparison to implement and directly relevant to your deferral framework.

**Pick residuals**: Where you have ground truth (ISC-reviewed picks), compute the time difference between PhaseNet's pick and the human pick. The distribution of these residuals in California (tight, centered near zero) versus Kazakhstan (wider, possibly biased) directly quantifies performance degradation.

### Why This Matters for Safety

In California, PhaseNet achieves P-wave pick accuracy within ±0.1 seconds of human analysts (Zhu & Beroza, 2019). Even a modest degradation to ±0.5 seconds in Kazakhstan translates to kilometres of error in earthquake location (the exact amount depends on the station geometry and the velocity model). For a sparse network like Kazakhstan's 13 stations, where each pick carries more weight in the location algorithm (less redundancy), this error is amplified further.

This is the concrete pathway from distribution shift to real-world harm: shifted distributions → degraded picks → wrong locations → wrong hazard assessments → inadequate building codes.

### The Kazakhstan Network

Kazakhstan's seismic monitoring is operated by the Institute of Geophysical Research (IGR) through the Kazakhstan National Data Centre (KNDC) in Almaty. The network includes:

**Seismic arrays** (clusters of sensors): MKAR (Makanchi), KKAR (Karatau), ABKAR (Akbulak), KURK (Kurchatov-Cross), BVAR (Borovoye) — each with approximately 10 elements.

**Three-component stations**: AKTO (Aktyubinsk), BORK (Borovoye, IRIS/IDA), KURK (Kurchatov, IRIS/IDA), MAKZ (Makanchi, IRIS/GSN), OTUK (Ortau), PDGK (Podgornoye), KASK (Kaskelen), KNDC (Almaty).

The three IRIS stations (BORK, KURK, MAKZ) are the primary targets for this project — their waveforms are openly available through the IRIS FDSN web service, and they use modern broadband instruments (upgraded in 2019). The KZ network stations are also accessible through IRIS and provide additional coverage.

For contrast: the Northern California Seismic Network has 889+ stations. Kazakhstan's entire national network has 13 stations covering a country roughly four times the area of Texas.

### Key References for This Section

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227–244.
- Mikhailova, N.N. & Sokolova, I.N. (2019). Monitoring System of the Institute of Geophysical Research of the Ministry of Energy of the Republic of Kazakhstan. *Summary of the Bulletin of the ISC*, 53(1), 27–38.

---

## 4. Uncertainty Quantification Methods

### Why Standard UQ Breaks Down

Before building the deferral framework, we need to understand what information PhaseNet gives us about its own uncertainty — and why that information is insufficient under distribution shift.

We compare three approaches to uncertainty quantification, each with different assumptions and failure modes:

### Method 1: Native Probabilities (Softmax Output)

**What it is**: PhaseNet already outputs probability values for each time sample. The peak probability at a detected pick is a natural confidence score.

**The maths**: For a given time step t, PhaseNet outputs logits z_P(t), z_S(t), z_N(t) and applies softmax:

P(class = P | x, t) = exp(z_P(t)) / [exp(z_P(t)) + exp(z_S(t)) + exp(z_N(t))]

The peak value of this probability trace at a detected pick serves as the confidence.

**Why it fails under distribution shift**: Softmax probabilities are calibrated to the training distribution. On out-of-distribution inputs, neural networks are known to produce arbitrarily confident predictions that are wrong (Guo et al., 2017). A PhaseNet confidence of 0.8 on a Kazakhstan waveform does not mean the same thing as 0.8 on a California waveform.

**What it is useful for**: Despite poor calibration, relative ordering may still be informative — picks with higher confidence may still tend to be more accurate than picks with lower confidence, even if the absolute probabilities are meaningless. This is worth testing empirically.

### Method 2: MC Dropout

**What it is**: At inference time, keep dropout layers active and run the same input through the network T times. Each forward pass produces a slightly different output due to the random dropout mask. The variance across these T outputs estimates the model's epistemic uncertainty.

**The maths**: For a single pick, perform T stochastic forward passes to obtain T pick times {t̂_1, ..., t̂_T} and T confidence scores {p_1, ..., p_T}. The uncertainty estimates are:

- **Pick time uncertainty**: σ_t = std({t̂_1, ..., t̂_T}) — how much the predicted arrival time varies across passes.
- **Confidence uncertainty**: σ_p = std({p_1, ..., p_T}) — how much the confidence varies.
- **Mean confidence**: p̄ = mean({p_1, ..., p_T}).

A pick where σ_t is large (the model gives different arrival times each pass) or σ_p is large (the model is sometimes confident and sometimes not) is a pick the model is uncertain about.

**Why it partially helps under distribution shift**: MC Dropout captures epistemic uncertainty — uncertainty due to lack of knowledge, which is reducible with more data. Out-of-distribution inputs tend to produce higher epistemic uncertainty because the model's learned representations are less stable for unfamiliar patterns. So MC Dropout will often (but not always) flag distribution-shifted inputs as uncertain.

**Limitations**: MC Dropout is an approximate Bayesian method (Gal & Ghahramani, 2016) and its uncertainty estimates are not guaranteed to be well-calibrated. The quality of the uncertainty estimate depends on the dropout architecture and rate, which were chosen for California performance, not Kazakhstan uncertainty estimation. It also requires T forward passes per input, increasing computational cost by a factor of T (typically T = 20–50).

**Key reference**: Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML 2016*.

### Method 3: Bootstrap Ensemble

**What it is**: Train multiple instances of PhaseNet (or fine-tune the pre-trained model) on different bootstrap samples of the data. At inference time, run the input through all ensemble members and measure disagreement.

**The maths**: Given K ensemble members, obtain K pick times {t̂_1, ..., t̂_K} and K confidences {p_1, ..., p_K}. Uncertainty is measured by:

- **Pick time spread**: σ_t = std({t̂_1, ..., t̂_K})
- **Confidence spread**: σ_p = std({p_1, ..., p_K})
- **Prediction disagreement**: the fraction of ensemble members that disagree on whether a pick exists at all.

**Why it is useful**: Ensemble disagreement is one of the most robust indicators of out-of-distribution inputs (Lakshminarayanan et al., 2017). If the ensemble members agree on California data but disagree on Kazakhstan data, this directly measures the effect of distribution shift on model predictions.

**Practical constraint**: Training multiple PhaseNet instances from scratch requires substantial compute. A more practical approach for this project is to use the single pre-trained model with MC Dropout as a pseudo-ensemble — this approximates ensemble disagreement without the training cost.

**Key reference**: Lakshminarayanan, B., Pritzel, A. & Garnett, R. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.

### Comparing the Methods

The key experiment in this stage is to evaluate all three UQ methods on both California (in-distribution) and Kazakhstan (out-of-distribution) data, and measure:

1. **Calibration**: Are the uncertainty estimates well-calibrated? When the model says it is 80% confident, is it correct 80% of the time? Use reliability diagrams and Expected Calibration Error (ECE).

2. **Discrimination**: Do the uncertainty estimates separate correct picks from incorrect picks? Use AUROC (Area Under the Receiver Operating Characteristic curve) treating "pick is correct" as the positive class and the uncertainty estimate as the score.

3. **Shift detection**: Do the uncertainty estimates increase systematically from California to Kazakhstan? Plot the distributions side by side.

The hypothesis is that all three methods will be reasonably well-calibrated on California but poorly calibrated on Kazakhstan, with MC Dropout and Bootstrap showing better shift detection than native probabilities. The gap between in-distribution calibration and out-of-distribution calibration is the quantitative evidence that motivates the external deferral framework.

### Key References for This Section

- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML 2017*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML 2016*.
- Lakshminarayanan, B., Pritzel, A. & Garnett, R. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
- Naeini, M.P., Cooper, G.F. & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning into quantiles. *AAAI 2015*.

---

## 5. The Seismic Regime Model

### Motivation

Seismic environments are non-stationary. After a significant earthquake, the region enters a fundamentally different state — aftershock sequences change the rate and character of seismic activity, the local stress field is redistributed, and the waveforms recorded at nearby stations change in character. This non-stationarity means that PhaseNet's reliability is not constant over time — it depends on the current seismic state of the region.

This directly mirrors the non-stationarity problem in SLTD (Joshi et al., 2022), where a patient's disease dynamics change over time, and the ML model's reliability changes with them. SLTD models this as a sequence of MDPs {M_t}, where each M_t has its own dynamics. We adapt this idea to seismology using a Hidden Markov Model over seismic regimes.

### Regime Definitions

We define three seismic regimes:

**Regime Q (Quiet)**: Background seismicity. The region is experiencing its typical level of low-level seismic activity. Waveforms are relatively standard — dominated by background noise with occasional clear regional events. PhaseNet is expected to perform at its best (for this region) during quiet periods, because the waveforms, while different from California, are at least relatively consistent and predictable.

**Regime A (Active)**: A significant earthquake (M ≥ 5.0 within 200 km of the monitoring station) has occurred. The region is producing mainshock and aftershock waveforms that are more complex: higher amplitude, potentially overlapping events, more complex source mechanisms. PhaseNet is expected to perform at its worst during active periods, because the waveforms are furthest from anything in the training data — complex, overlapping, with potentially clipped or saturated records.

**Regime D (Decaying)**: The aftershock sequence is winding down following Omori's Law (aftershock rates decay approximately as 1/t after the mainshock). Waveforms are intermediate — smaller than during the active phase but still different from background, with residual aftershock activity. PhaseNet's reliability is expected to be intermediate.

### The Markov Chain

The regimes evolve according to a first-order Markov chain (Assumption 2.1 from our framework). This means the probability of being in a given regime at time t depends only on the regime at time t-1, not on the full history:

P(R_t | R_{t-1}, R_{t-2}, ..., R_1) = P(R_t | R_{t-1})

The transition dynamics are captured by a 3×3 transition matrix T, where T_{ij} = P(R_t = j | R_{t-1} = i):

```
             To Q      To A      To D
From Q    [ 1 - q     q         0       ]
From A    [ 0         1 - a     a       ]
From D    [ d_Q       d_A       1-d_Q-d_A ]
```

The key structural constraints are:

- **Q → D is impossible** (probability 0): You cannot enter a decaying phase without first being active. An aftershock sequence requires a mainshock.
- **A → Q is impossible** (probability 0): An active sequence does not jump straight back to quiet — it must decay first. This is physically motivated by Omori's Law.
- **D can go anywhere**: From decaying, the sequence may return to quiet (aftershocks end), stay decaying (aftershocks continue at reduced rate), or reactivate (a large aftershock triggers renewed activity).

### Setting Transition Probabilities

The transition probabilities are parameters of the model that can be estimated from data or set from domain knowledge. For a monthly time window (see Section 10 for data details), reasonable starting values based on seismological understanding of aftershock statistics are:

- **q ≈ 0.05**: In any given month, there is roughly a 5% chance that a quiet region experiences a M ≥ 5.0 event. This can be refined using the historical earthquake rate for southern Kazakhstan from the ISC catalog.
- **a ≈ 0.30**: After the first month of an active sequence, there is roughly a 30% chance of transitioning to the decaying phase. Motivated by the observation that aftershock rates typically drop significantly within the first few weeks.
- **d_Q ≈ 0.20**: From the decaying phase, roughly 20% chance of returning to quiet in any given month. Aftershock sequences for M5-6 events typically last 1-6 months.
- **d_A ≈ 0.10**: Roughly 10% chance of reactivation from decaying (a large aftershock restarts the active phase).

These are starting values. In Section 5.1, we discuss how to estimate them from data.

### 5.1: Estimating Transition Probabilities from the Earthquake Catalog

Given a sequence of regime labels R_1, R_2, ..., R_T (one per time window), the maximum likelihood estimate of each transition probability is simply the empirical frequency:

T̂_{ij} = N_{ij} / N_i

where N_{ij} is the number of observed transitions from regime i to regime j, and N_i = Σ_j N_{ij} is the total number of transitions from regime i.

To label the regime sequence from the earthquake catalog:

1. Download the ISC event catalog for your geographic region (southern Kazakhstan, approximately 38°N–50°N, 65°E–85°E) for the period 2020–2023.
2. For each monthly time window, count the number of events and identify the maximum magnitude.
3. Apply the classification rule:
   - If a M ≥ 5.0 event occurred this month (or occurred in the previous month and aftershock rate remains above 2× background): **Active**.
   - If aftershock rate is between 1.2× and 2× the background rate: **Decaying**.
   - Otherwise: **Quiet**.
4. The background rate is estimated as the median monthly event count over the full period, excluding months classified as Active.

This produces a labelled regime sequence from which transition probabilities can be counted directly.

**Bayesian alternative**: Rather than point estimates, place a Dirichlet prior on each row of the transition matrix:

T_i ~ Dirichlet(α_{i,Q}, α_{i,A}, α_{i,D})

With symmetric α = 1 (uniform prior), the posterior after observing transitions is:

T_i | data ~ Dirichlet(1 + N_{iQ}, 1 + N_{iA}, 1 + N_{iD})

This gives you uncertainty over the transition probabilities themselves, which propagates into uncertainty over regime inference. This is a direct parallel to SLTD's use of Dirichlet posteriors over MDP transition dynamics.

### 5.2: Regime Inference via the Forward Algorithm

At any point in time, we do not directly observe which regime we are in — we observe seismic activity (event counts, magnitudes) and must infer the regime. This is a standard HMM inference problem.

Let **o_t** denote the observation at time t (e.g., the number of events and maximum magnitude in the time window). We need to compute:

P(R_t = r | o_1, o_2, ..., o_t)  for r ∈ {Q, A, D}

This is computed using the **forward algorithm**. Define:

α_t(r) = P(R_t = r, o_1, ..., o_t)

The recursion is:

**Initialisation**: α_1(r) = P(R_1 = r) × P(o_1 | R_1 = r)

**Recursion**: α_t(r) = P(o_t | R_t = r) × Σ_{r'} T_{r',r} × α_{t-1}(r')

**Normalisation**: P(R_t = r | o_1, ..., o_t) = α_t(r) / Σ_r α_t(r)

The **emission probabilities** P(o_t | R_t = r) model how likely the observed seismic activity is under each regime:

- **Under Quiet**: Event counts follow a Poisson distribution with low rate λ_Q (the background rate), and magnitudes are typically small (below M4).
- **Under Active**: Event counts follow a Poisson distribution with high rate λ_A >> λ_Q, and at least one event has magnitude ≥ 5.0.
- **Under Decaying**: Event counts follow a Poisson distribution with intermediate rate λ_D, decaying over time following Omori's Law.

For simplicity, you can use Poisson emission distributions:

P(o_t = n events | R_t = r) = Poisson(n; λ_r)

where λ_Q, λ_A, λ_D are estimated from the labelled training sequence.

### 5.3: Model Justification — Why an HMM?

An HMM is a suitable model for non-stationarity when the underlying process is **regime-switching and approximately piecewise stationary** — that is, the system alternates between distinct states, each with its own stable statistical properties, and transitions between states have temporal dependence. It is not a universal solution to distribution shift. This subsection makes the case for why an HMM is appropriate for our specific problem and explicitly acknowledges the forms of non-stationarity it cannot capture.

**Why the HMM fits seismic regime-switching.** The canonical seismic pattern — quiet background activity, followed by a mainshock and aftershock burst, followed by a decay phase — maps directly to the generative model an HMM assumes. Each regime (Quiet, Active, Decaying) persists for an extended interval (weeks to months, not seconds). Each regime has distinct, internally consistent statistical properties: Quiet periods have low event rates and standard waveforms; Active periods have high event rates, complex overlapping waveforms, and potentially saturated signals; Decaying periods have intermediate rates declining over time. The transitions between regimes are driven by discrete physical events (a M5+ earthquake triggers Q→A; aftershock rate decay triggers A→D; return to background triggers D→Q). These transitions exhibit temporal dependence — the probability of entering an Active regime depends on the current state, not on events from years ago. This is precisely the structure an HMM is designed to model.

**What the HMM does not capture.** There are forms of non-stationarity where an HMM would be the wrong tool:

*Gradual distributional drift*: If the data distribution evolves smoothly and continuously — for example, through sensor degradation, seasonal noise changes, or slow tectonic deformation — there are no discrete regimes to switch between, and the piecewise-stationary assumption breaks down. In our study, we control for sensor effects by restricting the analysis to 2020–2023, after the 2019 instrument upgrades at BORK and KURK — the instruments are stable across the entire study window. Seasonal noise variations exist (groundwater levels, temperature effects on electronics) but operate at much lower amplitude than regime-level changes; the difference in waveform statistics between a quiet period and an active aftershock sequence dwarfs the difference between summer and winter quiet periods. Tectonic deformation operates on timescales of decades to centuries and is negligible over a 4-year window. We propose an empirical verification of this claim: comparing waveform summary statistics (SNR, dominant frequency) across quiet-period windows in different seasons and years, and showing that the within-regime variation is small relative to the between-regime variation.

*High-frequency volatility*: If the system exhibits rapid, unpredictable switching — microseismic bursts, rapid swarm activity with no clear mainshock-aftershock structure — the discrete regime model may not capture the dynamics well. In southern Kazakhstan's Tien Shan region, the dominant mode of seismicity is mainshock-aftershock sequences (characteristic of intraplate settings with relatively simple stress regimes), not swarm activity. This makes the regime-switching model a natural fit for this specific study region. We acknowledge that extending the framework to swarm-dominated regions (e.g., volcanic areas) would require a different modelling approach.

*Non-Markov dynamics*: The first-order Markov assumption (A2.1) states that the current regime depends only on the previous regime, not the full history. The physical justification is that PhaseNet's reliability depends on the **current waveform characteristics**, which are determined by the current seismic state — not by how the region arrived at that state. A quiet period produces similar waveforms regardless of whether it follows a major aftershock sequence or a prolonged calm. An active period has similar waveform complexity whether it is the first M5 event in years or the second in six months. What matters for the deferral decision is what PhaseNet sees *right now*, and that is determined by the current regime.

The honest limitation is that this is not perfectly true. A region that has experienced multiple recent large earthquakes may have a more complex residual stress field, leading to subtly different waveform patterns even during nominally "quiet" periods — the quiet after a storm is not identical to a quiet that was never disturbed. A second-order model P(R_t | R_{t-1}, R_{t-2}) could capture this kind of history dependence. We adopt the first-order assumption as a tractable starting point that is standard in HMM applications and consistent with how SLTD estimates dynamics from recent data. Testing higher-order Markov dependence is a natural extension for future work, and the Beta-Bernoulli reliability tracking provides a partial safeguard: even if the Markov assumption causes occasional regime misclassification, the posterior over reliability will adapt as new pick outcomes are observed.

### Connection to SLTD

The regime model maps directly to SLTD's framework:

| SLTD Concept | Our Implementation |
|---|---|
| Sequence of MDPs {M_t} | Sequence of regimes {R_t} ∈ {Q, A, D} |
| Non-stationary dynamics | Regime transitions governed by Markov chain |
| Posterior over dynamics P(M_t) | Posterior over regime P(R_t) via forward algorithm |
| Dirichlet prior on transitions | Dirichlet prior on each row of T |

The key simplification relative to SLTD is that we have a finite, small number of regimes (3) rather than a potentially infinite set of MDPs. This makes inference tractable without the posterior sampling and dynamic programming that SLTD requires.

### Key References for This Section

- Joshi, S., Parbhoo, S. & Doshi-Velez, F. (2022). Sequential Learning-to-Defer under Non-Stationarity. arXiv:2109.06312v2.
- Omori, F. (1894). On the aftershocks of earthquakes. *Journal of the College of Science, Imperial University of Tokyo*, 7, 111–200.
- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401), 9–27. [ETAS model]
- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.

---

## 6. Bayesian Reliability Tracking

### The Core Idea

We maintain a Bayesian belief about PhaseNet's reliability — the probability that any given pick is correct — and we maintain this belief **separately for each regime**. This is because PhaseNet's reliability genuinely differs across regimes: it is more reliable during quiet background seismicity than during complex aftershock sequences.

### The Beta-Bernoulli Model

For each regime r ∈ {Q, A, D}, we model pick correctness as a Bernoulli random variable:

C_i | θ_r ~ Bernoulli(θ_r)

where C_i = 1 if the i-th pick (made during regime r) is correct, C_i = 0 if incorrect, and θ_r is the unknown reliability parameter for regime r.

We place a Beta prior on θ_r:

θ_r ~ Beta(α_r^0, β_r^0)

The Beta distribution is the conjugate prior for the Bernoulli likelihood, which means the posterior after observing data has the same functional form — it is also a Beta distribution. This is a major practical advantage: all updates are analytical, requiring no sampling or numerical approximation.

### Prior Selection

The prior parameters α_r^0 and β_r^0 encode our initial belief about PhaseNet's reliability in each regime, before seeing any Kazakhstan data. These priors should reflect the distribution shift — we expect lower reliability than in California, and our uncertainty about the exact reliability level.

**For California (in-distribution, used as reference)**:

θ_Q^CA ~ Beta(9, 1)  — strong prior that PhaseNet is ~90% reliable during quiet periods in California. This is informed by published performance metrics.

**For Kazakhstan Quiet regime**:

θ_Q^KZ ~ Beta(2, 2)  — weakly informative, centered at 50%, reflecting genuine uncertainty about PhaseNet's out-of-distribution performance. The prior mean is E[θ] = α/(α+β) = 0.5, and the prior variance is high (Var = αβ/[(α+β)²(α+β+1)] = 0.05), expressing that we really do not know how well PhaseNet will work.

**For Kazakhstan Active regime**:

θ_A^KZ ~ Beta(1.5, 2.5)  — weakly informative, centered below 50% (prior mean = 0.375), reflecting the expectation that PhaseNet will be less reliable during active periods with complex, overlapping waveforms.

**For Kazakhstan Decaying regime**:

θ_D^KZ ~ Beta(2, 2)  — same as Quiet initially, or perhaps slightly lower. The data will differentiate them.

### Posterior Updates

After observing n picks in regime r, of which k are correct (verified against ISC ground truth), the posterior is:

θ_r | data ~ Beta(α_r^0 + k, β_r^0 + n - k)

This is the standard conjugate update. The posterior mean is:

E[θ_r | data] = (α_r^0 + k) / (α_r^0 + β_r^0 + n)

As n grows, the posterior concentrates around the empirical reliability k/n, and the prior becomes less influential. With limited data (the Kazakhstan scenario), the prior has a meaningful effect — which is exactly why we use informative priors rather than flat ones.

### Example Update Sequence

Suppose during a Quiet period in Kazakhstan, we observe 30 PhaseNet picks and 21 are correct (verified against ISC picks). Starting from the prior θ_Q ~ Beta(2, 2):

Posterior: θ_Q | data ~ Beta(2 + 21, 2 + 9) = Beta(23, 11)

Posterior mean: 23/34 ≈ 0.676 — PhaseNet is about 68% reliable during quiet periods in Kazakhstan.

Now suppose during an Active period, we observe 15 picks and only 6 are correct. Starting from θ_A ~ Beta(1.5, 2.5):

Posterior: θ_A | data ~ Beta(1.5 + 6, 2.5 + 9) = Beta(7.5, 11.5)

Posterior mean: 7.5/19 ≈ 0.395 — PhaseNet is only about 40% reliable during active periods.

The difference between 68% and 40% reliability is substantial and directly affects the deferral decision. During quiet periods, PhaseNet is trusted more often. During active periods, more picks are deferred to human analysts.

### Handling Non-Stationarity Within Regimes

A subtlety: even within a single regime type, PhaseNet's reliability might change over time — for example, if the station characteristics drift, or if the nature of quiet-period seismicity changes seasonally. To handle this, we introduce a **forgetting factor** λ ∈ (0, 1) that gradually shrinks the posterior back toward the prior between time windows:

At the boundary between time windows:

α_r^{new} = λ × α_r^{posterior} + (1 - λ) × α_r^0

β_r^{new} = λ × β_r^{posterior} + (1 - λ) × β_r^0

When λ = 1, we retain the full posterior (no forgetting — stationary assumption within regimes). When λ = 0, we reset to the prior each window (complete forgetting — each window is independent). A value like λ = 0.8 provides moderate memory, retaining most of what was learned but allowing for gradual drift.

This posterior discounting is analogous to how SLTD re-estimates posteriors over dynamics at each time step, but implemented in a computationally trivial way.

### Key References for This Section

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd Edition. CRC Press. [Chapters 2–3 on conjugate models]
- Murphy, K.P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. [Section 3.3 on Beta-Bernoulli model]

---

## 7. The Deferral Policy

### Combining Regime Beliefs with Reliability Posteriors

At time t, the deferral decision for a PhaseNet pick combines two levels of inference:

**Level 1 — Regime inference**: The HMM forward algorithm (Section 5.2) gives us:

π_t(r) = P(R_t = r | o_1, ..., o_t)  for r ∈ {Q, A, D}

This is a probability distribution over regimes — our current belief about which regime we are in.

**Level 2 — Regime-specific reliability**: The Beta posteriors (Section 6) give us:

θ_r | data ~ Beta(α_r, β_r)  for r ∈ {Q, A, D}

This is our belief about PhaseNet's reliability in each regime.

### The Combined Reliability Estimate

The overall reliability estimate at time t is a **mixture** — we weight each regime's reliability by the probability of being in that regime:

**Point estimate (posterior mean)**:

θ̂_t = Σ_r π_t(r) × E[θ_r | data]

     = π_t(Q) × [α_Q/(α_Q + β_Q)] + π_t(A) × [α_A/(α_A + β_A)] + π_t(D) × [α_D/(α_D + β_D)]

This gives a single number summarising our current best estimate of PhaseNet's reliability.

**Full posterior** (for the deferral decision, we need more than the mean — we need the probability of being unreliable):

P(θ_t < θ_safe) = Σ_r π_t(r) × P(θ_r < θ_safe | data)

                 = Σ_r π_t(r) × I_{θ_safe}(α_r, β_r)

where I_x(a, b) is the regularised incomplete Beta function — the CDF of the Beta distribution evaluated at x. This is available as `scipy.stats.beta.cdf(theta_safe, alpha_r, beta_r)` in Python.

### The Deferral Decision Rule

The deferral rule is:

**Defer if** P(θ_t < θ_safe | data, regime beliefs) > τ

In words: defer the pick to a human analyst if the probability that PhaseNet's reliability is below a safety threshold exceeds a deferral tolerance.

There are two parameters to set:

**θ_safe — the safety threshold**: Below what reliability is PhaseNet unacceptable? This is a domain-specific choice. For seismology, if picks need to be correct at least 70% of the time to produce useful earthquake locations, then θ_safe = 0.7.

**τ — the deferral tolerance**: How much risk of unreliability are we willing to accept? A low τ (e.g., 0.1) means we defer aggressively — even a 10% chance of being below the safety threshold triggers deferral. A high τ (e.g., 0.5) means we only defer when it is more likely than not that PhaseNet is unreliable.

### Incorporating Per-Pick Uncertainty

The framework above uses only the **regional** reliability estimate — it does not use the per-pick uncertainty from Section 4 (MC Dropout, confidence scores). We can incorporate per-pick information as an additional deferral trigger:

**Combined deferral rule**:

Defer if:
- P(θ_t < θ_safe | data, regime beliefs) > τ  [regional reliability is suspect], OR
- the per-pick uncertainty (e.g., MC Dropout standard deviation) exceeds a threshold σ_max  [this specific pick is uncertain, regardless of regional reliability]

This two-level structure means that even during a quiet period where regional reliability is high, an individual pick with unusually high uncertainty will still be deferred. And during an active period where regional reliability is low, all picks may be deferred regardless of their individual confidence.

### Connection to SLTD

In SLTD, the deferral policy g(s, t) maps the current state s at time t to a binary decision: defer (d=1) or not (d=0). The policy is optimised to maximise long-term value using dynamic programming over the value function.

Our deferral rule is a **myopic approximation** — we make the decision based on current beliefs without considering how today's deferral affects future outcomes. The SLTD paper shows that the full sequential optimisation outperforms myopic policies in their healthcare setting.

However, the myopic policy is appropriate for this project for several reasons:

1. **Tractability**: The full RL optimisation requires defining state spaces, reward functions, and solving Bellman equations — a substantial engineering and theoretical challenge beyond MSc scope.

2. **Data limitations**: RL requires enough data to estimate value functions, which we do not have for Kazakhstan.

3. **Demonstrating the concept**: The regime-aware myopic policy already captures the key insight from SLTD — that deferral decisions should account for non-stationary dynamics. The extension to full sequential optimisation can be discussed as future work.

### Analysis of the Deferral Policy

For your report, you should analyse the deferral policy's behaviour by examining:

1. **Deferral rate by regime**: What fraction of picks are deferred during Quiet vs Active vs Decaying? The hypothesis is that Active has the highest deferral rate.

2. **Deferral rate over time**: Plot the deferral rate across your time period. It should spike after significant events and decay gradually — mirroring the regime transitions.

3. **Sensitivity to parameters**: How does the deferral rate change as you vary θ_safe and τ? This reveals the trade-off between safety (deferring more) and automation (deferring less).

4. **Comparison with static threshold**: Compare the regime-aware policy against a simple fixed threshold on PhaseNet confidence. The regime-aware policy should defer more during active periods and less during quiet periods, achieving better overall performance (higher accuracy when not deferring, without excessive deferral during quiet times).

### Key References for This Section

- Joshi, S., Parbhoo, S. & Doshi-Velez, F. (2022). Sequential Learning-to-Defer under Non-Stationarity. arXiv:2109.06312v2.
- Madras, D. et al. (2018). Predict responsibly: Improving fairness and accuracy by learning to defer. *NeurIPS 2018*.
- Mozannar, H. & Sontag, D. (2020). Consistent estimators for learning to defer to an expert. *ICML 2020*.

---

## 8. Human-in-the-Loop Simulation

### The Synthetic Analyst

When the deferral framework decides to defer a pick, it hands the decision to a human seismologist. In a real deployment, this would be an actual analyst at KNDC reviewing the waveform. For your project, you simulate this with a **synthetic analyst** whose properties are grounded in the literature:

**Accuracy**: The synthetic analyst picks correctly with probability p_human = 0.95. This is based on inter-analyst agreement rates reported in seismological studies, where experienced analysts agree on P-wave picks within ±0.1 seconds roughly 95% of the time for clear signals.

**Capacity constraint**: The analyst can review at most C picks per time window, where C represents 20–30% of the total picks in that window. This reflects the reality that human analysis is slow and expensive. If the deferral framework defers more than C picks, the excess must be either processed by PhaseNet (accepting the risk) or discarded (losing data).

**Cost**: Each deferral incurs a constant cost c representing analyst time. This mirrors Assumption 5.2 from the framework. In practice, this cost varies (an analyst is more "expensive" during an active swarm when they are already overloaded), but the constant cost assumption keeps the analysis tractable.

### The System Performance Metric

The overall system performance combines three components:

**Picks handled by PhaseNet** (not deferred):
- Accuracy: E[θ_r] for the current regime r — the model's reliability
- Cost: 0 (automated)

**Picks deferred to the analyst** (up to capacity):
- Accuracy: p_human = 0.95
- Cost: c per pick

**Picks deferred but exceeding capacity** (overflow):
- These must be handled by PhaseNet anyway (or discarded)
- Accuracy: E[θ_r] (same as automated)
- This is a failure mode — it means the deferral policy is asking for more human help than is available

The overall system accuracy is:

A_system = (1/N) × [Σ_{not deferred} E[θ_r] × 1(correct) + Σ_{deferred, within capacity} p_human × 1(correct) + Σ_{overflow} E[θ_r] × 1(correct)]

The key trade-off: deferring more picks improves accuracy (replacing model picks with human picks) but is constrained by capacity and incurs cost. The deferral policy should find the sweet spot — defer the picks where the model is most unreliable, up to the capacity limit.

### What to Measure

1. **System accuracy vs deferral rate**: As you increase the deferral tolerance τ (allowing more deferral), how does overall system accuracy change? There should be an optimal point where accuracy is maximised given the capacity constraint.

2. **Accuracy gain from regime-awareness**: Compare the regime-aware deferral policy against a static policy that defers the same total number of picks. The regime-aware policy should achieve higher accuracy because it concentrates deferrals during active periods where the model is least reliable.

3. **Capacity utilisation**: What fraction of the analyst's capacity is used? During quiet periods, it might be near zero. During active periods, it might saturate. This is directly relevant to real-world deployment — seismological agencies need to know how much analyst time the system requires.

---

## 9. Evaluation Framework

### Metrics for Each Stage

**Stage 1 — PhaseNet Baseline (California)**:
- Precision, recall, F1 score for pick detection
- Mean absolute error (MAE) of pick times relative to catalog picks
- Calibration curve (reliability diagram) for confidence scores

**Stage 2 — Distribution Shift Analysis (Kazakhstan)**:
- Same metrics as Stage 1, to quantify degradation
- Confidence score distributions: California vs Kazakhstan
- Pick residual distributions: California vs Kazakhstan

**Stage 3 — UQ Method Comparison**:
- Expected Calibration Error (ECE) for each method, in both regions
- AUROC for separating correct/incorrect picks using uncertainty
- Selective prediction curves: accuracy vs coverage as you reject increasingly uncertain picks

**Stage 4 — Regime-Aware Deferral**:
- Deferral rate by regime (Q/A/D)
- System accuracy at various deferral rates
- Comparison: regime-aware vs static threshold vs no deferral
- Calibration of the regime-specific posteriors (does the Beta model match observed reliability?)

**Stage 5 — Human-in-the-Loop**:
- System accuracy with analyst at various capacity levels
- Capacity utilisation by regime
- Cost-accuracy Pareto frontier

**Stage 6 — Equity Analysis**:
- Quantify the performance gap between California and Kazakhstan at each stage
- Show how the deferral framework narrows this gap
- Discuss the implications: without deferral, automated seismology amplifies inequality. With deferral, it manages the gap by routing uncertain decisions to humans.

### The Equity Argument

The equity analysis is not a technical add-on — it is the motivating context for the entire project. The argument runs as follows:

1. ML models trained on data from wealthy, well-instrumented regions are increasingly deployed globally.
2. Their performance degrades in data-sparse regions (this project demonstrates this concretely for seismology).
3. Without any safeguard, deploying these models in data-sparse regions provides a false sense of security — automated results look professional but may be wrong.
4. Learning-to-defer frameworks provide a principled safeguard: the system explicitly identifies where it cannot be trusted and routes those decisions to humans.
5. This does not eliminate the performance gap — Kazakhstan will still get worse automated performance than California — but it prevents silent failure and ensures that the highest-stakes decisions (during active seismic periods in data-sparse regions) receive human attention.

An important nuance (from our Assumption 3.1): the asymmetry between data-rich and data-sparse regions means that the deferral framework itself behaves differently. In California, deferral rates should be low across all regimes (the model is reliable even during active periods, because it was trained on similar data). In Kazakhstan, deferral rates should be high during active periods and moderate during quiet periods. The deferral framework thus acts as a **dynamic safety net** that automatically tightens when and where it is most needed.

---

## 10. Data Sources and Access

### California Baseline Data

**Waveforms**: Northern California Earthquake Data Center (NCEDC), accessible via FDSN web services through ObsPy. Network code: NC (Northern California Seismic Network). Use `Client("NCEDC")` in ObsPy.

**Ground truth picks**: The NCEDC catalog includes analyst-reviewed P and S arrival times at each station. This is the same dataset PhaseNet was originally trained on, so performance here represents in-distribution behaviour.

**URL**: https://ncedc.org

### Kazakhstan Data

**Waveforms**: Available through IRIS FDSN web services. Two access routes:

1. **IRIS/IDA stations** (guaranteed open access): BORK (network II), KURK (network II), MAKZ (network IU). Use `Client("IRIS")` in ObsPy. These stations were upgraded in 2019 with STS-5/STS-6 sensors and Quanterra Q330HR digitisers.

2. **KZ network stations**: MKAR, KKAR, ABKAR, AKTO, OTUK, PDGK, KASK, KNDC (network KZ). Also available through IRIS. Data from July 1994 onward.

**Recommended time period**: 2020–2023, to avoid instrument change confounds from the 2019 upgrades and to remain within the ISC Reviewed Bulletin window.

**Ground truth picks**: The International Seismological Centre (ISC) Reviewed Bulletin. KNDC is a contributing agency to the ISC and submits phase arrival times from its network. The ISC arrivals search (http://www.isc.ac.uk/iscbulletin/search/arrivals/) allows querying by station code and time range, returning CSV files with arrival times, phase names, residuals, and quality flags.

The reviewed bulletin has been manually checked by ISC analysts, and is available approximately 24 months behind real-time (currently up to roughly early 2024).

**Event catalog (for regime classification)**: ISC event catalogue (http://www.isc.ac.uk/iscbulletin/search/catalogue/). Search by geographic region (bounding box around Kazakhstan: approximately 38°N–50°N, 65°E–85°E) and time range. Output as CSV. This gives you dates, locations, magnitudes, and depths for all detected events — the input to your regime classification.

**Data volume guidance**: KNDC monitors Central Asian seismicity down to magnitude ~3.5 (lower in some areas), recording approximately 15,000 natural earthquakes per year across the region. The number of events with ISC-reviewed station-level picks at your specific stations will be substantially smaller — likely hundreds to low thousands over 2020–2023. This is sufficient for the Beta-Bernoulli model (which is designed for data-limited settings) but you should verify the actual count early in the project.

### SeisBench

SeisBench (Woollam et al., 2022) provides a standardised interface for seismological ML models, including PhaseNet. Use `seisbench.models.PhaseNet.from_pretrained("original")` to load the model trained on Northern California data. The "original" weights (rather than "stead") provide the cleanest distribution shift narrative, as the training data is exclusively from Northern California.

**Installation**: `pip install seisbench` (also installs ObsPy as a dependency).

---

## 11. Assumptions Summary

For reference, here is the complete set of assumptions underlying the framework, with justifications and acknowledged limitations:

**A1.1 — Non-stationarity**: The seismic environment is non-stationary — the probability distribution governing seismic activity changes over time. *Justification*: Aftershock sequences (Omori's Law), stress redistribution (Coulomb stress transfer), and tectonic loading are well-established non-stationary processes. *Limitation*: We do not model the specific physical mechanisms of non-stationarity — we treat it as a statistical property.

**A1.2 — Regime approximation**: Non-stationarity is modelled as transitions between three discrete regimes (Quiet, Active, Decaying). *Justification*: Seismologists already think in terms of background seismicity vs aftershock sequences vs swarm activity. The ETAS model in seismology essentially models earthquakes as triggering regime changes. *Limitation*: Reality is continuous; the discrete approximation loses fine-grained temporal structure.

**A2.1 — First-order Markov**: Regime transitions depend only on the current regime, not the full history. *Justification*: The most relevant information about current seismic state is the recent state. Consistent with SLTD's approach of estimating dynamics from recent data. *Limitation*: Long-term tectonic loading cycles (decades to centuries) are not captured.

**A3.1 — Uncertainty growth after events in data-sparse regions**: PhaseNet becomes less reliable after significant seismic activity in Kazakhstan, because post-event waveforms are further from the California training distribution. *Justification*: Post-event waveforms involve complex source effects, overlapping phases, potentially clipped records, and changed velocity structures — all unfamiliar to a California-trained model. *Important nuance*: In data-rich regions (California), the opposite may hold — more data and station redundancy can reduce uncertainty after events. This asymmetry is central to the equity argument.

**A4.1 — Noisy observation**: PhaseNet's output is treated as a noisy observation of pick quality, not ground truth. *Justification*: This is the Bayesian perspective — model outputs are evidence to be combined with prior beliefs, not facts.

**A4.2 — Beta priors**: Regional reliability follows a Beta distribution with regime-specific parameters. *Justification*: Conjugacy with Bernoulli likelihood makes updates analytical. The Beta family is flexible enough to represent beliefs ranging from complete uncertainty to strong confidence.

**A4.3 — Conditional independence**: Picks within a time window are independent given the regime. *Justification*: Simplifying assumption for tractability. *Limitation*: Picks from the same aftershock sequence are correlated in reality.

**A5.1 — Expert superiority**: The human analyst is more accurate (95%) but capacity-limited (20–30% of picks). *Justification*: Inter-analyst agreement rates in seismology literature; operational constraints of seismological agencies.

**A5.2 — Constant deferral cost**: Each deferral incurs the same cost regardless of context. *Justification*: Tractability. *Limitation*: In reality, analyst cost varies with workload — deferral during a crisis is more expensive than during calm periods.

**A5.3 — Binary deferral**: The decision is defer or don't — no partial deferral or graded confidence flags. *Justification*: Keeps the framework clean and directly comparable to SLTD. *Future work*: Graded deferral could be explored.

---

## 12. Key References

### Seismology and PhaseNet
- Zhu, W. & Beroza, G.C. (2019). PhaseNet: A deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261–273.
- Woollam, J. et al. (2022). SeisBench — A toolbox for machine learning in seismology. *Seismological Research Letters*, 93(3), 1695–1709.
- Omori, F. (1894). On the aftershocks of earthquakes. *Journal of the College of Science, Imperial University of Tokyo*, 7, 111–200.
- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401), 9–27.
- Mikhailova, N.N. & Sokolova, I.N. (2019). Monitoring System of the Institute of Geophysical Research. *Summary of the Bulletin of the ISC*, 53(1), 27–38.

### Learning to Defer and Safety-Critical ML
- Joshi, S., Parbhoo, S. & Doshi-Velez, F. (2022). Sequential Learning-to-Defer under Non-Stationarity. arXiv:2109.06312v2.
- Madras, D. et al. (2018). Predict responsibly: Improving fairness and accuracy by learning to defer. *NeurIPS 2018*.
- Mozannar, H. & Sontag, D. (2020). Consistent estimators for learning to defer to an expert. *ICML 2020*.

### Uncertainty Quantification
- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML 2017*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML 2016*.
- Lakshminarayanan, B., Pritzel, A. & Garnett, R. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
- Naeini, M.P., Cooper, G.F. & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning into quantiles. *AAAI 2015*.

### Bayesian Methods and HMMs
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd Edition. CRC Press.
- Murphy, K.P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.

### Distribution Shift
- Shimodaira, H. (2000). Improving predictive inference under covariate shift. *Journal of Statistical Planning and Inference*, 90(2), 227–244.
- Quinonero-Candela, J. et al. (2009). *Dataset Shift in Machine Learning*. MIT Press.

### U-Net Architecture (PhaseNet Basis)
- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
