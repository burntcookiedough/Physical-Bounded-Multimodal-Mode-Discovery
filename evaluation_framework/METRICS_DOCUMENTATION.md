# Physical-Bounded Multimodal Mode Discovery: Evaluation Framework Documentation

This document explicitly defines the mathematical principles, physical justifications, and interpretability frameworks for all metrics utilized in evaluating the **Physical-Bounded Multimodal Mode Discovery System**. 

The fundamental premise of this architecture is that operational modes in physical systems (e.g., pumps, bearings, turbines) are not merely statistical artifacts but thermodynamic and mechanical realities. Therefore, conventional Machine Learning metrics (while necessary) are insufficient. They must be augmented by consensus, temporal, and physical bounds.

---

## 1. Core Clustering Metrics (Unsupervised Subspace Quality)
These metrics evaluate the geometric organization of the feature space per modality *before* consensus arbitration.

### 1.1 Density-Based Clustering Validation (DBCV)
* **What it measures:** The relative density between clusters versus within clusters, specifically designed for non-globular clusters produced by HDBSCAN.
* **Formula:** $DBCV = \sum_{C_i \in C} \frac{|C_i|}{|C|} V_C(C_i)$ where $V_C = \frac{\min(d_{sep}(C_i, C_j)) - \max(d_{core}(x \in C_i))}{\max(\dots)}$.
* **System Relevance:** HDBSCAN handles the variable density of physical signals (e.g., some faults are tightly clustered exact states; others are loose combinations of mechanical wear). DBCV explicitly validates this variable-density geometry.
* **Interpretation:** 
  * *Good:* $+1$ (Perfectly separated dense regions).
  * *Bad:* $-1$ (Clusters overlap in low-density space).

### 1.2 Noise/Outlier Ratio
* **What it measures:** The percentage of samples labeled as `-1` (noise) by HDBSCAN.
* **Formula:** $R_{noise} = \frac{|x_{noise}|}{N_{total}}$
* **System Relevance:** In industrial IoT, strict cluster assignment forces anomalies into healthy modes. Leaving ambiguous data as "noise" allows the consensus filter or physical filter to flag it as transient/undefined physically.
* **Interpretation:** A moderate ratio (1% - 15%) is *good* (indicates robust outlier rejection). >30% means feature extraction failed to find signal.

### 1.3 Calinski-Harabasz & Davies-Bouldin
* **CH Score (Variance Ratio):** Ratio of between-cluster dispersion to within-cluster dispersion. *Higher is better.*
* **DBI:** Average similarity between each cluster and its most similar one. *Lower is better.*
* **System Relevance:** Evaluates macroscopic mode separation. Is the difference between "Bearing Fault" and "Normal Run" statistically significant on the current manifold?

---

## 2. Multimodal Consensus Metrics (Hierarchical Arbitration)
These metrics evaluate the system's ability to fuse conflicting views from Vibration, Current, and Temperature.

### 2.1 Modality Agreement Rate & Conflict Rate
* **What it measures:** The frequency at which the modality-specific independent HDBSCAN algorithms propose the *exact same* mode transition.
* **Formula:** $\text{Agreement} = \frac{1}{N} \sum_{t=1}^N \mathbf{I}(M_{vib}[t] = M_{cur}[t] = M_{temp}[t])$
* **System Relevance:** Physical systems transmit energy across domains. A bearing fault (vibration) increases friction (temperature) and requires more torque (current). However, delays exist (temperature lags). Conflict rate tracks these temporal mismatches.
* **Interpretation:** 
  * *Good Agreement:* ~75-85% (High consensus, but captures domain-specific nuances).
  * *Bad Agreement:* 100% (Implies redundant sensors) or <40% (System is decoupled/sensors failing).

### 2.2 Dominance Entropy (Modality Balance)
* **What it measures:** How evenly distributed the "winning" votes are across modalities for a specific discovered mode $k$.
* **Formula:** $H_{dom}(k) = - \sum_{m \in \{V,C,T\}} p_m(k) \log p_m(k)$ where $p_m(k)$ is the probability that modality $m$ had the highest confidence score for cluster $k$.
* **System Relevance:** A thermal runway mode should have Temperature dominating. A micro-fracture mode should have Vibration dominating. If Dominance Entropy is always 0, only one sensor is doing the work.
* **Interpretation:** *Good:* Varies by mode context. *Bad:* Uniformally low across the whole system (over-reliance on one sensor type).

---

## 3. Transition & Temporal Metrics (Behavioral Evolution)

### 3.1 Transition Entropy (System Uncertainty)
* **What it measures:** The unpredictability of the next operational state given the current state.
* **Formula:** $H_T = - \sum_i \pi_i \sum_j P(i \to j) \log P(i \to j)$ (where $P$ is the Transition Probability Matrix).
* **System Relevance:** A healthy machine transitions predictably (Off $\to$ Startup $\to$ Steady). A failing machine exhibits high Transition Entropy (rapid cycling, unstable modes).
* **Interpretation:** *Good:* Low entropy (stable operation). *Bad:* Spikes in rolling entropy (incipient failure).

### 3.2 Inter-Mode Degradation Trajectory Index (IMDTI)
* **What it measures:** The distance metric between the mode graph of a healthy run vs. a degrading run.
* **Formula:** $IMDTI = \sum_{t} || \Delta \text{Centroid}_{t} ||_2 \times P(M_k|M_{k-1})$ projected along the primary failure manifold.
* **System Relevance:** Tracks not just *that* a mode changed, but *how far* the system is moving structurally towards defined failure bounds.

---

## 4. Physics-Constraint Metrics (Thermodynamic bounds)
The core novelty of the framework. Machine learning proposes; Physics disposes.

### 4.1 Joule Heating Consistency Error
* **What it measures:** The deviation from the expected electro-thermal relationship $\Delta T \propto I^2 R$.
* **Formula:** $E_{joule}(C_k) = \frac{1}{|C_k|} \sum_{x \in C_k} \left| \Delta T(x) - \alpha \cdot I(x)^2 \right|$
* **System Relevance:** If the ML algorithm clusters a set of points where current is extremely high, but $\Delta T$ is zero or negative, the cluster represents a sensor ghost/fault, *not* a real physical mode.
* **Interpretation:** *Good:* Value near 0. *Bad:* Large deviations (cluster is physically impossible).

### 4.2 Energy Balance & Thermal Rate Violations
* **What it measures:** Whether the temporal derivative of temperature $\frac{dT}{dt}$ exceeds the maximum thermal mass capability of the material.
* **Formula:** $V_{thermal} = \mathbf{I}(\frac{\Delta T}{\Delta t} > \tau_{max})$
* **System Relevance:** AI often discovers high-frequency noise clusters. A physical filter rejecting points where $\Delta T$ implies instantaneous infinite heat transfer strictly bounds the latent space to physical reality.
* **Interpretation:** Constraint Violation Rate should Ideally be *0%* for accepted modes. Any rejected mode is discarded as invalid.

---

## 5. Generative & Augmentation Metrics

### 5.1 Physical Constraint Adherence of Synthetic Data
* **What it measures:** When using generative models (VAEs/GANs) to augment minority modes, do the synthetic points obey the thermodynamic laws?
* **Formula:** Ratio of synthetic points $X_{syn}$ passing the physical feasibility filter $F_{phys}(X) = 1$.
* **System Relevance:** Standard GANs hallucinate physically impossible data. This metric proves the generator is specifically *Physics-Bounded*.

### 5.2 Wasserstein Distance (Overlaps)
* **What it measures:** The Earth Mover's Distance between the real distribution $P_r$ and augmented distribution $P_g$.
* **Formula:** $W_1(P_r, P_g) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x,y)\sim \gamma} [||x-y||]$

---

## 6. Cross-Domain Generalization Metrics

### 6.1 Domain Transfer Consistency Score
* **What it measures:** Given two distinct domains (e.g., CWRU bearing dataset vs. CMAPSS turbine dataset), how well does the alignment matrix preserve topological mode connections?
* **System Relevance:** Industrial models fail when moved to a new machine. Scoring topological similarity proves the physics-first modes generalize across assets.

---

## 7. System Performance Metrics
* **Inference Latency:** Time to process one multimodal window frame ($T_n - T_0$). Real-world relevance: Must be $< 10$ ms for real-time PLC integration.
* **Throughput:** Processed operations per second.
* **Scalability:** Log-linear polynomial fit of CPU time $O(n \log n)$ verifying HDBSCAN and arbitration scale efficiently.
