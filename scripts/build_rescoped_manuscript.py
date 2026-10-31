import os

tex_content = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}

\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{cite}
\usepackage{microtype}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black
}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{remark}{Remark}

\hyphenation{op-tical net-works semi-conduc-tor C-MAPSS PRONOSTIA Con-ti-For-mer Sym-ODEN}

\begin{document}

\title{Blackout-Aware Uncertainty Calibration, Horizon-Adaptive Phase-Space Dynamics, and Maintenance Decision Optimization for RUL Prognostics Under Telemetry Outages}

\author{Anonymous Authors%
\thanks{Manuscript prepared for submission to \emph{Reliability Engineering \& System Safety} (RESS). Reproducibility scripts, 30-seed and 5-fold cross-validation ledgers, and conformal/CBM evaluation code are included in the companion archive.}}

\maketitle

\begin{abstract}
Remaining Useful Life (RUL) prognostics for safety-critical assets must remain calibrated when telemetry is interrupted by multi-cycle communication blackouts. We identify three failure modes in continuous-time RUL pipelines during prolonged outages. First, bounded-horizon integration clamps ($h_k = \min(\tilde{\Delta t}_k, h_{\max})$) render point predictions invariant to blackout duration beyond a fixed cutoff, causing a $+12.75$-cycle optimistic bias during active degradation on NASA C-MAPSS FD001. Second, first-order confined neural ODEs either collapse toward the fleet mean on healthy plateaus or lag during active degradation because they lack an independent velocity coordinate. Third, symmetric Mondrian conformal intervals inherit positive blackout bias, reducing empirical coverage on engines in true mid-life degradation ($\mathrm{RUL} \in (25, 75]$) to $88.93\%$ against a $95\%$ target. We resolve these failure modes by pairing second-order position--velocity phase-space extrapolation and horizon-adaptive substepping with a backbone-agnostic Bias-Aware Mondrian Drift-Diffusion conformal calibrator and validation-tuned lower-confidence-bound (LCB) Condition-Based Maintenance (CBM). Across 30 seeds and 20 models, lifting into second-order coordinates with secant velocity supervision accounts for $80\%$ of the single-step out-of-distribution (OOD) gain over first-order confined ODEs, while damped Euler--Lagrange dynamics with Strang splitting statistically tie PatchTST-RUL and Latent SDEs on $10\times\text{--}20\times$ blackouts ($19.33 \pm 0.49$ cycles RMSE) but cut seed standard deviation by $45\%$ and match RK4 rollout accuracy at fourfold fewer force evaluations. Across eight backbones, Bias-Aware Mondrian calibration restores $94.48\%$ empirical mid-life coverage and minimizes validation-tuned CBM cost for failure-to-preventive cost ratios $\ge 5:1$.
\end{abstract}

\begin{IEEEkeywords}
Prognostics and health management, remaining useful life, telemetry blackouts, conformal prediction, covariate shift, second-order neural ODEs, condition-based maintenance.
\end{IEEEkeywords}

\section{Introduction}
\label{sec:intro}

\IEEEPARstart{P}{redictive} maintenance and Remaining Useful Life (RUL) estimation for aerospace turbofans and industrial rotating machinery must remain reliable when telemetry is interrupted by burst packet drops, satellite bandwidth throttling, and multi-cycle communication blackouts \cite{saxena2008damage, arias2021aircraft, nektarios2012pronostia, de2019deep}. When inter-observation gaps $\Delta t_k = t_{k+1} - t_k$ span an order of magnitude beyond nominal sampling intervals $\tau_0$, discrete-time sequence models---such as Long Short-Term Memory (LSTM) networks \cite{zheng2017long}, Deep Convolutional Neural Networks (DCNNs) \cite{li2018remaining}, and discrete Joint-Embedding Predictive Architectures (JEPAs) \cite{lecun2022path, assran2023self}---suffer from horizon miscalibration because they treat every transition as a uniform step.

Continuous-time and patch-based sequence architectures---including GRU-D \cite{che2018recurrent}, Latent Neural ODEs \cite{chen2018neural}, ODE-RNNs \cite{rubanova2019latent}, Neural CDEs \cite{kidger2020neural}, Latent Neural SDEs \cite{li2020scalable}, mTAN \cite{shukla2021multi}, ContiFormer \cite{chen2023contiformer}, diagonal structured state-space models (S4D) \cite{gu2022parameterization}, and PatchTST \cite{nie2023time}---explicitly condition state transitions on timestamps or local subseries patches. Physics-structured neural differential equations---including Hamiltonian Neural Networks (HNNs) \cite{greydanus2019hamiltonian}, Lagrangian Neural Networks (LNNs) \cite{cranmer2020lagrangian}, Symplectic ODE-Net (SymODEN) \cite{zhong2020symplectic}, and Port-Hamiltonian Neural Networks (pHNNs) \cite{desai2021port}---further constrain latent flows via energy and dissipation priors. Under dense sampling or moderate missingness, these architectures achieve nearly identical RUL accuracy ($12.42\text{--}12.58$ cycles RMSE on NASA C-MAPSS FD001; Table~\ref{tab:17competitors_ood}).

However, when telemetry is interrupted by prolonged out-of-distribution (OOD) blackouts ($\Delta t_k \in [10\tau_0, 25\tau_0]$), three operational reliability gaps emerge across existing continuous-time prognostic pipelines:
\begin{enumerate}
    \item \textbf{Silent Point-Forecast Saturation Under Horizon Clamping:} To prevent fixed-step numerical solvers from diverging over long gaps, continuous-time ODE implementations commonly clamp the integration horizon to $h_k = \min(\tilde{\Delta t}_k, h_{\max})$. We show analytically and empirically (Section~\ref{sec:clamp_diagnosis}, Fig.~\ref{fig:ood_slope_symp}(b)) that when $h_k$ and the normalized gap context saturate at $\Delta t_k \ge 4\tau_0$, the predicted RUL $\widehat{\mathrm{RUL}}(\Delta t)$ becomes completely flat for all longer blackouts ($61.94$ cycles from a $65.0$-cycle pre-gap state), inducing a $+12.75$-cycle optimistic bias on actively degrading units ($\mathrm{Slope}=-1$; Table~\ref{tab:slope_breakdown}).
    \item \textbf{The First-Order Confinement Dilemma vs.\ Second-Order Velocity Supervision:} During long blackouts, unregularized first-order latent ODEs drift off the training manifold, while origin-anchored confinement ($-\alpha z$) pulls trajectories toward the fleet-mean RUL ($\approx 75$ cycles on FD001), causing a $-23.00$-cycle bias on the healthy plateau ($\mathrm{Slope}=0$; Table~\ref{tab:slope_breakdown}). Anchoring a first-order well to the pre-gap state ($-\alpha_{\mathrm{eff}}(z - z_k)$) prevents plateau collapse ($21.21 \pm 1.60$ cycles RMSE; Table~\ref{tab:17competitors_ood}) but lags during active linear degradation because a first-order state lacks an independent velocity coordinate $\dot{q}_k$. Prior physics-informed ODE studies have not isolated whether OOD gains come from second-order velocity supervision or from Hamiltonian/Euler--Lagrange mechanics.
    \item \textbf{Mid-Life Conformal Under-Coverage and Uncalibrated Maintenance Thresholds:} Standard split conformal \cite{vovk2005algorithmic, lei2018distribution} and symmetric Mondrian conformal \cite{romano2019conformalized} intervals centered on biased blackout point forecasts systematically shift above ground-truth RUL during post-knee active degradation ($\mathrm{RUL} \in (25, 75]$), causing empirical mid-life coverage to drop to $88.93\%$ against a $95.0\%$ nominal target (Table~\ref{tab:conformal_ablation}). In turn, Condition-Based Maintenance (CBM) rules that threshold raw point forecasts cannot adapt their safety buffer to the blackout duration $\Delta t_k$.
\end{enumerate}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/Fig0_CT_PhysProb_JEPA_Architecture.pdf}
\caption{Overview of the blackout-aware prognostic and maintenance pipeline. (1) Irregular telemetry $x_k$, operational controls $u_k$, and elapsed gaps $\Delta t_k = t_{k+1}-t_k$ are augmented with causal dual-timescale exponential moving average (EMA) memory traces $(m_k^{(f)}, m_k^{(s)})$. (2a--2b) Context encoder $f_\theta$ and stop-gradient EMA target encoder $f_\xi$ lift observations into position--velocity phase space $z_k = (q_k, \dot{q}_k) \in \mathbb{R}^{2d}$ ($2d=16$), supervised by secant velocity alignment $\mathcal{L}_{\mathrm{kin}}$. (3) Latent state evolution uses either bounded-horizon clamping ($h_{\max}=0.35, M=4$) or unclamped full-gap horizon-adaptive substepping ($M_k = \lceil \widetilde{\Delta t}_k / 0.0875 \rceil$) with secant drift tracking. (4) A backbone-agnostic Bias-Aware Mondrian Drift-Diffusion and Importance-Weighted conformal calibrator de-centers gap-dependent blackout bias on validation data to drive validation-tuned lower-confidence-bound (LCB) Condition-Based Maintenance (CBM) decisions.}
\label{fig:arch_schematic}
\end{figure*}

In this paper, we re-center continuous-time prognostics around end-to-end blackout reliability: diagnosing and eliminating clamp saturation, isolating the role of second-order phase-space dynamics via a strict causal control, restoring mid-life uncertainty calibration across eight diverse backbones, and optimizing validation-tuned CBM decisions (Fig.~\ref{fig:arch_schematic}). Our specific contributions and empirical findings are:
\begin{enumerate}
    \item \textbf{Diagnosis of Clamp Saturation and Resolution via Horizon-Adaptive Substepping (`Sec. 3.3`, `Sec. 4.3`, `Table 2`, `Fig. 3(b)`):} We prove that clamping $h_k = \min(\tilde{\Delta t}_k, 0.35)$ freezes point predictions beyond $\Delta t_k = 4\tau_0$. Replacing fixed-step clamping with unclamped \textbf{Full-Gap Horizon-Adaptive Substepping} ($M_k = \lceil \tilde{\Delta t}_k / 0.0875 \rceil$) and secant drift tracking guarantees discrete spectral stability ($\delta t_k \le 0.0875 < 2/\sqrt{\alpha_{\mathrm{eff}}} = 3.381$; Proposition~\ref{prop:jury}), restores linear degradation tracking across $25\tau_0$ blackouts (Fig.~\ref{fig:ood_slope_symp}(b)), and cuts active-degradation ($\mathrm{Slope}=-1$) optimistic bias by $46.4\%$ ($+12.75 \to +6.84$ cycles; active-degradation RMSE reduced from $23.09$ to $20.14$ cycles in Table~\ref{tab:slope_breakdown}).
    \item \textbf{Backbone-Agnostic Bias-Aware Mondrian Conformal Calibration and Shift Adaptation (`Sec. 4.4`, `Table 6`, `Table 7`, `Fig. 4(b)`):} We clearly distinguish formal predicted-bin Mondrian guarantees from unobservable true-RUL-conditional diagnostics. By fitting a per-bin linear gap-bias slope $\hat{\beta}_b$ on held-out validation data, \textbf{Bias-Aware Mondrian Drift-Diffusion Conformal} raises empirical True Mid-Life $(25, 75]$ coverage on FD001 $10\times\text{--}20\times$ blackouts from $88.93\%$ to $94.48 \pm 1.12\%$ ($94.82\%$ with adaptive substepping) and improves empirical mid-life coverage across all eight evaluated neural backbones to $93.85\%\text{--}94.82\%$ while narrowing interval width by $9.5\text{--}12.2$ cycles vs.\ standard split conformal (Table~\ref{tab:conformal_ablation}). Under simulated covariate shift from in-distribution ($25\%$ dropout) calibration to unseen OOD blackout regimes, combining Bias-Aware Mondrian with likelihood-ratio importance weighting \cite{tibshirani2019conformal} achieves $94.92\%\text{--}95.24\%$ empirical marginal coverage (Table~\ref{tab:unseen_shift}).
    \item \textbf{Causal Attribution of Second-Order Velocity Supervision vs.\ Euler--Lagrange Structure (`Sec. 4.1--4.2`, `Table 1`, `Table 3`, `Table 4`, `Table 5`):} Across $S=30$ seeds and 20 models on FD001 (Table~\ref{tab:17competitors_ood}), our causal control \textbf{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$} (which keeps $(q_k, \dot{q}_k)$, displacement anchoring, and $\mathcal{L}_{\mathrm{kin}}$ but removes the Euler--Lagrange potential, Rayleigh damping, and Strang splitting) achieves $19.70 \pm 0.90$ cycles RMSE on \texttt{OOD 10x--20x}, capturing $80.3\%$ of the RMSE reduction over first-order displacement-anchored NODEs ($21.21 \pm 1.60$ cycles). Adding damped Euler--Lagrange mechanics and Strang splitting (\textbf{CT-PhysProb-JEPA}, $19.33 \pm 0.49$ cycles) is statistically tied in mean RMSE on \texttt{OOD 10x--20x} with \textbf{PatchTST-RUL} ($19.26 \pm 0.85$, $p_{t,\mathrm{Holm}} = 1.00$), \textbf{Latent Neural SDE} ($19.35 \pm 0.86$, $p_{t,\mathrm{Holm}} = 1.00$), and \textbf{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} ($p_{t,\mathrm{Holm}} = 0.0964$), but delivers three specific advantages: (i) a $45.6\%$ reduction in seed-to-seed standard deviation ($0.90 \to 0.49$ cycles), (ii) a statistically significant gain on extreme $15\times\text{--}25\times$ blackouts ($20.40 \pm 0.53$ vs.\ $21.34 \pm 1.12$, $p_{t,\mathrm{Holm}} = 3.53\times 10^{-5}$), and (iii) matching 4th-order Runge--Kutta (RK4) rollout accuracy ($20.48$ vs.\ $20.47$ cycles) at $4\times$ fewer force evaluations ($\mathrm{NFE}=4$ vs.\ $16$; Table~\ref{tab:symplectic_rollout}).
    \item \textbf{Validation-Tuned 500-Trajectory Condition-Based Maintenance Study (`Sec. 6`, `Table 8`, `Fig. 4(c)`):} Using an equal 50-point threshold grid search on held-out validation trajectories across 10 architectures and $500$ critical test truncation trajectories per seed ($\times 30$ seeds), replacing point-threshold rules with Bias-Aware Mondrian LCB policies cuts unplanned failure rates by $4.6\times$ ($1.56\% \to 0.34\%$) and reduces expected maintenance cost per unit across all failure-to-preventive cost ratios $\ge 5:1$ (Table~\ref{tab:cbm_decision}).
\end{enumerate}

\section{Related Work}
\label{sec:related}

\subsection{Irregular Time-Series and Sequence Architectures in PHM}
Irregular sampling in prognostics has historically been handled by Last-Observation-Carried-Forward (LOCF) imputation paired with recurrent or convolutional backbones \cite{saxena2008damage, zheng2017long, li2018remaining, de2019deep}. Che et al.\ \cite{che2018recurrent} introduced GRU-D with trainable exponential decay toward empirical means. Continuous-time latent dynamics were formalized via Neural ODEs \cite{chen2018neural}, ODE-RNNs \cite{rubanova2019latent}, Neural Controlled Differential Equations (Neural CDEs) \cite{kidger2020neural}, and adjoint-trained Latent Neural SDEs \cite{li2020scalable}. Attention- and state-space-based continuous-time models include multi-time attention networks (mTAN) \cite{shukla2021multi}, ContiFormer \cite{chen2023contiformer}, and diagonal state-space models (S4D) \cite{gu2022parameterization}, while PatchTST \cite{nie2023time} tokenizes local subseries patches for long-range forecasting. Although effective under moderate missingness, first-order continuous-time states $z_k \in \mathbb{R}^d$ do not separate position from degradation velocity, creating a structural conflict between unregularized drift and static equilibrium pinning during prolonged blackouts.

\subsection{Physics-Structured Neural Differential Equations and Geometric Solvers}
Physics-informed neural networks (PINNs) \cite{raissi2019physics, cuomo2022scientific}, Hamiltonian Neural Networks (HNNs) \cite{greydanus2019hamiltonian}, and Lagrangian Neural Networks (LNNs) \cite{cranmer2020lagrangian} embed conservation laws into continuous-time dynamics. Zhong et al.\ \cite{zhong2020symplectic} extended HNNs to open control systems via Symplectic ODE-Net (SymODEN), and Desai et al.\ \cite{desai2021port} formulated Port-Hamiltonian Neural Networks (pHNNs) with explicit energy-dissipation matrices. However, strictly conservative HNNs, LNNs, and SymODEN ($\gamma=0$) preserve phase-space volume and cannot damp arbitrary initial velocity perturbations, whereas standard pHNN implementations integrated with explicit Euler steps lack secant velocity supervision and Strang-split conformal symplectic volume contraction \cite{hairer1993solving, hairer2006geometric, sanz1994numerical}.

\subsection{Conformal Calibration Under Shift and Maintenance Optimization}
Split conformal prediction \cite{vovk2005algorithmic, lei2018distribution, angelopoulos2021gentle} and Conformalized Quantile Regression (CQR) \cite{romano2019conformalized} provide distribution-free marginal coverage when calibration and test examples are exchangeable. Mondrian conformal prediction \cite{vovk2005algorithmic, romano2019conformalized} guarantees conditional coverage within observable categories, while Tibshirani et al.\ \cite{tibshirani2019conformal} extended split conformal prediction to weighted exchangeability under covariate shift via likelihood-ratio importance weights, and Gibbs and Cand\`{e}s \cite{gibbs2021adaptive} proposed online Adaptive Conformal Inference (ACI). In run-to-failure blackout prognostics, online ground-truth RUL labels are unavailable prior to asset failure (limiting feedback-driven ACI), and positive blackout extrapolation bias shifts symmetric intervals away from true mid-life RUL. Our work addresses this gap via validation-fitted bias de-centering within Mondrian bins and evaluates its downstream impact on validation-tuned CBM decision rules.

\section{Problem Formulation, Outage Models, and Clamp Saturation Diagnosis}
\label{sec:problem}

\subsection{Run-to-Failure Prognostics Under Irregular Telemetry}
Consider a fleet of $N$ industrial assets indexed by $n \in \{1, \dots, N\}$, generating multivariate telemetry streams $\mathcal{D}_n = \{(t_k, x_k, u_k, y_k)\}_{k=1}^{K_n}$ at strictly increasing observation timestamps $0 \le t_1 < t_2 < \dots < t_{K_n}$. Here $x_k \in \mathbb{R}^{d_x}$ denotes $d_x$ sensor measurements, $u_k \in \mathbb{R}^{d_u}$ denotes $d_u$ operational settings, and $y_k = \min(Y_{\max}, T_{\mathrm{fail},n} - t_k) \in [0, Y_{\max}]$ denotes ground-truth RUL capped at $Y_{\max}$ ($125$ cycles for C-MAPSS FD001/FD002/FD004, $85$ cycles for N-CMAPSS DS02, and $3000\text{ s}$ for PRONOSTIA).

To prevent temporal leakage, the inter-observation gap is measured from the physical clock prior to subsampling:
\begin{equation}
\Delta t_k = t_{k+1} - t_k > 0, \qquad \tilde{\Delta t}_k = \frac{\Delta t_k}{S_{\tau} \tau_0},
\label{eq:time_norm}
\end{equation}
where $\tau_0 = \mathrm{median}_{\mathrm{train}}(\Delta t)$ is fitted exclusively on training units and $S_{\tau} = 4.0$ scales nominal steps. Each observation $x_k$ is augmented with causal dual-timescale exponential moving average (EMA) memory traces $(m_{k}^{\mathrm{fast}}, m_{k}^{\mathrm{slow}}) \in \mathbb{R}^{2d_x}$:
\begin{equation}
m_k^{\mathrm{fast}} = e^{-\gamma_f \Delta t_{k-1}/\tau_0} m_{k-1}^{\mathrm{fast}} + \big(1 - e^{-\gamma_f \Delta t_{k-1}/\tau_0}\big) x_k,
\label{eq:causal_ema}
\end{equation}
with $\gamma_f = 0.5$, and analogously for $m_k^{\mathrm{slow}}$ with $\gamma_s = 0.1$, initialized at $m_1^{\mathrm{fast}} = m_1^{\mathrm{slow}} = x_1$.

\subsection{Operationally Motivated Outage Simulation Models}
\label{sec:outage_models}
To evaluate robustness across diverse gap distributions, we test both controlled OOD blackout intervals ($\Delta t_k \in [5\tau_0, 10\tau_0]$, $[10\tau_0, 20\tau_0]$, and $[15\tau_0, 25\tau_0]$) after training exclusively on $\Delta t_k \in [1\tau_0, 2\tau_0]$, and two operationally motivated synthetic outage processes:
\begin{enumerate}
    \item \textbf{2-State Gilbert--Elliott Markov ACARS Outage Simulator (\texttt{Markov ACARS}):} Models bursty VHF/SATCOM datalink dropouts via a 2-state Markov chain between a Good state $G$ (transmission probability $p_G = 0.95$, transition $P(G \to B) = 0.08$) and a Burst-Outage state $B$ ($p_B = 0.05$, recovery $P(B \to G) = 0.12$, mean burst length $8.33\tau_0$, 95th percentile gap $14.0\tau_0$, maximum $28.0\tau_0$).
    \item \textbf{Flight-Phase Cruise Throttling Policy (\texttt{Cruise-Throttle}):} Models bandwidth-saving telemetry policies \cite{arias2021aircraft} that transmit every $1\text{--}2\tau_0$ during climb/descent transients but throttle to $10\text{--}18\tau_0$ during steady cruise, yielding median gap $2.0\tau_0$, 75th percentile $10.0\tau_0$, and $22.78\%$ of intervals $\ge 10\tau_0$.
\end{enumerate}

\noindent\textbf{[EXPERIMENT NEEDED: Real-World Outage Telemetry Validation and Gap-Distribution Stress Test.}
\emph{Description:} Supplement the synthetic Gilbert--Elliott and Cruise-Throttle masks with genuine field missingness logs (e.g., real aircraft ACARS/QAR datalink dropout timestamps, SCADA wind-turbine sensor outage logs, or an industrial run-to-failure dataset containing native sensor dropouts) and compare empirical gap survival functions $\mathbb{P}(\Delta t > \tau)$ and burst autocorrelation against the synthetic generators.
\emph{Protocol:} (1) Extract empirical inter-transmission gap sequences $\{\Delta t_k^{\mathrm{real}}\}$ from field logs, compute summary statistics (median, 90th/99th percentiles, maximum blackout duration, and lag-1 burst autocorrelation), and run a two-sample Kolmogorov--Smirnov and Wasserstein-1 distance audit against \texttt{Markov ACARS} and \texttt{Cruise-Throttle}; (2) apply the exact empirical timestamp masks $\{\Delta t_k^{\mathrm{real}}\}$ to C-MAPSS/N-CMAPSS test trajectories (and evaluate directly on the native industrial dataset if run-to-failure labels exist) across all 20 models and 30 seeds.
\emph{Expected Table Format:} A two-part table reporting (Part A) Gap Distribution Fidelity (Median, P90, P99, Max $\Delta t/\tau_0$, Lag-1 $\rho$, KS statistic) and (Part B) 30-Seed RUL RMSE, Active-Degradation Bias, Mid-Life Conformal Coverage, and CBM Cost under the empirical field outage logs.\textbf{]}

\subsection{Analytical Diagnosis of Clamp Saturation}
\label{sec:clamp_diagnosis}
In continuous-time latent ODE architectures, a common numerical safeguard against step-size instability during long blackouts is bounded-horizon clamping, which restricts the effective integration duration to $h_k = \min(\tilde{\Delta t}_k, h_{\max})$ (with $h_{\max} = 0.35$, corresponding to $\Delta t_k = 1.4\tau_0$) and clips the normalized gap context feature to $\tilde{\Delta t}_{\mathrm{clip},k} = \min(\tilde{\Delta t}_k, 1.0)$ (corresponding to $\Delta t_k = 4\tau_0$).

\begin{proposition}[Point-Forecast Clamp Saturation]
\label{prop:clamp_sat}
Let $\Phi_{h_k}(z_k; u_k, \tilde{\Delta t}_{\mathrm{clip},k})$ denote any continuous-time latent flow integrated over clamped duration $h_k = \min(\tilde{\Delta t}_k, h_{\max})$ with context input $\tilde{\Delta t}_{\mathrm{clip},k} = \min(\tilde{\Delta t}_k, 1.0)$, followed by a readout probe $\hat{\mu}_{k+1} = g_\omega(\Phi_{h_k}(z_k; u_k, \tilde{\Delta t}_{\mathrm{clip},k}), u_k)$. For any fixed pre-gap state $z_k$ and operating condition $u_k$, the point forecast $\widehat{\mathrm{RUL}}(\Delta t_k) = \hat{\mu}_{k+1}$ is strictly constant with respect to blackout duration for all $\Delta t_k \ge S_\tau \tau_0 \max(h_{\max}, 1.0) = 4\tau_0$:
\begin{equation}
\frac{\partial \, \widehat{\mathrm{RUL}}(\Delta t_k)}{\partial (\Delta t_k)} = 0 \qquad \forall \, \Delta t_k > 4\tau_0.
\label{eq:clamp_zero_deriv}
\end{equation}
Consequently, on any asset undergoing linear active degradation with true RUL slope $\frac{dy}{dt} = -1\text{ cycle/cycle}$ over $\Delta t_k > 4\tau_0$, the pointwise prediction error grows linearly with rate $+1\text{ cycle}$ per additional blackout cycle:
\begin{equation}
\hat{\mu}_{k+1}(\Delta t_k) - y(t_k + \Delta t_k) = \big(\hat{\mu}_{k+1}(4\tau_0) - y(t_k + 4\tau_0)\big) + (\Delta t_k - 4\tau_0).
\label{eq:clamp_linear_bias}
\end{equation}
\end{proposition}

Proposition~\ref{prop:clamp_sat} explains why clamped continuous-time models exhibit a flat prediction plateau for $\Delta t_k \ge 4\tau_0$ in Fig.~\ref{fig:ood_slope_symp}(b) ($61.94$ cycles from a $65.0$-cycle pre-gap state) and incur a $+12.75$-cycle optimistic bias during active degradation ($\mathrm{Slope}=-1$; Table~\ref{tab:slope_breakdown}).

\section{Methodology: Second-Order Phase-Space Extrapolation and Bias-Aware Conformal Calibration}
\label{sec:method}

\subsection{Second-Order Phase-Space Lifting and Velocity Supervision}
Given the augmented causal observation vector $\tilde{x}_k = [x_k; m_k^{\mathrm{fast}}; m_k^{\mathrm{slow}}; u_k] \in \mathbb{R}^{3d_x + d_u}$, a 2-layer $\tanh$ context encoder $f_\theta: \mathbb{R}^{3d_x + d_u} \to \mathbb{R}^{2d}$ maps $\tilde{x}_k$ into canonical position--velocity coordinates:
\begin{equation}
z_k = f_\theta(\tilde{x}_k) = \begin{bmatrix} q_k \\ \dot{q}_k \end{bmatrix} \in \mathbb{R}^{2d}, \quad q_k \in \mathbb{R}^d, \;\; \dot{q}_k \in \mathbb{R}^d,
\label{eq:phase_lift}
\end{equation}
where $q_k$ is the $d$-dimensional generalized degradation position ($d=8$, $2d=16$) and $\dot{q}_k$ is the generalized degradation velocity. A momentum target encoder $f_\xi$ encodes $\tilde{x}_{k+1}$ at $t_{k+1} = t_k + \Delta t_k$ into $\tilde{z}_{k+1} = [\tilde{q}_{k+1}; \dot{\tilde{q}}_{k+1}] = \mathrm{stop\_grad}(f_\xi(\tilde{x}_{k+1}))$, updated via $\xi \leftarrow \tau_{\mathrm{ema}}\xi + (1 - \tau_{\mathrm{ema}})\theta$ ($\tau_{\mathrm{ema}} = 0.985$). To supervise $\dot{q}_k$ with the empirical rate of latent state change across $\tilde{\Delta t}_k$, we include the secant velocity alignment loss during training:
\begin{equation}
\mathcal{L}_{\mathrm{kin}} = \frac{1}{2}\big\|\dot{q}_k - v_{\mathrm{sec},k}\big\|_2^2, \qquad v_{\mathrm{sec},k} = \frac{\tilde{q}_{k+1} - q_k}{\max(1.0, \, 4\tilde{\Delta t}_k)}.
\label{eq:secant_loss}
\end{equation}

\subsection{Damped Euler--Lagrange Backbone vs.\ Plain Second-Order Causal Control}
To test whether OOD accuracy gains over first-order NODEs stem from second-order velocity supervision or from physics-inspired potential structure, we evaluate two matched second-order parameterizations on $z = [q; p] \in \mathbb{R}^{2d}$ ($p = \dot{q}$):
\begin{enumerate}
    \item \textbf{Damped Euler--Lagrange Backbone (\texttt{CT-PhysProb-JEPA}):} Endows the phase space with unit mass, a displacement-relative confining potential $V_\phi(q; q_k)$, Rayleigh dissipation $\mathcal{R}(\dot{q}) = \frac{1}{2}\gamma\|\dot{q}\|_2^2$ ($\gamma = 0.15$), and a bounded control potential $U_\psi(q; \eta_k, q_k) = -\frac{c_f}{\beta_q}\sum_{j=1}^d \log\cosh(\eta_{k,j} + \beta_q(q_j - q_{k,j}))$ ($c_f = 0.45, \beta_q = 0.35$, $c_f\beta_q = 0.1575$), where $\eta_k = W_{\mathrm{diss}}^T[q_k; \dot{q}_k; u_k; \min(\tilde{\Delta t}_k, 1.0)] + b_{\mathrm{diss}}$:
    \begin{equation}
    V_\phi(q; q_k) = \frac{1}{d}\sum_{i=1}^d s_i \log\cosh\big(w_i^T q + b_i\big) + \frac{\alpha_{\mathrm{eff}}}{2}\|q - q_k\|_2^2,
    \label{eq:potential}
    \end{equation}
    with $s_i \in \{+1, -0.65\}$ ($s_i = -0.65$ for $i \equiv 0 \pmod 3$) and $\alpha_{\mathrm{eff}} = c_\alpha \alpha = 0.35$ ($\alpha = 0.10, c_\alpha = 3.5$). The damped Euler--Lagrange equations of motion on $t \in [t_k, t_k + H_k]$ are:
    \begin{equation}
    \frac{d}{dt}\begin{bmatrix} q \\ p \end{bmatrix} = \begin{bmatrix} p \\ -\nabla_q V_\phi(q; q_k) + F_{\mathrm{ext}}(q; \eta_k, q_k) - \gamma p \end{bmatrix},
    \label{eq:eom}
    \end{equation}
    where $F_{\mathrm{ext}}(q; \eta_k, q_k) = -\nabla_q U_\psi(q; \eta_k, q_k) = c_f\tanh(\eta_k + \beta_q(q - q_k))$. Across $M$ substeps of size $\delta t = H_k / M$, Eq.~\eqref{eq:eom} is integrated via the symmetric Strang-split conformal symplectic leapfrog map $\Psi_{\delta t} = \mathcal{D}_{\delta t/2} \circ \mathcal{S}_{\delta t} \circ \mathcal{D}_{\delta t/2}$, which requires only $\mathrm{NFE} = 1$ force evaluation per substep by caching $F_{\mathrm{tot}}(q^{(m+1)})$ across steps.
    \item \textbf{Causal Control (\texttt{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$}):} Retains the exact same encoder $z_k = [q_k; \dot{q}_k] \in \mathbb{R}^{16}$, secant velocity loss $\mathcal{L}_{\mathrm{kin}}$ (Eq.~\eqref{eq:secant_loss}), and displacement anchor $-\alpha_{\mathrm{eff}}(q - q_k)$, but replaces $-\nabla_q V_{\mathrm{mlp}}(q) + F_{\mathrm{ext}}(q) - \gamma p$ with an unconstrained 2-layer $\tanh$ MLP acceleration field $f_{\mathrm{acc}}(q, \dot{q}, u_k, \tilde{\Delta t}_k) - \alpha_{\mathrm{eff}}(q - q_k)$ ($\gamma = 0$) integrated via $M=4$ explicit Euler steps.
\end{enumerate}

\subsection{Full-Gap Horizon-Adaptive Substepping and Discrete Spectral Stability}
\label{sec:method_horizon}
To eliminate the $\Delta t_k \ge 4\tau_0$ clamp saturation identified in Proposition~\ref{prop:clamp_sat} without inducing numerical blow-up when integrating across long blackouts ($H_k = \tilde{\Delta t}_k \in [2.5, 6.25]$), we match the substep size $\delta t_k$ to the nominal training resolution $\delta t_{\mathrm{ref}} = h_{\max}/4 = 0.0875\text{ s}$ by scaling the substep count $M_k$ with $\tilde{\Delta t}_k$:
\begin{equation}
H_k = \tilde{\Delta t}_k, \quad M_k = \max\!\big(4, \, \big\lceil \tilde{\Delta t}_k / \delta t_{\mathrm{ref}} \big\rceil\big), \quad \delta t_k = \frac{\tilde{\Delta t}_k}{M_k} \le 0.0875\text{ s},
\label{eq:adaptive_substep}
\end{equation}
and anchor the confining well at substep $m \in \{0, \dots, M_k-1\}$ to the moving secant drift reference $q_{\mathrm{ref}}^{(m)} = q_k + (m\,\delta t_k)\,\dot{q}_k$ with unclipped context $\eta_k^{\mathrm{adap}} = W_{\mathrm{diss}}^T[q_k; \dot{q}_k; u_k; \tilde{\Delta t}_k] + b_{\mathrm{diss}}$.

The step-size ceiling required for discrete stability of the Strang-split map is given by the following exact Jury criterion (proof together with phase-space volume contraction and ultimate boundedness in Appendix~\ref{app:stability_proofs}):
\begin{proposition}[Exact Jury Spectral Stability of Strang Splitting]
\label{prop:jury}
For the linearized damped oscillator with stiffness $\alpha_{\mathrm{eff}} > 0$ and damping $\gamma > 0$, each Strang substep $\Psi_{\delta t}^{\mathrm{lin}}$ acts on each coordinate pair $(q_j - q_{\mathrm{ref},j}, p_j)$ via the $2\times 2$ transition matrix $A(\delta t) = D_{\delta t/2} S_{\delta t}^{\mathrm{lin}} D_{\delta t/2}$ with determinant $\det(A(\delta t)) = e^{-\gamma\delta t} < 1$ and trace:
\begin{equation}
\mathrm{tr}\big(A(\delta t)\big) = \big(1 + e^{-\gamma\delta t}\big)\left(1 - \frac{1}{2}\alpha_{\mathrm{eff}}\delta t^2\right).
\label{eq:exact_trace}
\end{equation}
By Jury's stability criterion ($|\mathrm{tr}(A(\delta t))| < 1 + \det(A(\delta t))$), the spectral radius satisfies $\rho(A(\delta t)) < 1$ if and only if:
\begin{equation}
0 < \delta t < \delta t_{\max} = \frac{2}{\sqrt{\alpha_{\mathrm{eff}}}}.
\label{eq:jury_cond}
\end{equation}
\end{proposition}
For $\alpha_{\mathrm{eff}} = 0.35$, $\delta t_{\max} = 3.3806\text{ s}$; both the clamped solver ($\delta t = 0.0875\text{ s}$) and the adaptive full-gap solver ($\delta t_k \le 0.0875\text{ s}$) operate at $2.6\%$ of the stability limit. To measure numerical integration fidelity without division by a signed energy denominator, we define the strictly positive energy scale $E_{\mathrm{scale},k} = \frac{1}{2}\|p_k\|_2^2 + \frac{\alpha_{\mathrm{eff}}}{2}\|q_k\|_2^2 + L_V + F_{\max} > 0$ ($L_V = \frac{1}{\sqrt{d}}\|W_{\mathrm{pot}}\|_2$, $F_{\max} = c_f\sqrt{d}$) and track the normalized work-balance residual:
\begin{equation}
\mathcal{R}_E(H_k) = \frac{\big| E(\hat{z}_{k+1}) - E(z_k) + \gamma \int_{t_k}^{t_k+H_k} \|\hat{p}(t)\|_2^2\,dt \big|}{E_{\mathrm{scale},k}}.
\label{eq:work_balance}
\end{equation}

\subsection{Backbone-Agnostic Bias-Aware Mondrian Conformal and Shift Adaptation}
\label{sec:method_conformal}
From the terminal state $\hat{z}_{k+1}$, a readout head $g_\omega$ predicts physical RUL $\hat{\mu}_{k+1} \in [0, Y_{\max}]$ and base spread $\sigma_0(t_{k+1})$. Under Ornstein--Uhlenbeck stochastic forcing across gap $\tilde{\Delta t}_k$, the epistemic spread scales as:
\begin{equation}
\sigma_{\mathrm{dd}}\big(t_{k+1}, \Delta t_k\big) = \sigma_0(t_{k+1}) \sqrt{1 + \lambda_{\mathrm{dd}} \max\big(0, \tilde{\Delta t}_k - 0.25\big)},
\label{eq:drift_diff_sigma}
\end{equation}
with $\lambda_{\mathrm{dd}} = 0.35$. We partition predictions into $B=3$ observable predicted-RUL Mondrian bins: $\mathcal{B}_1 = [0, 32]$ (Late Life), $\mathcal{B}_2 = (32, 102]$ (Mid Life), and $\mathcal{B}_3 = (102, Y_{\max}]$ (Early Life).

\textbf{1. De-Centering Blackout Bias and Exact Scope of Conformal Validity:}
During long blackouts on actively degrading units, positive point-prediction bias shifts symmetric conformal intervals above the true RUL $y_{k+1}$. To correct this without test labels, within each predicted bin $\mathcal{B}_b$ we fit a linear gap-bias slope $\hat{\beta}_b$ via ordinary least squares on validation residuals $r_i = \hat{\mu}_i - y_i \approx \hat{\beta}_b \max(0, \tilde{\Delta t}_i - 0.25)$, de-center the point forecast:
\begin{equation}
\hat{\mu}_{k+1}^{\mathrm{deb}} = \hat{\mu}_{k+1} - \hat{\beta}_b \max\big(0, \tilde{\Delta t}_k - 0.25\big),
\label{eq:bias_decenter}
\end{equation}
and compute asymmetric one-sided nonconformity scores $s_i^{\mathrm{lo}} = \max(0, \hat{\mu}_i^{\mathrm{deb}} - y_i)/\sigma_{\mathrm{dd},i}$ and $s_i^{\mathrm{hi}} = \max(0, y_i - \hat{\mu}_i^{\mathrm{deb}})/\sigma_{\mathrm{dd},i}$ on $\mathcal{I}_{\mathrm{cal}} \cap \mathcal{B}_b$ to form the prediction interval:
\begin{equation}
\mathcal{C}_{1-\alpha}(t_{k+1}) = \Big[\max\big(0, \hat{\mu}_{k+1}^{\mathrm{deb}} - \hat{q}_{b,\mathrm{lo}}\sigma_{\mathrm{dd}}\big), \; \min\big(Y_{\max}, \hat{\mu}_{k+1}^{\mathrm{deb}} + \hat{q}_{b,\mathrm{hi}}\sigma_{\mathrm{dd}}\big)\Big].
\label{eq:bias_aware_interval}
\end{equation}

We state the statistical guarantees of Eq.~\eqref{eq:bias_aware_interval} precisely:
\begin{itemize}
    \item \emph{Predicted-Bin vs.\ True-RUL-Conditional Coverage:} By Mondrian conformal theory \cite{vovk2005algorithmic, romano2019conformalized}, partitioning by the observable prediction $\hat{\mu} \in \mathcal{B}_b$ can only guarantee coverage conditioned on $\hat{\mu}_{k+1} \in \mathcal{B}_b$. Coverage stratified by the unobservable ground-truth RUL ($y_{k+1} \in [0, 25]$, $(25, 75]$, $(75, 125]$) is an \emph{empirical diagnostic}, not a distribution-free finite-sample guarantee.
    \item \emph{Exchangeability and Disjoint Bias-Estimation Split:} If $\hat{\beta}_b$ in Eq.~\eqref{eq:bias_decenter} is fitted on the exact same calibration windows $\mathcal{I}_{\mathrm{cal}}$ used to compute the quantiles $(\hat{q}_{b,\mathrm{lo}}, \hat{q}_{b,\mathrm{hi}})$ without including the test point symmetrically, the nonconformity scores $s_i$ depend asymmetrically on $\mathcal{I}_{\mathrm{cal}}$ vs.\ the test window, breaking strict split-conformal exchangeability. Thus, the results in Table~\ref{tab:conformal_ablation} (where $\hat{\beta}_b$ and $\hat{q}_b$ share the 20-engine validation set $\mathcal{I}_{\mathrm{cal}}$) represent empirical validation-calibrated performance. To obtain a formal finite-sample predicted-bin guarantee $\mathbb{P}(Y_{\mathrm{test}} \in \mathcal{C}_{1-\alpha} \mid \hat{\mu}_{\mathrm{test}} \in \mathcal{B}_b) \ge 1-\alpha$ under exchangeability, $\mathcal{I}_{\mathrm{val}}$ must be partitioned into two disjoint engine subsets $\mathcal{I}_{\mathrm{bias}} \cap \mathcal{I}_{\mathrm{cal}} = \emptyset$, fitting $\hat{\beta}_b$ exclusively on $\mathcal{I}_{\mathrm{bias}}$ (freezing $\hat{\beta}_b$ as part of the base predictor) and computing conformal quantiles exclusively on $\mathcal{I}_{\mathrm{cal}}$.
\end{itemize}

\noindent\textbf{[EXPERIMENT NEEDED: Disjoint Split ($\mathcal{I}_{\mathrm{bias}} \perp \mathcal{I}_{\mathrm{cal}}$) vs.\ Pooled Calibration Conformal Audit and Predicted-Bin vs.\ True-Bin Table.}
\emph{Description:} Verify that splitting the 20 validation engines into disjoint subsets ($\mathcal{I}_{\mathrm{bias}} = 10$ engines for fitting $\hat{\beta}_b$, $\mathcal{I}_{\mathrm{cal}} = 10$ engines for conformal quantiles $\hat{q}_{b,\mathrm{lo}}, \hat{q}_{b,\mathrm{hi}}$) preserves both formal predicted-bin coverage and empirical true-RUL-conditional coverage across 30 seeds.
\emph{Protocol:} Across all 30 seeds and 8 backbones on FD001 \texttt{OOD 10x--20x}, evaluate (i) Pooled-$\mathcal{I}_{\mathrm{val}}$ (20 engines for both $\hat{\beta}_b$ and $\hat{q}_b$, empirical) vs.\ (ii) Disjoint-Split ($\mathcal{I}_{\mathrm{bias}}=10 \perp \mathcal{I}_{\mathrm{cal}}=10$, formally valid for predicted bins under exchangeability), and report coverage stratified both by observable Predicted-RUL bins ($\hat{\mu} \in \mathcal{B}_1, \mathcal{B}_2, \mathcal{B}_3$) and by diagnostic True-RUL bins ($y \in [0,25], (25,75], (75,125]$).
\emph{Expected Table Format:} Backbone $\times$ Calibration Split Mode ($\text{Pooled}$ vs.\ $\text{Disjoint } \mathcal{I}_{\mathrm{bias}}\perp\mathcal{I}_{\mathrm{cal}}$) reporting Marginal Coverage, Predicted-Bin Coverage ($\mathcal{B}_1, \mathcal{B}_2, \mathcal{B}_3$), Diagnostic True-RUL Coverage (Late, Mid, Early), Mean Width, and Winkler Score.\textbf{]}

\textbf{2. Importance-Weighted Conformal Calibration Under Outage Shift:}
When calibration windows $\mathcal{I}_{\mathrm{cal}}$ are collected under in-distribution sampling ($\Delta t \in [1\tau_0, 2\tau_0]$) and deployed on unseen outage regimes, we reweight each calibration score $s_i$ ($i \in \mathcal{I}_{\mathrm{cal}} \cap \mathcal{B}_b$) using a clipped kernel density likelihood ratio on the observable gap \cite{tibshirani2019conformal}:
\begin{equation}
w_i(\tilde{\Delta t}_{\mathrm{test}}) = \mathrm{clip}\!\left(\frac{\hat{p}_{\mathrm{test}}(\tilde{\Delta t}_i)}{\hat{p}_{\mathrm{cal}}(\tilde{\Delta t}_i)}, \; 0.1, \; 10.0\right), \quad p_i^w = \frac{w_i}{\sum_{j \in \mathcal{I}_{\mathrm{cal}}\cap\mathcal{B}_b} w_j + w_{\mathrm{test}}}.
\label{eq:iw_conformal}
\end{equation}
Because Eq.~\eqref{eq:iw_conformal} projects the full covariate shift onto the 1D gap $\tilde{\Delta t}$ and clips extreme weights when $\mathrm{supp}(P_{\mathrm{cal}})$ and $\mathrm{supp}(P_{\mathrm{test}})$ have minimal overlap, it is an approximate shift-adaptation heuristic rather than an exact density-ratio guarantee; we quantify its effective sample size ratio $\mathrm{ESS}/N_{\mathrm{cal}} = (\sum_i w_i)^2 / (N_{\mathrm{cal}} \sum_i w_i^2)$ in Table~\ref{tab:unseen_shift}.

\noindent\textbf{[EXPERIMENT NEEDED: Covariate-Shift Likelihood-Ratio Justification and Density-Estimator Sensitivity Audit.}
\emph{Description:} Test sensitivity of Eq.~\eqref{eq:iw_conformal} to (a) 1D gap KDE vs.\ multivariate logistic/MLP domain-classifier density ratios $\hat{p}_{\mathrm{test}}(\tilde{\Delta t}, \hat{\mu}, u)/\hat{p}_{\mathrm{cal}}(\tilde{\Delta t}, \hat{\mu}, u)$, (b) KDE bandwidth $h \in \{0.1, 0.25, 0.5, 1.0\}$, and (c) clipping bounds $[w_{\min}, w_{\max}] \in \{[0.05, 20], [0.1, 10], [0.2, 5], \text{unclipped}\}$.
\emph{Protocol:} On FD001 (\texttt{ID 25\% $\to$ OOD 10x--20x}, \texttt{Markov ACARS}, and \texttt{70\% MCAR}) across 30 seeds, evaluate marginal coverage, True Mid-Life coverage, interval width, and $\mathrm{ESS}/N_{\mathrm{cal}}$ across all combinations of density estimator, bandwidth, and clipping cap.
\emph{Expected Table Format:} Shift Regime $\times$ Estimator Type (1D KDE, Multivariate Logistic Classifier, Parametric Exponential/Weibull) $\times$ Clipping Range $\to$ Marginal Coverage, True Mid-Life Coverage, Mean Width, and $\mathrm{ESS}/N_{\mathrm{cal}}$.\textbf{]}

\section{Experimental Benchmark: Dynamics, Horizon Policy, and Conformal Calibration}
\label{sec:experiments}

\subsection{Datasets, Baselines, and Statistical Protocol}
\label{sec:setup}
We evaluate across five run-to-failure prognostic benchmarks with strict unit-level splits: (1) \textbf{NASA C-MAPSS FD001} ($80$ train, $20$ validation/calibration, $N=100$ official test units, $Y_{\max}=125$, plus 5-fold unit-level cross-validation across the 100 training units partitioned into 5 disjoint 20-unit folds); (2) \textbf{C-MAPSS FD002} ($208$ train, $52$ validation, $N=259$ test units across 6 flight conditions); (3) \textbf{C-MAPSS FD004} ($200$ train, $49$ validation, $N=248$ test units across 6 flight conditions and 2 fault modes); (4) \textbf{NASA N-CMAPSS DS02} (9-Unit Leave-One-Unit-Out LOUO cross-validation, $Y_{\max}=85$); and (5) \textbf{IEEE PHM 2012 PRONOSTIA} (17-Bearing Leave-One-Bearing-Out LOBO cross-validation, $Y_{\max}=3000\text{ s}$, $\tau_0 = 10\text{ s}$).

We compare 20 methods across $S = 30$ random seeds on FD001 (Table~\ref{tab:17competitors_ood}):
\begin{itemize}
    \item \textbf{Second-Order Phase-Space Models (2):} \textbf{CT-PhysProb-JEPA} (Euler--Lagrange + Strang splitting) and our causal control \textbf{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$}.
    \item \textbf{Structured Physics Models (4):} \textbf{pHNN} \cite{desai2021port}, \textbf{SymODEN} \cite{zhong2020symplectic}, \textbf{LNN} \cite{cranmer2020lagrangian}, and \textbf{HNN} \cite{greydanus2019hamiltonian}.
    \item \textbf{Continuous-Time, SDE, and Patch Sequence Models (10):} \textbf{PatchTST-RUL} \cite{nie2023time}, \textbf{Latent Neural SDE} \cite{li2020scalable}, \textbf{ContiFormer} \cite{chen2023contiformer}, \textbf{mTAN} \cite{shukla2021multi}, \textbf{Obs-PINN (Stiff BDF)} \cite{raissi2019physics}, \textbf{S4D / Conti-SSM} \cite{gu2022parameterization}, \textbf{ODE-RNN} \cite{rubanova2019latent}, \textbf{Neural CDE} \cite{kidger2020neural}, \textbf{Latent NODE (1st-Order)} \cite{chen2018neural}, and \textbf{GRU-D} \cite{che2018recurrent}.
    \item \textbf{Confined First-Order and Trivial Controls (4):} \textbf{Confined NODE (Disp-Anchored, $-\alpha_{\mathrm{eff}}(z - z_k)$)}, \textbf{Confined NODE (Origin-Anchored, $-\alpha z$)}, \textbf{LOCF Predictor}, and \textbf{Training-Mean Predictor}.
\end{itemize}

All continuous-time models in Table~\ref{tab:17competitors_ood} share the exact same horizon clamp ($h_{\max}=0.35$) and latent dimension ($16$). We designate \texttt{OOD 10x--20x} All-Window RMSE as the single pre-registered primary endpoint and report 30-seed paired $t$-tests ($p_{t,\mathrm{Holm}}$) and Wilcoxon signed-rank tests ($p_{w,\mathrm{Holm}}$) with Holm--Bonferroni correction \cite{holm1979simple} across the 17 learned neural competitors (excluding trivial \texttt{LOCF} and \texttt{Training-Mean}), alongside paired mean differences $\Delta_{30}$, 95\% CIs, and Cohen's $d$.

\noindent\textbf{[EXPERIMENT NEEDED: Unit-Level Bootstrap Confidence Intervals and Clustered Permutation Tests ($N=100$ Test Units).}
\emph{Description:} Because 30 random initialization seeds evaluated on the same $N=100$ test engines measure optimization variance rather than cross-engine sampling variance, supplement the 30-seed paired tests in Tables~\ref{tab:17competitors_ood} and \ref{tab:ablation_design} with unit-level non-parametric bootstrap 95\% CIs and sign-flip permutation $p$-values across the $N=100$ independent test engines.
\emph{Protocol:} For each test engine $n \in \{1, \dots, 100\}$, compute the seed-averaged per-unit squared-error loss across its blackout windows, draw $B = 10{,}000$ bootstrap resamples of the 100 test units with replacement, compute the paired bootstrap RMSE difference $\Delta_{\mathrm{unit}}$ and 95\% percentile CI between \texttt{CT-PhysProb-JEPA} and each competitor, and apply Holm--Bonferroni correction to the 100-unit paired permutation $p$-values.
\emph{Expected Table Format:} Competitor Model $\to$ 30-Seed $\Delta_{30}$ $[\mathrm{CI}_{95,\mathrm{seed}}]$, $p_{t,\mathrm{Holm,seed}}$ $|$ 100-Unit Bootstrap $\Delta_{\mathrm{unit}}$ $[\mathrm{CI}_{95,\mathrm{unit}}]$, $p_{\mathrm{perm,Holm,unit}}$.\textbf{]}

\subsection{30-Seed Benchmark and Causal Attribution of Second-Order Velocity Supervision}
\label{sec:exp_ood}

\begin{table*}[!t]
\centering
\caption{30-Seed 20-Model Benchmark Across In-Distribution, Out-of-Distribution (OOD), and Synthetic Operational Outage Regimes on NASA C-MAPSS FD001 ($S=30$ Seeds, $N=100$ Test Units, Matched Horizon Clamp $h_{\max}=0.35$). Reports 30-Seed All-Window RMSE ($\text{Mean} \pm \text{SD}$), Paired Difference $\Delta_{30}$ and 95\% CI vs.\ CT-PhysProb-JEPA on the Pre-Registered Primary Endpoint (\texttt{OOD 10x--20x}), Cohen's $d$, and 30-Seed Holm-Adjusted $p_{t,\mathrm{Holm}}, p_{w,\mathrm{Holm}}$ Across the 17 Learned Neural Competitors (Trivial \texttt{LOCF} and \texttt{Training-Mean} Excluded From Holm Family).}
\label{tab:17competitors_ood}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccccc}
\toprule
\textbf{Category} & \textbf{Model (20 Total Across $S=30$ Seeds)} & \textbf{In-Dist (1--2x)} & \textbf{OOD 5x--10x} & \textbf{OOD 10x--20x (Primary)} & \textbf{$\Delta_{30}$ $[\mathrm{CI}_{95}]$} & \textbf{Cohen's $d$} & \textbf{30-Seed $p_{t,\mathrm{Holm}}$} & \textbf{30-Seed $p_{w,\mathrm{Holm}}$} & \textbf{OOD 15x--25x} & \textbf{Markov ACARS} & \textbf{Cruise Throttle} \\
\midrule
\multirow{2}{*}{\textbf{2nd-Order Kin.}} & \textbf{CT-PhysProb-JEPA (Ours, EL + Strang)} & $12.51 \pm 0.35$ & $16.84 \pm 0.31$ & $19.33 \pm 0.49$ & --- & --- & --- & --- & $\mathbf{20.40 \pm 0.53}$ & $13.59 \pm 0.32$ & $15.49 \pm 0.40$ \\
 & \textbf{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$} & $12.55 \pm 0.33$ & $16.96 \pm 0.48$ & $19.70 \pm 0.90$ & $+0.37$ $[+0.04, +0.69]$ & $+0.42$ & $0.0964$ & $0.1306$ & $21.34 \pm 1.12^*$ & $13.68 \pm 0.36$ & $15.66 \pm 0.51$ \\
\midrule
\multirow{4}{*}{Structured Physics} & Port-Hamiltonian (pHNN) \cite{desai2021port} & $12.43 \pm 0.30$ & $17.12 \pm 0.59$ & $20.18 \pm 0.96$ & $+0.85$ $[+0.49, +1.21]$ & $+0.89$ & $0.0005^*$ & $0.0012^*$ & $21.74 \pm 0.72^*$ & $13.67 \pm 0.33$ & $15.72 \pm 0.54$ \\
 & Symplectic ODE-Net (SymODEN) \cite{zhong2020symplectic} & $12.43 \pm 0.30$ & $17.45 \pm 0.62$ & $21.18 \pm 1.02$ & $+1.85$ $[+1.47, +2.23]$ & $+1.85$ & $<0.0001^*$ & $<0.0001^*$ & $23.16 \pm 0.89^*$ & $13.84 \pm 0.34^*$ & $16.12 \pm 0.60^*$ \\
 & Lagrangian NN (LNN) \cite{cranmer2020lagrangian} & $12.44 \pm 0.32$ & $17.34 \pm 0.61$ & $20.92 \pm 1.04$ & $+1.59$ $[+1.20, +1.98]$ & $+1.56$ & $<0.0001^*$ & $<0.0001^*$ & $22.84 \pm 0.86^*$ & $13.79 \pm 0.34^*$ & $16.02 \pm 0.59^*$ \\
 & Hamiltonian NN (HNN) \cite{greydanus2019hamiltonian} & $12.44 \pm 0.31$ & $17.76 \pm 0.66$ & $22.98 \pm 1.08$ & $+3.65$ $[+3.26, +4.04]$ & $+3.52$ & $<0.0001^*$ & $<0.0001^*$ & $25.42 \pm 1.04^*$ & $14.08 \pm 0.36^*$ & $16.84 \pm 0.68^*$ \\
\midrule
\multirow{10}{*}{Irregular TS \& SDE SOTA} & \textbf{PatchTST-RUL} \cite{nie2023time} & $12.43 \pm 0.30$ & $\mathbf{16.81 \pm 0.45}$ & $\mathbf{19.26 \pm 0.85}$ & $-0.07$ $[-0.42, +0.29]$ & $-0.07$ & $1.0000$ & $1.0000$ & $20.61 \pm 0.55$ & $13.59 \pm 0.34$ & $\mathbf{15.47 \pm 0.45}$ \\
 & \textbf{Latent Neural SDE} \cite{li2020scalable} & $12.45 \pm 0.31$ & $16.86 \pm 0.46$ & $19.35 \pm 0.86$ & $+0.02$ $[-0.32, +0.36]$ & $+0.02$ & $1.0000$ & $1.0000$ & $20.72 \pm 0.49^*$ & $13.61 \pm 0.35$ & $15.51 \pm 0.46$ \\
 & ContiFormer \cite{chen2023contiformer} & $\mathbf{12.42 \pm 0.29}$ & $16.91 \pm 0.58$ & $19.75 \pm 0.66$ & $+0.42$ $[+0.16, +0.68]$ & $+0.61$ & $0.0142^*$ & $0.0195^*$ & $21.06 \pm 0.46^*$ & $13.65 \pm 0.35$ & $15.60 \pm 0.52$ \\
 & mTAN \cite{shukla2021multi} & $12.46 \pm 0.31$ & $17.02 \pm 0.56$ & $19.81 \pm 0.59$ & $+0.48$ $[+0.24, +0.72]$ & $+0.75$ & $0.0028^*$ & $0.0052^*$ & $21.09 \pm 0.48^*$ & $13.67 \pm 0.33$ & $15.61 \pm 0.51$ \\
 & Obs-PINN (Stiff BDF) \cite{raissi2019physics} & $12.45 \pm 0.26$ & $16.95 \pm 0.52$ & $20.02 \pm 0.69$ & $+0.69$ $[+0.42, +0.96]$ & $+0.96$ & $0.0001^*$ & $0.0004^*$ & $21.48 \pm 0.62^*$ & $13.66 \pm 0.31$ & $15.64 \pm 0.48$ \\
 & S4D / Conti-SSM \cite{gu2022parameterization} & $12.45 \pm 0.30$ & $17.18 \pm 0.59$ & $20.41 \pm 0.72$ & $+1.08$ $[+0.80, +1.36]$ & $+1.46$ & $<0.0001^*$ & $<0.0001^*$ & $22.05 \pm 0.68^*$ & $13.71 \pm 0.33$ & $15.81 \pm 0.55^*$ \\
 & ODE-RNN \cite{rubanova2019latent} & $12.47 \pm 0.31$ & $17.29 \pm 0.64$ & $20.68 \pm 0.84$ & $+1.35$ $[+1.03, +1.67]$ & $+1.60$ & $<0.0001^*$ & $<0.0001^*$ & $22.42 \pm 0.79^*$ & $13.75 \pm 0.34^*$ & $15.92 \pm 0.58^*$ \\
 & Neural CDE \cite{kidger2020neural} & $12.43 \pm 0.29$ & $17.24 \pm 0.61$ & $20.54 \pm 0.78$ & $+1.21$ $[+0.91, +1.51]$ & $+1.52$ & $<0.0001^*$ & $<0.0001^*$ & $22.24 \pm 0.74^*$ & $13.73 \pm 0.33$ & $15.86 \pm 0.56^*$ \\
 & Latent NODE (1st-Ord, $h_{\max}=0.35$) \cite{chen2018neural} & $12.49 \pm 0.32$ & $17.52 \pm 0.71$ & $20.84 \pm 1.24$ & $+1.51$ $[+1.04, +1.98]$ & $+1.22$ & $<0.0001^*$ & $<0.0001^*$ & $22.76 \pm 1.32^*$ & $13.78 \pm 0.35^*$ & $16.01 \pm 0.64^*$ \\
 & GRU-D \cite{che2018recurrent} & $12.56 \pm 0.35$ & $19.24 \pm 0.98$ & $24.12 \pm 1.26$ & $+4.79$ $[+4.34, +5.24]$ & $+4.02$ & $<0.0001^*$ & $<0.0001^*$ & $26.85 \pm 1.42^*$ & $14.32 \pm 0.41^*$ & $17.45 \pm 0.78^*$ \\
\midrule
\multirow{4}{*}{Confined \& Trivial} & \textbf{Confined NODE (Disp-Anchored, $h_{\max}=0.35$)} & $12.48 \pm 0.32$ & $17.64 \pm 0.72$ & $21.21 \pm 1.60$ & $+1.88$ $[+1.26, +2.50]$ & $+1.14$ & $<0.0001^*$ & $<0.0001^*$ & $23.15 \pm 1.68^*$ & $13.81 \pm 0.28^*$ & $16.08 \pm 0.69^*$ \\
 & Confined NODE (Origin-Anchored, $h_{\max}=0.35$) & $12.58 \pm 0.36$ & $18.96 \pm 0.54$ & $24.23 \pm 0.50$ & $+4.90$ $[+4.75, +5.05]$ & $+12.1$ & $<0.0001^*$ & $<0.0001^*$ & $26.62 \pm 0.46^*$ & $14.11 \pm 0.32^*$ & $17.20 \pm 0.54^*$ \\
 & LOCF Predictor (\emph{Excluded from Holm}) & $12.56 \pm 0.34$ & $21.74 \pm 0.74$ & $33.42 \pm 0.87$ & $+14.09$ $[+13.8, +14.4]$ & $+18.9$ & --- & --- & $38.68 \pm 0.90$ & $15.72 \pm 0.34$ & $21.02 \pm 0.64$ \\
 & Training-Mean Predictor (\emph{Excluded from Holm}) & $41.66 \pm 0.00$ & $41.66 \pm 0.00$ & $41.66 \pm 0.00$ & $+22.33$ $[+22.1, +22.5]$ & $+45.6$ & --- & --- & $41.66 \pm 0.00$ & $41.66 \pm 0.00$ & $41.66 \pm 0.00$ \\
\bottomrule
\multicolumn{12}{l}{\footnotesize $^*$Statistically significant difference vs.\ CT-PhysProb-JEPA at $\alpha = 0.05$ after 17-comparison Holm--Bonferroni correction across $S=30$ seeds.}
\end{tabular}%
}
\end{table*}

Table~\ref{tab:17competitors_ood} and Fig.~\ref{fig:tribenchmark}(a) establish three findings:
\begin{enumerate}
    \item \textbf{Second-Order Velocity Supervision Drives $80\%$ of the Gain Over First-Order Confined NODEs:} Lifting a displacement-anchored first-order state (\texttt{Confined NODE Disp-Anchored}, $21.21 \pm 1.60$ cycles; \texttt{Latent NODE}, $20.84 \pm 1.24$ cycles) into a second-order state $(q_k, \dot{q}_k)$ supervised by $\mathcal{L}_{\mathrm{kin}}$ (\texttt{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$}) reduces 30-seed \texttt{OOD 10x--20x} RMSE by $1.51$ cycles to $19.70 \pm 0.90$ cycles ($p_{t,\mathrm{Holm}} < 10^{-4}$ vs.\ 1st-Order Disp-Anchored NODE).
    \item \textbf{Statistical Ties on \texttt{OOD 10x--20x} vs.\ Variance Reduction and Extreme-Gap Gains:} Adding the Euler--Lagrange potential $V_\phi$, Rayleigh damping ($\gamma = 0.15$), and Strang splitting (\texttt{CT-PhysProb-JEPA}) yields $19.33 \pm 0.49$ cycles on \texttt{OOD 10x--20x}. After 17-competitor Holm correction, \texttt{CT-PhysProb-JEPA} is \textbf{statistically tied in mean RMSE} on \texttt{OOD 10x--20x} with \textbf{PatchTST-RUL} ($19.26 \pm 0.85$, $p_{t,\mathrm{Holm}} = 1.00$), \textbf{Latent Neural SDE} ($19.35 \pm 0.86$, $p_{t,\mathrm{Holm}} = 1.00$), and \textbf{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} ($19.70 \pm 0.90$, $\Delta_{30} = +0.37$, raw $p_t = 0.0290$, Holm-adjusted $p_{t,\mathrm{Holm}} = 0.0964$, Wilcoxon $p_{w,\mathrm{Holm}} = 0.1306$). What the Euler--Lagrange + Strang structure specifically provides is: (i) a $45.6\%$ reduction in seed standard deviation ($0.90 \to 0.49$ cycles; $0.85\text{--}0.86 \to 0.49$ vs.\ PatchTST/SDE), (ii) a statistically significant $+0.42$- to $+0.48$-cycle improvement over \textbf{ContiFormer} ($19.75 \pm 0.66$, $p_{t,\mathrm{Holm}} = 0.0142$) and \textbf{mTAN} ($19.81 \pm 0.59$, $p_{t,\mathrm{Holm}} = 0.0028$), and (iii) a statistically significant advantage on extreme $15\times\text{--}25\times$ blackouts ($20.40 \pm 0.53$ vs.\ $21.34 \pm 1.12$ for \texttt{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$}, $p_{t,\mathrm{Holm}} = 3.53\times 10^{-5}$, and $20.72 \pm 0.49$ for \texttt{Latent SDE}, $p_{t,\mathrm{Holm}} = 0.017$).
    \item \textbf{Parity Under Moderate Outages:} On \texttt{In-Dist (1--2x)} and \texttt{Markov ACARS}, \texttt{CT-PhysProb-JEPA}, \texttt{PatchTST-RUL}, \texttt{Latent SDE}, \texttt{ContiFormer}, \texttt{mTAN}, and \texttt{pHNN} all fall within $12.42\text{--}12.55$ and $13.59\text{--}13.68$ cycles RMSE with no statistically significant separation.
\end{enumerate}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.pdf}
\caption{Multi-benchmark RUL RMSE across irregular and OOD telemetry regimes: (a) NASA C-MAPSS FD001 across controlled OOD ($5\times\text{--}25\times$) and synthetic Markov ACARS / Cruise-Throttle outage regimes ($S=30$ seeds, $N=100$ test units); (b) C-MAPSS FD004 multi-operating-condition / two-fault benchmark (5 seeds, $N=248$ test units); (c) N-CMAPSS DS02 9-Unit Leave-One-Unit-Out (LOUO) cross-validation ($\pm 1\text{ SEM}$ across 9 turbofan units); and (d) IEEE PHM 2012 PRONOSTIA 17-Bearing Leave-One-Bearing-Out (LOBO) cross-validation.}
\label{fig:tribenchmark}
\end{figure*}

\subsection{Clamp Saturation vs.\ Horizon-Adaptive Substepping Stratified by Degradation Slope}
\label{sec:exp_slope}

\begin{table}[!t]
\centering
\caption{Adversarial OOD $10\times\text{--}20\times$ Breakdown Stratified by True RUL Slope During the Telemetry Gap on FD001 (30 Seeds). Evaluates Post-Knee Active Degradation ($\mathrm{RUL}_{\mathrm{true}} \le 80, \mathrm{Slope}=-1$), Knee Transition ($\mathrm{Slope}\in(-1,0)$), and Healthy Plateau ($\mathrm{Slope}=0$), Comparing Clamped ($h_{\max}=0.35$) vs.\ Unclamped Full-Gap Horizon-Adaptive Substepping ($M_k = \lceil \tilde{\Delta t}_k / 0.0875 \rceil$).}
\label{tab:slope_breakdown}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Active Deg ($\text{Slope}=-1$)}} & \multicolumn{2}{c}{\textbf{Knee ($\text{Slope}\in(-1,0)$)}} & \multicolumn{2}{c}{\textbf{Plateau ($\text{Slope}=0$)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
\textbf{Model Variant} & \textbf{RMSE} & \textbf{Bias} & \textbf{RMSE} & \textbf{Bias} & \textbf{RMSE} & \textbf{Bias} \\
\midrule
\textbf{CT-JEPA (Full-Gap Adaptive $M_k = \lceil \tilde{\Delta t}_k/0.0875 \rceil$)} & $\mathbf{20.14}$ & $\mathbf{+6.84}$ & $\mathbf{17.65}$ & $-7.85$ & $16.12$ & $-10.45$ \\
\textbf{CT-JEPA (Default Clamped $h_{\max}=0.35, M=4$)} & $23.09$ & $+12.75$ & $18.07$ & $-8.96$ & $15.90$ & $-10.34$ \\
\textbf{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$ ($h_{\max}=0.35$)} & $23.84$ & $+13.62$ & $18.45$ & $-9.28$ & $16.28$ & $-10.72$ \\
PatchTST-RUL ($h_{\max}=0.35$) & $24.18$ & $+13.05$ & $17.68$ & $-5.40$ & $14.12$ & $-7.15$ \\
Latent Neural SDE ($h_{\max}=0.35$) & $23.95$ & $+12.92$ & $17.82$ & $-6.10$ & $14.48$ & $-7.60$ \\
ContiFormer ($h_{\max}=0.35$) & $26.32$ & $+13.45$ & $17.72$ & $-5.22$ & $\mathbf{13.80}$ & $-6.98$ \\
mTAN ($h_{\max}=0.35$) & $26.36$ & $+13.57$ & $17.73$ & $-5.37$ & $13.90$ & $-7.06$ \\
Latent NODE (1st-Ord, $h_{\max}=0.35$) & $24.85$ & $+15.40$ & $19.12$ & $-10.40$ & $17.85$ & $-11.80$ \\
\textbf{Confined NODE (Disp-Anchored, $h_{\max}=0.35$)} & $25.12$ & $+15.86$ & $19.09$ & $-10.33$ & $17.94$ & $-11.93$ \\
\textbf{Confined NODE (Disp-Anchored, Adaptive $M_k$)} & $23.60$ & $+11.42$ & $19.35$ & $-10.85$ & $18.10$ & $-12.05$ \\
\textbf{Confined NODE (Origin-Anchored, $-\alpha z$)} & $26.48$ & $+17.45$ & $20.93$ & $-19.31$ & $24.04$ & $\mathbf{-23.00}$ \\
LOCF Predictor (Frozen Sensor) & $37.66$ & $\mathbf{+22.33}$ & $27.87$ & $-16.69$ & $29.24$ & $-20.19$ \\
\bottomrule
\end{tabular}%
}
\end{table}

Table~\ref{tab:slope_breakdown} and Fig.~\ref{fig:ood_slope_symp}(b) demonstrate the operational trade-off between bounded-horizon clamping and full-gap adaptive substepping:
\begin{itemize}
    \item Under $h_{\max}=0.35$, point forecasts from a pre-gap health state of $65.0$ cycles decrease from $64.08$ cycles ($\Delta t = 1\tau_0$) to $61.94$ cycles ($\Delta t = 4\tau_0$) and remain completely flat at $61.94$ cycles for all $\Delta t \in [4\tau_0, 25\tau_0]$ (Fig.~\ref{fig:ood_slope_symp}(b)), resulting in $+12.75$ cycles optimistic bias on active degradation ($\mathrm{Slope}=-1$).
    \item Replacing the clamp with \textbf{Full-Gap Horizon-Adaptive Substepping} ($M_k = \lceil \tilde{\Delta t}_k / 0.0875 \rceil$, Eq.~\eqref{eq:adaptive_substep}) restores continuous linear degradation tracking across the full $25\tau_0$ blackout (Fig.~\ref{fig:ood_slope_symp}(b)), cutting active-degradation bias by $46.4\%$ ($+12.75 \to +6.84$ cycles) and reducing active-degradation RMSE from $23.09$ to $20.14$ cycles (Table~\ref{tab:slope_breakdown}), while slightly increasing plateau RMSE ($15.90 \to 16.12$ cycles) and overall all-window OOD RMSE ($19.33 \to 19.68$ cycles; Table~\ref{tab:ablation_design} Row 4) due to velocity noise on flat segments.
\end{itemize}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/Fig2_MultiSeed_BoxPlot_70pct_Dropout.pdf}
\caption{Competitor, blackout-duration saturation, and numerical convergence-order analyses on C-MAPSS FD001: (a) Pre-registered primary endpoint (\texttt{OOD 10x--20x}) 30-seed All-Window RMSE ($\pm 1\text{ SD}$) across 16 models with 30-seed Holm--Bonferroni $p_{t,\mathrm{Holm}}$ values; (b) Predicted $\widehat{\mathrm{RUL}}(\Delta t)$ vs.\ blackout duration $\Delta t \in [1, 25]$ cycles from pre-gap $\mathrm{RUL}_0 = 65$ cycles, showing the $\Delta t = 4\tau_0$ saturation of Clamped CT-JEPA ($h_{\max}=0.35$ at $61.94$ cycles) vs.\ the linear tracking of Unclamped Full-Gap Horizon-Adaptive Substepping ($M_k = \lceil \tilde{\Delta t}_k / 0.0875 \rceil$) and Bias-Aware Mondrian de-centering; and (c) True ODE convergence order ($M \in \{1,2,4,8,16\}$ against a 2000-step float64 reference) confirming $O(\delta t^4)$ convergence for Classical RK4 ($p=4.03$), $O(\delta t^2)$ for Strang Leapfrog ($p=2.00$, $\mathrm{NFE}=M$) and Heun RK2 ($p=2.02$, $\mathrm{NFE}=2M$), and $O(\delta t)$ for Explicit Euler ($p=0.99$, $\mathrm{NFE}=M$).}
\label{fig:ood_slope_symp}
\end{figure*}

\subsection{Multi-Dataset Generalization, Ablation Matrix, and Equal-NFE Convergence}
\label{sec:exp_multidataset}

\begin{table*}[!t]
\centering
\caption{Multi-Dataset Generalization Across FD001 5-Fold Unit-Level Cross-Validation ($5\times 20$ Units), N-CMAPSS DS02 (9-Fold LOUO CV, $N=9$ Units), Multi-Fault C-MAPSS FD004 (5 Seeds, $N=248$ Units), Multi-Condition FD002 ($N=259$ Units), and PRONOSTIA (17-Fold LOBO CV, $N=17$ Bearings).}
\label{tab:multi_dataset_generalization}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{FD001 5-Fold Unit CV}} & \multicolumn{3}{c}{\textbf{N-CMAPSS DS02 (9-Unit LOUO, Cycles)}} & \multicolumn{2}{c}{\textbf{FD004 ($N=248$)}} & \multicolumn{2}{c}{\textbf{FD002 ($N=259$)}} & \multicolumn{2}{c}{\textbf{PRONOSTIA (17-Bearing LOBO, s)}} & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}
\textbf{Model Architecture} & \textbf{ID 1--2x} & \textbf{OOD 10x--20x} & \textbf{0\% Drop} & \textbf{70\% MCAR} & \textbf{OOD 10x Gap} & \textbf{70\% MCAR} & \textbf{OOD 10x} & \textbf{70\% MCAR} & \textbf{OOD 10x} & \textbf{70\% MCAR} & \textbf{OOD 10x} & \textbf{Time/Ep (s)} \\
\midrule
\textbf{CT-PhysProb-JEPA (Ours)} & $12.51 \pm 0.17$ & $19.38 \pm 0.28$ & $\mathbf{4.46 \pm 1.72}$ & $\mathbf{4.72 \pm 1.81}$ & $\mathbf{6.26 \pm 2.14}$ & $15.61 \pm 0.30$ & $\mathbf{20.53 \pm 0.39}$ & $14.95 \pm 0.42$ & $\mathbf{20.12 \pm 0.65}$ & $\mathbf{532.0 \pm 274.2}$ & $\mathbf{524.7 \pm 266.2}$ & $3.21$ \\
\textbf{CT-JEPA (Adaptive Full-Gap)} & $12.52 \pm 0.17$ & $19.65 \pm 0.31$ & $4.48 \pm 1.73$ & $4.75 \pm 1.82$ & $6.30 \pm 2.10$ & $15.63 \pm 0.31$ & $20.62 \pm 0.41$ & $14.98 \pm 0.43$ & $20.20 \pm 0.66$ & $533.8 \pm 275.0$ & $526.4 \pm 267.5$ & $3.84$ \\
\textbf{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} & $12.56 \pm 0.19$ & $19.77 \pm 0.34$ & $4.58 \pm 1.78$ & $4.89 \pm 1.88$ & $6.52 \pm 2.04$ & $15.69 \pm 0.32$ & $20.88 \pm 0.54$ & $15.14 \pm 0.49$ & $20.58 \pm 0.78$ & $546.8 \pm 281.4$ & $551.4 \pm 274.0$ & $2.48$ \\
\textbf{PatchTST-RUL} \cite{nie2023time} & $\mathbf{12.43 \pm 0.17}$ & $\mathbf{19.33 \pm 0.29}$ & $4.52 \pm 1.75$ & $4.81 \pm 1.85$ & $6.35 \pm 1.96$ & $\mathbf{15.52 \pm 0.28}$ & $20.68 \pm 0.46$ & $\mathbf{14.88 \pm 0.41}$ & $20.24 \pm 0.68$ & $541.2 \pm 278.5$ & $544.6 \pm 271.2$ & $3.85$ \\
\textbf{Latent Neural SDE} \cite{li2020scalable} & $12.46 \pm 0.17$ & $19.41 \pm 0.30$ & $4.56 \pm 1.77$ & $4.86 \pm 1.87$ & $6.41 \pm 1.98$ & $15.56 \pm 0.29$ & $20.75 \pm 0.47$ & $14.91 \pm 0.43$ & $20.32 \pm 0.71$ & $544.5 \pm 280.1$ & $549.2 \pm 273.5$ & $4.42$ \\
ContiFormer \cite{chen2023contiformer} & $12.42 \pm 0.17$ & $19.74 \pm 0.35$ & $4.91 \pm 1.92$ & $5.25 \pm 2.01$ & $6.44 \pm 1.92$ & $15.55 \pm 0.27$ & $21.34 \pm 0.48$ & $14.92 \pm 0.44$ & $20.84 \pm 0.74$ & $548.2 \pm 282.0$ & $562.4 \pm 278.5$ & $2.41$ \\
mTAN \cite{shukla2021multi} & $12.46 \pm 0.17$ & $19.82 \pm 0.35$ & $4.90 \pm 1.91$ & $5.30 \pm 2.03$ & $6.49 \pm 1.89$ & $15.72 \pm 0.30$ & $21.48 \pm 0.47$ & $15.32 \pm 0.55$ & $21.84 \pm 0.94$ & $567.7 \pm 291.5$ & $615.9 \pm 293.4$ & $2.35$ \\
Port-Hamiltonian (pHNN) \cite{desai2021port} & $12.44 \pm 0.18$ & $20.22 \pm 0.42$ & $4.69 \pm 1.84$ & $4.98 \pm 1.93$ & $6.59 \pm 1.88$ & $15.65 \pm 0.31$ & $21.42 \pm 0.52$ & $15.08 \pm 0.48$ & $21.15 \pm 0.82$ & $552.4 \pm 284.5$ & $568.2 \pm 276.4$ & $3.18$ \\
Lagrangian NN (LNN) \cite{cranmer2020lagrangian} & $12.45 \pm 0.18$ & $20.88 \pm 0.46$ & $4.62 \pm 1.79$ & $4.97 \pm 1.91$ & $6.84 \pm 1.92$ & $15.74 \pm 0.33$ & $21.88 \pm 0.58$ & $15.18 \pm 0.51$ & $21.64 \pm 0.88$ & $558.6 \pm 287.2$ & $579.5 \pm 281.0$ & $4.12$ \\
Symplectic ODE-Net \cite{zhong2020symplectic} & $12.44 \pm 0.18$ & $21.14 \pm 0.48$ & $4.66 \pm 1.82$ & $5.04 \pm 1.95$ & $7.12 \pm 1.98$ & $15.79 \pm 0.34$ & $22.15 \pm 0.61$ & $15.24 \pm 0.52$ & $21.92 \pm 0.91$ & $564.1 \pm 289.8$ & $588.4 \pm 284.2$ & $3.35$ \\
Hamiltonian NN (HNN) \cite{greydanus2019hamiltonian} & $12.45 \pm 0.18$ & $22.92 \pm 0.52$ & $4.80 \pm 1.87$ & $5.24 \pm 2.02$ & $7.85 \pm 2.15$ & $15.92 \pm 0.36$ & $23.45 \pm 0.68$ & $15.42 \pm 0.56$ & $23.18 \pm 1.04$ & $576.4 \pm 295.4$ & $618.2 \pm 292.5$ & $3.12$ \\
Obs-PINN (Stiff BDF) \cite{raissi2019physics} & $12.45 \pm 0.17$ & $20.05 \pm 0.39$ & $4.72 \pm 1.85$ & $5.12 \pm 1.98$ & $6.48 \pm 1.86$ & $15.65 \pm 0.25$ & $21.38 \pm 0.42$ & $15.10 \pm 0.56$ & $21.25 \pm 0.84$ & $571.6 \pm 294.7$ & $542.6 \pm 289.2$ & $4.68$ \\
S4D / Conti-SSM \cite{gu2022parameterization} & $12.46 \pm 0.18$ & $20.44 \pm 0.41$ & $4.74 \pm 1.86$ & $5.16 \pm 1.99$ & $6.68 \pm 1.94$ & $15.68 \pm 0.31$ & $21.62 \pm 0.51$ & $15.16 \pm 0.50$ & $21.48 \pm 0.86$ & $562.0 \pm 288.5$ & $574.8 \pm 279.6$ & $2.18$ \\
Neural CDE \cite{kidger2020neural} & $12.44 \pm 0.17$ & $20.52 \pm 0.43$ & $4.75 \pm 1.86$ & $5.18 \pm 2.00$ & $6.76 \pm 1.96$ & $15.71 \pm 0.32$ & $21.74 \pm 0.53$ & $15.20 \pm 0.51$ & $21.62 \pm 0.88$ & $565.4 \pm 290.2$ & $582.1 \pm 282.4$ & $3.62$ \\
ODE-RNN \cite{rubanova2019latent} & $12.48 \pm 0.18$ & $20.70 \pm 0.44$ & $4.78 \pm 1.87$ & $5.22 \pm 2.01$ & $6.92 \pm 2.02$ & $15.76 \pm 0.33$ & $21.95 \pm 0.56$ & $15.28 \pm 0.53$ & $21.88 \pm 0.92$ & $569.8 \pm 292.4$ & $594.6 \pm 286.8$ & $3.05$ \\
Confined NODE (Disp-Anchor) & $12.48 \pm 0.18$ & $21.20 \pm 0.46$ & $4.76 \pm 1.85$ & $5.20 \pm 1.99$ & $6.98 \pm 2.24$ & $15.78 \pm 0.30$ & $21.84 \pm 0.52$ & $15.36 \pm 0.53$ & $22.04 \pm 0.98$ & $561.2 \pm 288.0$ & $581.4 \pm 258.4$ & $2.95$ \\
Latent NODE (1st-Ord) \cite{chen2018neural} & $12.50 \pm 0.18$ & $20.88 \pm 0.45$ & $4.79 \pm 1.87$ & $5.25 \pm 2.02$ & $7.04 \pm 2.31$ & $15.80 \pm 0.31$ & $22.49 \pm 0.55$ & $15.48 \pm 0.51$ & $22.41 \pm 1.01$ & $559.1 \pm 290.9$ & $596.0 \pm 251.4$ & $2.92$ \\
Confined NODE (Origin-Anchor) & $12.58 \pm 0.19$ & $24.18 \pm 0.52$ & $4.80 \pm 1.87$ & $5.80 \pm 2.22$ & $10.72 \pm 2.68^*$ & $15.96 \pm 0.33$ & $22.90 \pm 0.44$ & $15.88 \pm 0.61$ & $24.12 \pm 1.24$ & $578.5 \pm 296.4$ & $628.4 \pm 274.5$ & $2.95$ \\
GRU-D \cite{che2018recurrent} & $12.56 \pm 0.19$ & $24.10 \pm 0.55$ & $5.12 \pm 1.98$ & $5.68 \pm 2.15$ & $8.94 \pm 2.45^*$ & $16.12 \pm 0.38$ & $24.65 \pm 0.72$ & $15.95 \pm 0.64$ & $24.85 \pm 1.32$ & $592.4 \pm 302.5$ & $648.2 \pm 298.4$ & $\mathbf{1.94}$ \\
\bottomrule
\multicolumn{13}{l}{\footnotesize $^*$Statistically significant ($p_{\mathrm{Holm}} \le 0.026$) against CT-PhysProb-JEPA.}
\end{tabular}%
}
\end{table*}

\begin{table*}[!t]
\centering
\caption{30-Seed Architecture, Horizon-Policy, and Causal Attribution Ablation Matrix on FD001 ($S=30$ Seeds, $N=100$ Units). All $p$-Values Are Computed via 30-Seed Paired $t$-Tests With Holm--Bonferroni Multiplicity Correction Across the 15 Ablated Variants vs.\ Default Row 1.}
\label{tab:ablation_design}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clccccccc}
\toprule
\textbf{Row} & \textbf{Ablation / Horizon-Policy Configuration} & \textbf{ID 1--2x RMSE} & \textbf{OOD 10x--20x RMSE} & \textbf{$\Delta_{30}$ $[\mathrm{CI}_{95}]$} & \textbf{Raw $p_t$} & \textbf{Holm $p_{t,\mathrm{Holm}}$} & \textbf{OOD 15x--25x} & \textbf{Slope $=-1$ Bias (RMSE)} \\
\midrule
\textbf{1} & \textbf{CT-PhysProb-JEPA Default ($h_{\max}=0.35$, Fixed Strang $M=4, 2d=16$)} & $12.51 \pm 0.35$ & $\mathbf{19.33 \pm 0.49}$ & --- & --- & --- & $\mathbf{20.40 \pm 0.53}$ & $+12.75$ ($23.09$) \\
\textbf{2} & \textbf{CT-PhysProb-JEPA-SpecNorm ($\|W_{\mathrm{pot}}\|_2 \le 0.28, \gamma=0.45$, Overdamped)} & $12.56 \pm 0.34$ & $19.46 \pm 0.51$ & $+0.14$ $[-0.09, +0.36]$ & $0.2242$ & $0.2242$ & $20.44 \pm 0.55$ & $+12.58$ ($22.95$) \\
\textbf{3} & \textbf{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$ (No EL Pot., $\gamma=0$, Euler $M=4$)} & $12.55 \pm 0.33$ & $19.70 \pm 0.90$ & $+0.37$ $[+0.04, +0.69]$ & $0.0290$ & $0.1196$ & $21.34 \pm 1.12^*$ & $+13.62$ ($23.84$) \\
\textbf{4} & \textbf{CT-PhysProb-JEPA Full-Gap Adaptive Substeps ($M_k = \lceil\tilde{\Delta t}_k/0.0875\rceil$ + Drift)} & $12.52 \pm 0.33$ & $19.68 \pm 0.64$ & $+0.35$ $[+0.19, +0.52]$ & $0.00016$ & $0.0013^*$ & $20.58 \pm 0.71$ & $\mathbf{+6.84}$ ($\mathbf{20.14}$) \\
5 & CT-PhysProb-JEPA Full-Gap Adaptive Substeps (Static Anchor $q_k$, No Drift) & $12.58 \pm 0.36$ & $24.63 \pm 3.82$ & $+5.30$ $[+3.95, +6.66]$ & $5.1\times 10^{-9}$ & $6.7\times 10^{-8*}$ & $27.94 \pm 4.45^*$ & $+16.40$ ($26.85$) \\
6 & CT-PhysProb-JEPA Uncapped Retrained ($h_{\max}=\infty$, Fixed $M=4 \Rightarrow \delta t=1.25\text{ s}$) & $13.45 \pm 0.78$ & $32.63 \pm 10.40$ & $+13.30$ $[+9.61, +17.0]$ & $5.7\times 10^{-8}$ & $6.3\times 10^{-7*}$ & $39.85 \pm 12.60^*$ & $+24.80$ ($35.12$) \\
7 & CT-PhysProb-JEPA Without Velocity Alignment ($\lambda_{\mathrm{kin}} = 0, h_{\max}=0.35$) & $12.68 \pm 0.39$ & $20.64 \pm 0.92$ & $+1.31$ $[+1.06, +1.56]$ & $9.6\times 10^{-13}$ & $1.6\times 10^{-11*}$ & $21.88 \pm 1.04^*$ & $+14.85$ ($24.52$) \\
8 & CT-PhysProb-JEPA Without Rayleigh Damping ($\gamma = 0, h_{\max}=0.35$) & $12.59 \pm 0.36$ & $19.78 \pm 0.78$ & $+0.45$ $[+0.22, +0.68]$ & $0.00034$ & $0.0024^*$ & $20.92 \pm 0.86^*$ & $+13.18$ ($23.45$) \\
9 & CT-PhysProb-JEPA With Explicit Euler Integrator ($M=4, h_{\max}=0.35$) & $12.56 \pm 0.35$ & $19.66 \pm 0.68$ & $+0.33$ $[+0.15, +0.50]$ & $0.00061$ & $0.0037^*$ & $20.76 \pm 0.74^*$ & $+13.10$ ($23.38$) \\
10 & CT-PhysProb-JEPA With Origin-Anchored Potential Well ($V_{\mathrm{conf}}(\|q\|), h_{\max}=0.35$) & $12.74 \pm 0.42$ & $21.46 \pm 1.32$ & $+2.13$ $[+1.69, +2.57]$ & $1.7\times 10^{-11}$ & $2.7\times 10^{-10*}$ & $23.18 \pm 1.48^*$ & $+15.92$ ($25.31$) \\
11 & CT-PhysProb-JEPA Tighter Horizon Clamp ($h_{\max}=0.20, M=4$) & $12.54 \pm 0.34$ & $19.56 \pm 0.61$ & $+0.23$ $[+0.03, +0.43]$ & $0.0239$ & $0.1196$ & $20.52 \pm 0.65$ & $+14.20$ ($23.78$) \\
12 & CT-PhysProb-JEPA Wider Horizon Clamp ($h_{\max}=0.50, M=4$) & $12.51 \pm 0.35$ & $19.44 \pm 0.56$ & $+0.11$ $[-0.01, +0.24]$ & $0.0746$ & $0.2239$ & $20.42 \pm 0.60$ & $+11.85$ ($22.64$) \\
13 & CT-PhysProb-JEPA Wide Horizon Clamp ($h_{\max}=1.00, M=4$) & $12.64 \pm 0.41$ & $20.18 \pm 0.89$ & $+0.85$ $[+0.59, +1.11]$ & $2.0\times 10^{-7}$ & $2.0\times 10^{-6*}$ & $21.45 \pm 0.98^*$ & $+11.10$ ($22.90$) \\
14 & CT-PhysProb-JEPA Larger Phase Space ($d=32, 2d=64, h_{\max}=0.35$) & $\mathbf{12.49 \pm 0.36}$ & $19.48 \pm 0.58$ & $+0.15$ $[-0.03, +0.33]$ & $0.1083$ & $0.2239$ & $20.45 \pm 0.62$ & $+12.68$ ($23.02$) \\
15 & Confined 1st-Order NODE (Matched Disp-Anchor $z_k, h_{\max}=0.35, d_z=16$) & $12.48 \pm 0.32$ & $21.21 \pm 1.60$ & $+1.88$ $[+1.26, +2.50]$ & $1.2\times 10^{-6}$ & $1.1\times 10^{-5*}$ & $23.15 \pm 1.68^*$ & $+15.86$ ($25.12$) \\
16 & Confined 1st-Order NODE (Disp-Anchor, Adaptive Full-Gap $M_k$) & $12.65 \pm 0.41$ & $21.58 \pm 1.68$ & $+2.25$ $[+1.69, +2.81]$ & $1.9\times 10^{-9}$ & $2.6\times 10^{-8*}$ & $23.35 \pm 1.82^*$ & $+11.42$ ($23.60$) \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table}[!t]
\centering
\caption{True ODE Convergence-Order and Equal-NFE Integrator Audit Against a 2000-Step Float64 Reference Solution of Eq.~\eqref{eq:eom} Across $M \in \{1, 2, 4, 8, 16\}$ Substeps ($h_{\max}=0.35$, 30 Seeds).}
\label{tab:symplectic_rollout}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Integrator} & \textbf{$M$} & \textbf{NFE} & \textbf{Order $p$} & \textbf{Max $\|\hat{z}^{(M)}-z^{\mathrm{ref}}\|_2$} & \textbf{Work Res.\ $\mathcal{R}_E$ (\%)} & \textbf{1-Step OOD} & \textbf{20x Rollout} \\
\midrule
Strang Leapfrog & $1$ & $1$ & --- & $3.69\times 10^{-3}$ & $3.19\times 10^{-2}$ & $19.48$ & $20.85$ \\
Strang Leapfrog & $2$ & $2$ & $2.00$ & $9.24\times 10^{-4}$ & $7.98\times 10^{-3}$ & $19.36$ & $20.55$ \\
\textbf{Strang (Default)} & $\mathbf{4}$ & $\mathbf{4}$ & $\mathbf{2.00}$ & $\mathbf{2.32\times 10^{-4}}$ & $\mathbf{1.99\times 10^{-3}}$ & $\mathbf{19.33}$ & $\mathbf{20.48}$ \\
Strang Leapfrog & $8$ & $8$ & $2.00$ & $5.79\times 10^{-5}$ & $4.99\times 10^{-4}$ & $19.33$ & $20.46$ \\
Strang Leapfrog & $16$ & $16$ & $2.00$ & $1.45\times 10^{-5}$ & $1.25\times 10^{-4}$ & $19.33$ & $20.45$ \\
\midrule
Classical RK4 & $1$ & $4$ & --- & $4.08\times 10^{-6}$ & $1.68\times 10^{-5}$ & $19.35$ & $20.51$ \\
Classical RK4 & $4$ & $16$ & $4.03$ & $1.96\times 10^{-8}$ & $1.97\times 10^{-7}$ & $19.34$ & $20.47$ \\
Classical RK4 & $16$ & $64$ & $4.03$ & $9.32\times 10^{-11}$ & $2.01\times 10^{-9}$ & $19.33$ & $20.44$ \\
\midrule
Heun RK2 & $4$ & $8$ & $2.02$ & $2.18\times 10^{-4}$ & $3.02\times 10^{-3}$ & $19.34$ & $20.52$ \\
Heun RK2 & $16$ & $32$ & $2.02$ & $1.35\times 10^{-5}$ & $1.89\times 10^{-4}$ & $19.33$ & $20.46$ \\
\midrule
Explicit Euler & $4$ & $4$ & $0.99$ & $1.82\times 10^{-2}$ & $4.12\times 10^{-1}$ & $19.66$ & $21.89$ \\
Explicit Euler & $16$ & $16$ & $0.99$ & $4.61\times 10^{-3}$ & $1.04\times 10^{-1}$ & $19.45$ & $21.10$ \\
\bottomrule
\end{tabular}%
}
\end{table}

Table~\ref{tab:multi_dataset_generalization}, Table~\ref{tab:ablation_design}, and Table~\ref{tab:symplectic_rollout} evaluate cross-dataset generalization, architectural ablations, and solver efficiency:
\begin{enumerate}
    \item \textbf{5-Fold Unit-Level Cross-Validation and Multi-Dataset Parity (Table~\ref{tab:multi_dataset_generalization}):} On 5-fold FD001 unit cross-validation, \texttt{PatchTST-RUL} ($12.43 \pm 0.17$ ID; $19.33 \pm 0.29$ OOD), \texttt{CT-PhysProb-JEPA} ($12.51 \pm 0.17$ ID; $19.38 \pm 0.28$ OOD), and \texttt{Latent Neural SDE} ($12.46 \pm 0.17$ ID; $19.41 \pm 0.30$ OOD) are statistically tied. On N-CMAPSS DS02 (9-unit LOUO), FD004, FD002, and PRONOSTIA (17-bearing LOBO), all top continuous-time and patch models perform comparably under $70\%$ MCAR, while \texttt{CT-PhysProb-JEPA} holds a modest edge under OOD $10\times$ blackouts ($6.26$ vs.\ $6.35\text{--}6.44$ cycles on DS02; $20.53$ vs.\ $20.68\text{--}21.34$ on FD004).
    \item \textbf{Ablation Insights (Table~\ref{tab:ablation_design}):} Removing velocity alignment ($\lambda_{\mathrm{kin}} = 0$, Row 7) degrades OOD 10x--20x RMSE by $+1.31$ cycles ($p_{t,\mathrm{Holm}} = 1.6\times 10^{-11}$), and replacing the displacement-anchored well with an origin-anchored well (Row 10) degrades OOD RMSE by $+2.13$ cycles ($p_{t,\mathrm{Holm}} = 2.7\times 10^{-10}$). By contrast, removing Rayleigh damping ($\gamma=0$, Row 8, $+0.45$ cycles) or replacing Strang splitting with Explicit Euler ($M=4$, Row 9, $+0.33$ cycles) produces smaller but statistically significant degradations ($p_{t,\mathrm{Holm}} \le 0.0037$). Note that in Table~\ref{tab:ablation_design} Row 3, \texttt{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} has the exact same raw $p_t = 0.0290$ as in Table~\ref{tab:17competitors_ood}, yielding $p_{t,\mathrm{Holm}} = 0.1196$ when adjusted across the 15 ablation rows (vs.\ $0.0964$ across the 17 competitors in Table~\ref{tab:17competitors_ood}).
    \item \textbf{True ODE Convergence Order and Equal-NFE Comparison (Table~\ref{tab:symplectic_rollout}, Fig.~\ref{fig:ood_slope_symp}(c)):} Against a 2000-step float64 reference, Classical RK4 achieves 4th-order convergence ($p = 4.03$) and lower ODE truncation error than 2nd-order Strang splitting ($p = 2.00$). However, because the neural readout error ($\sim 19.3$ cycles) is five orders of magnitude larger than the ODE discretization error at $M=4$ ($2.32\times 10^{-4}$), \textbf{Strang $M=4$ ($\mathrm{NFE}=4$) matches RK4 $M=4$ ($\mathrm{NFE}=16$) in both single-step ($19.33$ vs.\ $19.34$) and 20x rollout ($20.48$ vs.\ $20.47$) RUL RMSE at $4\times$ fewer force evaluations}, while outperforming equal-budget ($\mathrm{NFE}=4$) Explicit Euler $M=4$ ($19.66$ single-step, $21.89$ rollout).
\end{enumerate}

\subsection{Conformal Conditional Coverage Across Backbones and Outage Shifts}
\label{sec:exp_conformal}

\begin{table*}[!t]
\centering
\caption{Empirical True-RUL Conditional Coverage and Backbone-Agnostic Calibration Ablation Across 8 Architectures on FD001 Under \texttt{OOD 10x--20x} Blackouts (30 Seeds, Nominal $1-\alpha = 95.0\%$). All Quantiles and Bias De-Centering Slopes $\hat{\beta}_b$ Are Fitted on Held-Out Validation Data $\mathcal{I}_{\mathrm{cal}}$ (Zero Test Labels). Reports Marginal Coverage, Mean Interval Width, Winkler Score, 3-Stratum Diagnostic \textbf{True-RUL Coverage} ($\mathrm{Late}\ [0,25]$, $\mathrm{Mid}\ (25,75]$, $\mathrm{Early}\ (75,125]$), and \textbf{5-Bin True-RUL Coverage}.}
\label{tab:conformal_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{Overall Metrics}} & \multicolumn{3}{c}{\textbf{3-Stratum True-RUL Coverage (\%)}} & \multicolumn{5}{c}{\textbf{5-Bin True-RUL Conditional Coverage (\%)}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-13}
\textbf{Backbone Architecture} & \textbf{Calibration Method (Zero Test Labels)} & \textbf{Marg.\ Cov} & \textbf{Width} & \textbf{Winkler} & \textbf{Late $[0,25]$} & \textbf{Mid $(25,75]$} & \textbf{Early $(75,125]$} & \textbf{$[0,15]$} & \textbf{$(15,35]$} & \textbf{$(35,65]$} & \textbf{$(65,95]$} & \textbf{$(95,125]$} \\
\midrule
\multirow{6}{*}{\textbf{CT-PhysProb-JEPA (Clamped)}} & 1. Uncalibrated Gaussian Head & $78.42 \pm 1.85$ & $38.60$ & $142.8$ & $74.15$ & $71.80$ & $86.40$ & $73.20$ & $74.85$ & $70.90$ & $78.50$ & $88.90$ \\
 & 2. Standard Split Conformal & $95.12 \pm 0.78$ & $75.84$ & $96.4$ & $96.85$ & $91.24$ & $97.45$ & $97.10$ & $95.80$ & $90.40$ & $93.80$ & $98.10$ \\
 & 3. Adaptive Conformal Inference (ACI) \cite{gibbs2021adaptive} & $95.18 \pm 0.84$ & $72.46$ & $93.8$ & $96.10$ & $90.85$ & $97.60$ & $96.40$ & $95.20$ & $89.90$ & $93.50$ & $98.20$ \\
 & 4. Conformalized Quantile Regression (CQR) \cite{romano2019conformalized} & $95.08 \pm 0.81$ & $69.92$ & $91.2$ & $95.40$ & $91.60$ & $97.20$ & $95.60$ & $94.90$ & $90.85$ & $94.10$ & $97.80$ \\
 & 5. Symmetric Mondrian Drift-Diffusion (No Bias Corr.) & $95.41 \pm 0.72$ & $64.93$ & $86.5$ & $95.82$ & $88.93$ & $98.45$ & $96.15$ & $94.80$ & $87.90$ & $91.85$ & $98.90$ \\
 & \textbf{6. Bias-Aware Mondrian Drift-Diffusion (Proposed)} & $\mathbf{95.64 \pm 0.68}$ & $\mathbf{66.42}$ & $\mathbf{84.1}$ & $\mathbf{95.94}$ & $\mathbf{94.48}$ & $\mathbf{96.32}$ & $\mathbf{96.10}$ & $\mathbf{95.45}$ & $\mathbf{94.20}$ & $\mathbf{94.95}$ & $\mathbf{96.60}$ \\
\midrule
\textbf{CT-JEPA (Adaptive Full-Gap)} & \textbf{Bias-Aware Mondrian Drift-Diffusion (Proposed)} & $\mathbf{95.58 \pm 0.65}$ & $\mathbf{64.18}$ & $\mathbf{81.9}$ & $\mathbf{95.86}$ & $\mathbf{94.82}$ & $\mathbf{96.15}$ & $\mathbf{95.95}$ & $\mathbf{95.60}$ & $\mathbf{94.65}$ & $\mathbf{95.15}$ & $\mathbf{96.40}$ \\
\midrule
\multirow{2}{*}{PatchTST-RUL \cite{nie2023time}} & Standard Split Conformal & $95.06 \pm 0.85$ & $76.45$ & $97.8$ & $96.60$ & $90.80$ & $97.50$ & $96.85$ & $95.40$ & $89.95$ & $93.40$ & $98.15$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.52 \pm 0.74}$ & $\mathbf{67.18}$ & $\mathbf{85.4}$ & $\mathbf{95.78}$ & $\mathbf{94.25}$ & $\mathbf{96.28}$ & $\mathbf{95.90}$ & $\mathbf{95.30}$ & $\mathbf{93.95}$ & $\mathbf{94.75}$ & $\mathbf{96.55}$ \\
\midrule
\multirow{2}{*}{Latent Neural SDE \cite{li2020scalable}} & Standard Split Conformal & $95.14 \pm 0.82$ & $76.92$ & $98.2$ & $96.72$ & $90.95$ & $97.55$ & $96.95$ & $95.55$ & $90.10$ & $93.60$ & $98.20$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.58 \pm 0.72}$ & $\mathbf{67.45}$ & $\mathbf{85.8}$ & $\mathbf{95.84}$ & $\mathbf{94.36}$ & $\mathbf{96.35}$ & $\mathbf{96.00}$ & $\mathbf{95.40}$ & $\mathbf{94.10}$ & $\mathbf{94.85}$ & $\mathbf{96.60}$ \\
\midrule
\multirow{2}{*}{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} & Standard Split Conformal & $95.04 \pm 0.88$ & $78.12$ & $99.6$ & $96.55$ & $90.45$ & $97.60$ & $96.80$ & $95.25$ & $89.60$ & $93.20$ & $98.25$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.48 \pm 0.79}$ & $\mathbf{68.34}$ & $\mathbf{87.2}$ & $\mathbf{95.72}$ & $\mathbf{94.18}$ & $\mathbf{96.24}$ & $\mathbf{95.85}$ & $\mathbf{95.25}$ & $\mathbf{93.85}$ & $\mathbf{94.70}$ & $\mathbf{96.50}$ \\
\midrule
\multirow{2}{*}{ContiFormer \cite{chen2023contiformer}} & Standard Split Conformal & $95.10 \pm 0.80$ & $77.68$ & $98.9$ & $96.68$ & $90.65$ & $97.52$ & $96.90$ & $95.45$ & $89.80$ & $93.35$ & $98.18$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.54 \pm 0.71}$ & $\mathbf{67.92}$ & $\mathbf{86.6}$ & $\mathbf{95.80}$ & $\mathbf{94.30}$ & $\mathbf{96.30}$ & $\mathbf{95.95}$ & $\mathbf{95.35}$ & $\mathbf{94.00}$ & $\mathbf{94.80}$ & $\mathbf{96.55}$ \\
\midrule
\multirow{2}{*}{mTAN \cite{shukla2021multi}} & Standard Split Conformal & $95.08 \pm 0.81$ & $77.95$ & $99.3$ & $96.62$ & $90.55$ & $97.58$ & $96.85$ & $95.40$ & $89.70$ & $93.30$ & $98.22$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.50 \pm 0.70}$ & $\mathbf{68.15}$ & $\mathbf{86.9}$ & $\mathbf{95.76}$ & $\mathbf{94.22}$ & $\mathbf{96.28}$ & $\mathbf{95.90}$ & $\mathbf{95.30}$ & $\mathbf{93.90}$ & $\mathbf{94.75}$ & $\mathbf{96.52}$ \\
\midrule
\multirow{2}{*}{Confined NODE (Disp-Anchor)} & Standard Split Conformal & $95.02 \pm 0.92$ & $83.40$ & $106.4$ & $96.50$ & $89.80$ & $97.70$ & $96.75$ & $95.10$ & $88.90$ & $92.80$ & $98.35$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.44 \pm 0.84}$ & $\mathbf{72.85}$ & $\mathbf{92.8}$ & $\mathbf{95.68}$ & $\mathbf{94.05}$ & $\mathbf{96.20}$ & $\mathbf{95.80}$ & $\mathbf{95.15}$ & $\mathbf{93.75}$ & $\mathbf{94.60}$ & $\mathbf{96.45}$ \\
\midrule
\multirow{2}{*}{Latent NODE (1st-Ord) \cite{chen2018neural}} & Standard Split Conformal & $94.95 \pm 1.05$ & $94.80$ & $121.5$ & $96.40$ & $88.60$ & $97.85$ & $96.65$ & $94.85$ & $87.60$ & $91.90$ & $98.50$ \\
 & \textbf{Bias-Aware Mondrian Drift-Diffusion} & $\mathbf{95.38 \pm 0.94}$ & $\mathbf{82.60}$ & $\mathbf{105.4}$ & $\mathbf{95.60}$ & $\mathbf{93.85}$ & $\mathbf{96.15}$ & $\mathbf{95.70}$ & $\mathbf{95.00}$ & $\mathbf{93.50}$ & $\mathbf{94.45}$ & $\mathbf{96.40}$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table*}[!t]
\centering
\caption{Unseen Operational Outage Shift Evaluation Across 6 Backbones and 3 Shift Regimes on FD001 (30 Seeds, Calibrated Exclusively on In-Distribution $25\%$ Dropout Validation Data $\Delta t \in [1\tau_0, 2\tau_0]$ and Tested Without Recalibration on Unseen Outage Distributions).}
\label{tab:unseen_shift}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllccccc}
\toprule
\textbf{Unseen Outage Shift Regime} & \textbf{Backbone Architecture} & \textbf{Conformal Calibration Method (Trained on ID 25\% Only)} & \textbf{Marginal Coverage (\%)} & \textbf{True Mid $(25,75]$ Cov (\%)} & \textbf{True Late $[0,25]$ Cov (\%)} & \textbf{Mean Width (Cycles)} & \textbf{ESS Ratio ($N_{\mathrm{eff}}/N_{\mathrm{cal}}$)} \\
\midrule
\multirow{15}{*}{\textbf{ID 25\% $\to$ OOD 10x--20x}} & \multirow{5}{*}{\textbf{CT-PhysProb-JEPA}} & 1. Standard Split Conformal (Unweighted ID Cal) & $84.64 \pm 2.45$ & $76.40$ & $82.10$ & $52.80 \pm 1.85$ & $1.00$ \\
 &  & 2. Adaptive Conformal Inference (ACI) \cite{gibbs2021adaptive} & $90.84 \pm 1.92$ & $84.50$ & $89.60$ & $63.40 \pm 2.40$ & $1.00$ \\
 &  & 3. Symmetric Mondrian Drift-Diffusion (No Bias/IW) & $94.61 \pm 1.59$ & $88.15$ & $94.80$ & $65.80 \pm 2.12$ & $1.00$ \\
 &  & 4. Bias-Aware Mondrian Drift-Diffusion (Unweighted) & $94.88 \pm 1.42$ & $93.65$ & $95.10$ & $67.20 \pm 2.05$ & $1.00$ \\
 &  & \textbf{5. Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.24 \pm 1.18}$ & $\mathbf{94.42}$ & $\mathbf{95.48}$ & $\mathbf{68.45 \pm 2.15}$ & $0.64$ \\
\cmidrule(lr){2-8}
 & \multirow{2}{*}{PatchTST-RUL \cite{nie2023time}} & Standard Split Conformal (Unweighted ID Cal) & $83.65 \pm 2.58$ & $75.80$ & $81.50$ & $53.20 \pm 1.95$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.12 \pm 1.26}$ & $\mathbf{94.18}$ & $\mathbf{95.35}$ & $\mathbf{69.30 \pm 2.28}$ & $0.64$ \\
\cmidrule(lr){2-8}
 & \multirow{2}{*}{Latent Neural SDE \cite{li2020scalable}} & Standard Split Conformal (Unweighted ID Cal) & $83.90 \pm 2.52$ & $76.10$ & $81.80$ & $53.50 \pm 1.92$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.18 \pm 1.22}$ & $\mathbf{94.28}$ & $\mathbf{95.40}$ & $\mathbf{69.65 \pm 2.24}$ & $0.64$ \\
\cmidrule(lr){2-8}
 & \multirow{2}{*}{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} & Standard Split Conformal (Unweighted ID Cal) & $83.20 \pm 2.68$ & $75.20$ & $81.10$ & $54.10 \pm 2.08$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.04 \pm 1.34}$ & $\mathbf{94.05}$ & $\mathbf{95.25}$ & $\mathbf{70.60 \pm 2.38}$ & $0.64$ \\
\cmidrule(lr){2-8}
 & \multirow{2}{*}{ContiFormer \cite{chen2023contiformer}} & Standard Split Conformal (Unweighted ID Cal) & $83.50 \pm 2.55$ & $75.60$ & $81.40$ & $53.80 \pm 1.98$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.10 \pm 1.25}$ & $\mathbf{94.15}$ & $\mathbf{95.32}$ & $\mathbf{70.15 \pm 2.26}$ & $0.64$ \\
\cmidrule(lr){2-8}
 & \multirow{2}{*}{Confined NODE (Disp-Anchor)} & Standard Split Conformal (Unweighted ID Cal) & $81.80 \pm 2.85$ & $73.40$ & $79.80$ & $56.40 \pm 2.25$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{94.92 \pm 1.45}$ & $\mathbf{93.88}$ & $\mathbf{95.15}$ & $\mathbf{75.20 \pm 2.62}$ & $0.64$ \\
\midrule
\multirow{3}{*}{\textbf{ID 25\% $\to$ Markov ACARS}} & \multirow{3}{*}{\textbf{CT-PhysProb-JEPA}} & Standard Split Conformal (Unweighted ID Cal) & $92.85 \pm 1.15$ & $89.40$ & $91.80$ & $52.80 \pm 1.85$ & $1.00$ \\
 &  & Bias-Aware Mondrian Drift-Diffusion (Unweighted) & $95.18 \pm 0.82$ & $94.55$ & $95.40$ & $54.60 \pm 1.68$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.36 \pm 0.76}$ & $\mathbf{94.82}$ & $\mathbf{95.58}$ & $\mathbf{55.10 \pm 1.72}$ & $0.86$ \\
\midrule
\multirow{3}{*}{\textbf{ID 25\% $\to$ 70\% MCAR}} & \multirow{3}{*}{\textbf{CT-PhysProb-JEPA}} & Standard Split Conformal (Unweighted ID Cal) & $88.45 \pm 1.78$ & $82.90$ & $87.20$ & $52.80 \pm 1.85$ & $1.00$ \\
 &  & Bias-Aware Mondrian Drift-Diffusion (Unweighted) & $94.95 \pm 1.12$ & $93.92$ & $95.18$ & $61.40 \pm 1.92$ & $1.00$ \\
 &  & \textbf{Importance-Weighted Bias-Aware Mondrian} \cite{tibshirani2019conformal} & $\mathbf{95.28 \pm 0.98}$ & $\mathbf{94.50}$ & $\mathbf{95.45}$ & $\mathbf{62.35 \pm 1.98}$ & $0.74$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

Table~\ref{tab:conformal_ablation} and Table~\ref{tab:unseen_shift} show two main results:
\begin{enumerate}
    \item \textbf{Restoring Empirical Mid-Life Coverage Across All 8 Backbones (Table~\ref{tab:conformal_ablation}):} Under \texttt{OOD 10x--20x}, symmetric Mondrian conformal achieves $95.41\%$ marginal coverage but drops to $88.93\%$ on True Mid-Life $(25, 75]$ ($87.90\%$ in $(35, 65]$). \textbf{Bias-Aware Mondrian Drift-Diffusion} raises empirical True Mid-Life coverage to $94.48\%$ ($94.82\%$ with Adaptive Full-Gap) and reduces the Winkler score from $86.5$ to $84.1$ ($81.9$ with Adaptive Full-Gap). Applied identically to \texttt{PatchTST-RUL}, \texttt{Latent SDE}, \texttt{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$}, \texttt{ContiFormer}, \texttt{mTAN}, \texttt{Confined NODE}, and \texttt{Latent NODE}, it raises empirical True Mid-Life coverage from $88.6\%\text{--}90.9\%$ to $93.85\%\text{--}94.36\%$ while narrowing mean width by $9.5\text{--}12.2$ cycles vs.\ Standard Split Conformal.
    \item \textbf{Coverage Under Simulated Outage Shift (Table~\ref{tab:unseen_shift}):} When calibrated exclusively on in-distribution $25\%$ dropout ($\Delta t \in [1\tau_0, 2\tau_0]$) and tested without recalibration on \texttt{OOD 10x--20x}, unweighted Split Conformal drops to $81.80\%\text{--}84.64\%$ marginal coverage ($73.40\%\text{--}76.40\%$ on True Mid-Life). Most of the recovery comes from Drift-Diffusion scaling and Bias-Aware Mondrian de-centering ($94.88 \pm 1.42\%$ marginal, $93.65\%$ Mid-Life; Row 4), while likelihood-ratio importance weighting (Row 5, $\mathrm{ESS}/N_{\mathrm{cal}} = 0.64$) provides an additional $+0.36\%$ marginal and $+0.77\%$ Mid-Life gain to reach $95.24 \pm 1.18\%$ marginal and $94.42\%$ Mid-Life coverage.
\end{enumerate}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/Fig2_Calibrated_Uncertainty_Trajectory.pdf}
\caption{Reliability and maintenance decision analysis: (a) Per-engine run-to-failure trajectory on FD001 Unit \#84 across a healthy plateau blackout (cycles 49--86) and an active-degradation blackout (cycles 201--241); (b) Diagnostic True-RUL conditional coverage ablation on \texttt{OOD 10x--20x} showing True Mid-Life $(25,75]$ coverage lifted from $88.93\%$ to $94.48\%\text{--}94.82\%$ via Bias-Aware Mondrian Conformal; and (c) Validation-tuned 500-trajectory Condition-Based Maintenance (CBM) expected cost per unit ($\pm 1\text{ SD}$ across 30 seeds) across $C_{\mathrm{fail}}/C_{\mathrm{FA}} \in \{5:1, 10:1, 20:1, 50:1\}$.}
\label{fig:traj_conf_cbm}
\end{figure*}

\section{Condition-Based Maintenance (CBM) Decision Study}
\label{sec:cbm}

We evaluate maintenance decision policies across $N_{\mathrm{eval}} = 500$ critical test truncation trajectories per seed and $S=30$ seeds ($15{,}000$ unit evaluations per policy) on FD001 under \texttt{OOD 10x--20x} blackouts. Each policy triggers preventive maintenance when its lower confidence bound (or point prediction) crosses a threshold $\tau_{\mathrm{cal}}^* \in \{5.0, 6.0, \dots, 54.0\}$ cycles selected via an equal 50-point grid search exclusively on the held-out validation set $\mathcal{I}_{\mathrm{cal}}$ to minimize total expected cost:
\begin{equation}
C_{\mathrm{unit}} = C_{\mathrm{fail}}\,\mathbb{I}(\text{unplanned failure}) + C_{\mathrm{FA}}\,\mathbb{I}(\text{premature alarm}) + C_{\mathrm{waste}}\,\max(0, y_{\mathrm{trig}} - \tau_{\mathrm{safe}}),
\label{eq:cbm_cost}
\end{equation}
across cost ratios $C_{\mathrm{fail}}/C_{\mathrm{FA}} \in \{5:1, 10:1, 20:1, 50:1\}$ with $C_{\mathrm{FA}} = \$10{,}000$ and $C_{\mathrm{waste}} \in \{\$150, \$450, \$750\}/\text{cycle}$.

\begin{table*}[!t]
\centering
\caption{Validation-Tuned 500-Trajectory Condition-Based Maintenance (CBM) Evaluation Across 10 Architectures on FD001 Under \texttt{OOD 10x--20x} Blackouts ($N_{\mathrm{eval}}=500$ Critical Truncation Trajectories Per Seed Across $S=30$ Seeds). Every Policy's Intervention Threshold $\tau_{\mathrm{cal}}^*$ Is Selected via an Equal 50-Point Grid Search on the Held-Out Validation Set $\mathcal{I}_{\mathrm{cal}}$ Only.}
\label{tab:cbm_decision}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccc}
\toprule
 & & & \multicolumn{3}{c}{\textbf{Operating Rates (30-Seed Mean $\pm$ SD)}} & \multicolumn{4}{c}{\textbf{Mean $\pm$ SD Cost / Unit (\$ USD, $C_{\mathrm{waste}}=\$450$)}} & \multicolumn{2}{c}{\textbf{$C_{\mathrm{waste}}$ Sweep ($10:1$)}} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-10} \cmidrule(lr){11-12}
\textbf{Model Architecture} & \textbf{Validation-Tuned Policy (50-Pt Grid on $\mathcal{I}_{\mathrm{cal}}$)} & \textbf{$\tau_{\mathrm{cal}}^*$} & \textbf{Unplanned Fail (\%)} & \textbf{False Alarm (\%)} & \textbf{Waste (Cyc)} & \textbf{$5:1$ ($\$50\text{k}$)} & \textbf{$10:1$ ($\$100\text{k}$)} & \textbf{$20:1$ ($\$200\text{k}$)} & \textbf{$50:1$ ($\$500\text{k}$)} & \textbf{$\$150$/cyc} & \textbf{$\$750$/cyc} \\
\midrule
\multirow{4}{*}{\textbf{CT-JEPA (Adaptive Full-Gap)}} & \textbf{1. Bias-Aware Mondrian LCB} & $19.5$ & $\mathbf{0.34 \pm 0.22}$ & $\mathbf{5.28 \pm 0.65}$ & $6.94 \pm 0.43$ & $\mathbf{\$3{,}821 \pm 163}$ & $\mathbf{\$3{,}991 \pm 202}$ & $\mathbf{\$4{,}331 \pm 315}$ & $\mathbf{\$5{,}351 \pm 706}$ & $\mathbf{\$1{,}909}$ & $\mathbf{\$6{,}073}$ \\
 & 2. Standard Split-Conformal LCB & $17.0$ & $0.40 \pm 0.24$ & $6.97 \pm 0.75$ & $8.05 \pm 0.46$ & $\$4{,}520 \pm 179$ & $\$4{,}721 \pm 220$ & $\$5{,}122 \pm 341$ & $\$6{,}326 \pm 762$ & $\$2{,}306$ & $\$7{,}136$ \\
 & 3. Gaussian $\hat{\mu} - 1.645\sigma_0$ LCB & $24.0$ & $0.97 \pm 0.37$ & $7.81 \pm 0.79$ & $6.52 \pm 0.42$ & $\$4{,}202 \pm 203$ & $\$4{,}686 \pm 286$ & $\$5{,}655 \pm 504$ & $\$8{,}562 \pm 1207$ & $\$2{,}729$ & $\$6{,}643$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $28.5$ & $1.56 \pm 0.47$ & $8.71 \pm 0.84$ & $\mathbf{5.97 \pm 0.40}$ & $\$4{,}339 \pm 229$ & $\$5{,}121 \pm 345$ & $\$6{,}685 \pm 636$ & $\$11{,}377 \pm 1552$ & $\$3{,}330$ & $\$6{,}912$ \\
\midrule
\multirow{4}{*}{\textbf{CT-PhysProb-JEPA (Clamped)}} & \textbf{1. Bias-Aware Mondrian LCB} & $19.0$ & $0.38 \pm 0.24$ & $5.42 \pm 0.68$ & $7.12 \pm 0.45$ & $\$3{,}936 \pm 174$ & $\$4{,}126 \pm 217$ & $\$4{,}506 \pm 341$ & $\$5{,}646 \pm 768$ & $\$1{,}990$ & $\$6{,}262$ \\
 & 2. Standard Split-Conformal LCB & $16.5$ & $0.45 \pm 0.26$ & $7.15 \pm 0.78$ & $8.26 \pm 0.49$ & $\$4{,}656 \pm 191$ & $\$4{,}880 \pm 235$ & $\$5{,}329 \pm 367$ & $\$6{,}674 \pm 827$ & $\$2{,}402$ & $\$7{,}358$ \\
 & 3. Gaussian $\hat{\mu} - 1.645\sigma_0$ LCB & $23.5$ & $1.08 \pm 0.40$ & $8.02 \pm 0.83$ & $6.69 \pm 0.44$ & $\$4{,}355 \pm 218$ & $\$4{,}897 \pm 309$ & $\$5{,}980 \pm 547$ & $\$9{,}229 \pm 1314$ & $\$2{,}889$ & $\$6{,}905$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $28.0$ & $1.75 \pm 0.51$ & $8.94 \pm 0.87$ & $6.12 \pm 0.42$ & $\$4{,}524 \pm 246$ & $\$5{,}398 \pm 374$ & $\$7{,}146 \pm 693$ & $\$12{,}390 \pm 1696$ & $\$3{,}561$ & $\$7{,}235$ \\
\midrule
\multirow{2}{*}{PatchTST-RUL \cite{nie2023time}} & \textbf{1. Bias-Aware Mondrian LCB} & $19.0$ & $0.44 \pm 0.28$ & $5.68 \pm 0.74$ & $7.35 \pm 0.49$ & $\$4{,}096 \pm 195$ & $\$4{,}316 \pm 245$ & $\$4{,}756 \pm 392$ & $\$6{,}076 \pm 894$ & $\$2{,}111$ & $\$6{,}521$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $28.0$ & $2.02 \pm 0.60$ & $9.37 \pm 0.95$ & $6.32 \pm 0.45$ & $\$4{,}794 \pm 281$ & $\$5{,}806 \pm 432$ & $\$7{,}830 \pm 806$ & $\$13{,}902 \pm 1980$ & $\$3{,}910$ & $\$7{,}702$ \\
\midrule
\multirow{2}{*}{Latent Neural SDE \cite{li2020scalable}} & \textbf{1. Bias-Aware Mondrian LCB} & $19.0$ & $0.46 \pm 0.29$ & $5.75 \pm 0.76$ & $7.42 \pm 0.51$ & $\$4{,}144 \pm 201$ & $\$4{,}374 \pm 253$ & $\$4{,}834 \pm 405$ & $\$6{,}214 \pm 926$ & $\$2{,}148$ & $\$6{,}600$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $28.0$ & $2.12 \pm 0.62$ & $9.49 \pm 0.98$ & $6.38 \pm 0.47$ & $\$4{,}878 \pm 291$ & $\$5{,}936 \pm 447$ & $\$8{,}052 \pm 834$ & $\$14{,}400 \pm 2048$ & $\$4{,}022$ & $\$7{,}850$ \\
\midrule
\multirow{2}{*}{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} & \textbf{1. Bias-Aware Mondrian LCB} & $18.5$ & $0.52 \pm 0.31$ & $6.04 \pm 0.82$ & $7.58 \pm 0.54$ & $\$4{,}275 \pm 215$ & $\$4{,}535 \pm 271$ & $\$5{,}055 \pm 434$ & $\$6{,}615 \pm 990$ & $\$2{,}261$ & $\$6{,}809$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $27.5$ & $2.39 \pm 0.66$ & $9.97 \pm 1.05$ & $6.52 \pm 0.50$ & $\$5{,}126 \pm 311$ & $\$6{,}322 \pm 477$ & $\$8{,}714 \pm 891$ & $\$15{,}890 \pm 2186$ & $\$4{,}366$ & $\$8{,}278$ \\
\midrule
\multirow{2}{*}{ContiFormer \cite{chen2023contiformer}} & \textbf{1. Bias-Aware Mondrian LCB} & $18.5$ & $0.48 \pm 0.29$ & $5.86 \pm 0.78$ & $7.48 \pm 0.52$ & $\$4{,}192 \pm 203$ & $\$4{,}432 \pm 255$ & $\$4{,}912 \pm 406$ & $\$6{,}352 \pm 926$ & $\$2{,}188$ & $\$6{,}676$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $27.5$ & $2.21 \pm 0.62$ & $9.67 \pm 1.00$ & $6.43 \pm 0.48$ & $\$4{,}966 \pm 293$ & $\$6{,}070 \pm 448$ & $\$8{,}278 \pm 834$ & $\$14{,}902 \pm 2046$ & $\$4{,}140$ & $\$8{,}000$ \\
\midrule
\multirow{2}{*}{mTAN \cite{shukla2021multi}} & \textbf{1. Bias-Aware Mondrian LCB} & $18.5$ & $0.50 \pm 0.30$ & $5.95 \pm 0.80$ & $7.54 \pm 0.53$ & $\$4{,}238 \pm 209$ & $\$4{,}488 \pm 263$ & $\$4{,}988 \pm 419$ & $\$6{,}488 \pm 958$ & $\$2{,}226$ & $\$6{,}750$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $27.5$ & $2.30 \pm 0.64$ & $9.82 \pm 1.03$ & $6.48 \pm 0.49$ & $\$5{,}049 \pm 302$ & $\$6{,}199 \pm 462$ & $\$8{,}499 \pm 862$ & $\$15{,}399 \pm 2116$ & $\$4{,}254$ & $\$8{,}144$ \\
\midrule
\multirow{2}{*}{Port-Hamiltonian (pHNN) \cite{desai2021port}} & \textbf{1. Bias-Aware Mondrian LCB} & $18.0$ & $0.58 \pm 0.34$ & $6.32 \pm 0.86$ & $7.82 \pm 0.58$ & $\$4{,}441 \pm 230$ & $\$4{,}731 \pm 292$ & $\$5{,}311 \pm 471$ & $\$7{,}051 \pm 1084$ & $\$2{,}385$ & $\$7{,}077$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $27.0$ & $2.67 \pm 0.73$ & $10.43 \pm 1.10$ & $6.73 \pm 0.54$ & $\$5{,}404 \pm 337$ & $\$6{,}738 \pm 522$ & $\$9{,}406 \pm 980$ & $\$17{,}410 \pm 2414$ & $\$4{,}720$ & $\$8{,}756$ \\
\midrule
\multirow{2}{*}{Confined NODE (Disp-Anchor)} & \textbf{1. Bias-Aware Mondrian LCB} & $17.5$ & $0.68 \pm 0.38$ & $6.85 \pm 0.94$ & $8.24 \pm 0.64$ & $\$4{,}733 \pm 254$ & $\$5{,}073 \pm 324$ & $\$5{,}753 \pm 525$ & $\$7{,}793 \pm 1211$ & $\$2{,}601$ & $\$7{,}545$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $26.5$ & $3.13 \pm 0.81$ & $11.30 \pm 1.21$ & $7.09 \pm 0.59$ & $\$5{,}883 \pm 373$ & $\$7{,}447 \pm 579$ & $\$10{,}575 \pm 1090$ & $\$19{,}959 \pm 2686$ & $\$5{,}321$ & $\$9{,}573$ \\
\midrule
\multirow{2}{*}{Latent NODE (1st-Ord) \cite{chen2018neural}} & \textbf{1. Bias-Aware Mondrian LCB} & $16.5$ & $0.92 \pm 0.46$ & $8.12 \pm 1.12$ & $9.45 \pm 0.78$ & $\$5{,}524 \pm 306$ & $\$5{,}984 \pm 391$ & $\$6{,}904 \pm 635$ & $\$9{,}664 \pm 1465$ & $\$3{,}149$ & $\$8{,}819$ \\
 & 4. Point Forecast Threshold ($\hat{\mu} \le \tau_{\mathrm{cal}}^*$) & $25.5$ & $4.23 \pm 0.99$ & $13.40 \pm 1.44$ & $8.13 \pm 0.72$ & $\$7{,}114 \pm 455$ & $\$9{,}230 \pm 706$ & $\$13{,}462 \pm 1330$ & $\$26{,}158 \pm 3280$ & $\$6{,}792$ & $\$11{,}668$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

Table~\ref{tab:cbm_decision} and Fig.~\ref{fig:traj_conf_cbm}(c) show that across every evaluated backbone, replacing \texttt{Point Forecast Threshold} (Policy 4) with \texttt{Bias-Aware Mondrian LCB} (Policy 1) reduces unplanned failure rates by $4.6\times$ ($1.56\% \to 0.34\%$ for \texttt{CT-JEPA Adaptive}; $2.02\% \to 0.44\%$ for \texttt{PatchTST-RUL}) and lowers expected per-unit cost at $10:1$ by $22.1\%\text{--}35.2\%$ ($\$3{,}991 \pm 202$ for \texttt{CT-JEPA Adaptive} and $\$4{,}126 \pm 217$ for \texttt{CT-PhysProb-JEPA Clamped}, vs.\ $\$4{,}316 \pm 245$ for \texttt{PatchTST-RUL} and $\$4{,}374 \pm 253$ for \texttt{Latent SDE} under the same Bias-Aware Mondrian policy).

\noindent\textbf{[EXPERIMENT NEEDED: Full 4-Policy Symmetric CBM Grid Across All Competing Backbones Plus Non-Learned Periodic Maintenance Baseline.}
\emph{Description:} Complete Policies 2 (\texttt{Standard Split-Conformal LCB}) and 3 (\texttt{Gaussian $\hat{\mu} - 1.645\sigma_0$ LCB}) for the remaining 8 competing backbones in Table~\ref{tab:cbm_decision} (expanding from 24 rows to all 40 model--policy rows), and add a non-learned \textbf{Fixed-Interval Preventive Maintenance (Age-Based Replacement at $T_{\mathrm{rep}}^*$ cycles)} baseline tuned on $\mathcal{I}_{\mathrm{cal}}$ alongside a \textbf{Run-to-Failure / LOCF Threshold} baseline.
\emph{Protocol:} For all 10 architectures $\times$ 4 policies plus Age-Based Periodic Replacement ($T_{\mathrm{rep}}^* \in \{100, 105, \dots, 250\}$ cycles tuned on $\mathcal{I}_{\mathrm{cal}}$), evaluate Eq.~\eqref{eq:cbm_cost} across the 500 critical test truncation trajectories $\times$ 30 seeds and report $\tau_{\mathrm{cal}}^*$, Unplanned Failure Rate, False Alarm Rate, Waste Cycles, and Expected Cost across $C_{\mathrm{fail}}/C_{\mathrm{FA}} \in \{2:1, 5:1, 10:1, 20:1, 50:1\}$.
\emph{Expected Table Format:} Complete 42-row table (10 models $\times$ 4 policies + Periodic Age Replacement + LOCF Threshold) with identical columns to Table~\ref{tab:cbm_decision} plus $2:1$ low-penalty cost ratio.\textbf{]}

\section{Limitations and Field Deployment Scope}
\label{sec:limitations}

To ensure transparent interpretation for safety-critical reliability engineering, we explicitly state five limitations of this study and the field validation required prior to operational certification:
\begin{enumerate}
    \item \textbf{Synthetic Outage Models vs.\ Field Datalink Logs:} All missingness patterns evaluated in Tables~\ref{tab:17competitors_ood}--\ref{tab:cbm_decision} (\texttt{Markov ACARS}, \texttt{Cruise-Throttle}, $70\%$ MCAR, and $[10\tau_0, 25\tau_0]$ OOD blackouts) are synthetic simulation masks applied to C-MAPSS, N-CMAPSS, and PRONOSTIA trajectories. While motivated by aerospace datalink dropout mechanics \cite{arias2021aircraft}, they do not capture hardware-coupled missingness where sensor failure is directly triggered by physical asset degradation (Missing Not At Random, MNAR). Field certification requires executing the empirical datalink log protocol specified in Section~\ref{sec:outage_models}.
    \item \textbf{Statistical Ties in Mean Point-Forecast RMSE vs.\ Variance and Rollout Efficiency:} On the pre-registered primary endpoint (\texttt{FD001 OOD 10x--20x}) under matched clamping ($h_{\max}=0.35$), \texttt{CT-PhysProb-JEPA} ($19.33 \pm 0.49$) does not outperform \texttt{PatchTST-RUL} ($19.26 \pm 0.85$) or \texttt{Latent Neural SDE} ($19.35 \pm 0.86$) in mean RMSE, and its $+0.37$-cycle edge over our causal control \texttt{Plain 2nd-Order NODE + $\mathcal{L}_{\mathrm{kin}}$} ($19.70 \pm 0.90$) is not statistically significant after Holm correction ($p_{t,\mathrm{Holm}} = 0.0964$). The specific operational justification for the Euler--Lagrange + Strang backbone is its $45.6\%$ lower seed-to-seed standard deviation ($0.49$ vs.\ $0.85\text{--}0.90$ cycles), its $4\times$ lower NFE cost vs.\ RK4 ($4$ vs.\ $16$), and its accuracy advantage on extreme $15\times\text{--}25\times$ blackouts ($20.40$ vs.\ $21.34$ cycles).
    \item \textbf{Trade-Off Between Clamped Overall RMSE and Adaptive Active-Degradation Bias:} Replacing $h_{\max}=0.35$ with unclamped Full-Gap Horizon-Adaptive Substepping ($M_k = \lceil \tilde{\Delta t}_k / 0.0875 \rceil$) cuts active-degradation ($\mathrm{Slope}=-1$) optimistic bias from $+12.75$ to $+6.84$ cycles and reduces active-degradation RMSE from $23.09$ to $20.14$ cycles (Table~\ref{tab:slope_breakdown}), and achieves the lowest CBM cost ($\$3{,}991$/unit; Table~\ref{tab:cbm_decision}), but increases all-window OOD 10x--20x RMSE by $+0.35$ cycles ($19.33 \to 19.68$; Table~\ref{tab:ablation_design} Row 4) because propagating $\dot{q}_k$ over $20\tau_0$ slightly increases variance on flat early-life plateaus.
    \item \textbf{Empirical vs.\ Formal Conditional Conformal Coverage and Support Overlap:} Partitioning Mondrian bins by predicted RUL $\hat{\mu} \in \mathcal{B}_b$ guarantees coverage conditioned on $\hat{\mu} \in \mathcal{B}_b$ only when the bias slope $\hat{\beta}_b$ is fitted on a split $\mathcal{I}_{\mathrm{bias}}$ disjoint from the quantile calibration split $\mathcal{I}_{\mathrm{cal}}$; coverage conditioned on unobservable true RUL ($y \in (25, 75]$) is an empirical diagnostic. Similarly, 1D KDE importance weighting on $\tilde{\Delta t}_k$ between $[1\tau_0, 2\tau_0]$ and $[10\tau_0, 20\tau_0]$ relies on clipped kernel tail extrapolation.
    \item \textbf{Three-Stage Engineering Fallback Protocol:} In deployment, we pair $\Delta t_k$ and the Bias-Aware Mondrian interval width $W_{95}(t_k)$ with a three-stage operational protocol: (i) for $\Delta t_k \le 15\tau_0$ and $W_{95}(t_k) \le 62.5$ cycles, propagate $(q_k, \dot{q}_k)$ via adaptive substepping and trigger maintenance when $L_{95}^{\mathrm{Mond}}(t_k) \le \tau_{\mathrm{cal}}^*$; (ii) for $\Delta t_k > 15\tau_0$ or $W_{95}(t_k) > 62.5$ cycles, issue an automated ground-link polling or QAR download request; and (iii) if telemetry is not restored within $20\tau_0$ on an asset with pre-blackout RUL $< 60$ cycles, mandate a physical borescope or vibration inspection.
\end{enumerate}

\section{Conclusion}
\label{sec:conclusion}

We have presented an end-to-end reliability investigation of continuous-time RUL prognostics and maintenance decision optimization under multi-cycle telemetry blackouts. First, we proved analytically and demonstrated empirically that bounded-horizon integration clamping ($h_{\max}=0.35$) causes point predictions $\widehat{\mathrm{RUL}}(\Delta t)$ to saturate for all blackouts $\Delta t \ge 4\tau_0$, inducing a $+12.75$-cycle optimistic bias during active degradation that is cut to $+6.84$ cycles via unclamped Full-Gap Horizon-Adaptive Substepping ($M_k = \lceil \tilde{\Delta t}_k / 0.0875 \rceil$). Second, by introducing a matched causal control (\texttt{Plain 2nd-Order NODE + Disp-Anchor + $\mathcal{L}_{\mathrm{kin}}$}) across 30 seeds and 20 models, we showed that $80.3\%$ of the single-step OOD RMSE reduction over first-order confined NODEs stems from second-order position--velocity coordinates with secant velocity supervision and displacement anchoring, whereas damped Euler--Lagrange mechanics and Strang splitting reduce seed standard deviation by $45.6\%$, improve accuracy on extreme $15\times\text{--}25\times$ blackouts, and match 4th-order RK4 rollout accuracy at $4\times$ fewer force evaluations. Third, we developed a backbone-agnostic Bias-Aware Mondrian Drift-Diffusion conformal calibrator that raises empirical True Mid-Life $(25, 75]$ coverage from $88.93\%$ to $94.48\%\text{--}94.82\%$ across eight backbones and lowers validation-tuned CBM expected cost across $500$ critical truncation trajectories and $30$ seeds for all failure-to-preventive cost ratios $\ge 5:1$.

\appendices

\section{Proofs of Structural Stability Properties and Note on Finite-Horizon ODE Bounds}
\label{app:stability_proofs}

\begin{proposition}[Conformal Symplecticity and Volume Contraction]
\label{prop:symp}
For the Strang-split step $\Psi_{\delta t} = \mathcal{D}_{\delta t/2} \circ \mathcal{S}_{\delta t} \circ \mathcal{D}_{\delta t/2}$ on $\mathbb{R}^{2d}$ equipped with $\Omega = \begin{bmatrix} 0 & I_d \\ -I_d & 0 \end{bmatrix}$, the inner St\"{o}rmer--Verlet map $\mathcal{S}_{\delta t}$ has a symmetric force Jacobian $H(q) = \frac{\partial F_{\mathrm{tot}}}{\partial q} = H(q)^T$ and factors into three unit-determinant symplectic shears:
\begin{equation}
D\mathcal{S}_{\delta t} = \begin{bmatrix} I_d & 0 \\ \frac{\delta t}{2} H(q^{(m+1)}) & I_d \end{bmatrix} \begin{bmatrix} I_d & \delta t I_d \\ 0 & I_d \end{bmatrix} \begin{bmatrix} I_d & 0 \\ \frac{\delta t}{2} H(q^{(m)}) & I_d \end{bmatrix},
\end{equation}
satisfying $(D\mathcal{S}_{\delta t})^T \Omega (D\mathcal{S}_{\delta t}) = \Omega$ and $\det(D\mathcal{S}_{\delta t}) = 1$. Since $D\mathcal{D}_{\delta t/2} = \mathrm{diag}(I_d, e^{-\gamma\delta t/2}I_d)$ has determinant $e^{-d\gamma\delta t/2}$, determinant multiplicativity yields $\det(D\Psi_{\delta t}) = e^{-d\gamma\delta t}$ and $M$-step Jacobian determinant $\det(D\Phi_{H_k}) = e^{-d\gamma H_k}$ (verified via float64 autograd to $\le 5.75\times 10^{-9}$). \hfill $\blacksquare$
\end{proposition}

\begin{proposition}[Ultimate Boundedness Around the Pre-Gap Anchor]
\label{prop:uub}
In displacement coordinates $\tilde{q} = q - q_k$ and $p = \dot{q}$, consider the cross-coupled Lyapunov function $W(\tilde{q}, p) = \frac{1}{2}\|p\|_2^2 + \frac{1}{2}\alpha_{\mathrm{eff}}\|\tilde{q}\|_2^2 + \epsilon \tilde{q}^T p$ with $\epsilon \in (0, \min(\gamma, \sqrt{\alpha_{\mathrm{eff}}}))$. Its $2\times 2$ block symbol $P_0 = \begin{bmatrix} \alpha_{\mathrm{eff}} & \epsilon \\ \epsilon & 1 \end{bmatrix}$ has positive eigenvalues $\lambda_{\min,\max}(P) = \frac{(\alpha_{\mathrm{eff}}+1) \mp \sqrt{(\alpha_{\mathrm{eff}}-1)^2 + 4\epsilon^2}}{2} > 0$. For $\alpha_{\mathrm{eff}} = 0.35, \gamma = 0.15, \epsilon = 0.08$, $\lambda_{\min}(P) = 0.34031$ ($\sqrt{\lambda_{\min}(P)} = 0.58336$) and $\lambda_{\max}(P) = 1.00969$. Differentiating $W(\tilde{q}, p)$ along Eq.~\eqref{eq:eom} where $\|\tilde{F}(q)\|_2 = \|-\nabla_q V_{\mathrm{mlp}}(q) + F_{\mathrm{ext}}(q)\|_2 \le L_V + F_{\max}$ and applying Young's inequality yields $\dot{W} \le -c_q\|\tilde{q}\|_2^2 - c_p\|p\|_2^2 + C_F \le -\mu W + C_F$ with $c_q = \frac{1}{2}\epsilon\alpha_{\mathrm{eff}} = 0.0140$, $c_p = \frac{1}{2}(\gamma - \epsilon) - \frac{\epsilon\gamma^2}{4\alpha_{\mathrm{eff}}} = 0.02471$, and $\mu = \frac{2\min(c_q, c_p)}{\lambda_{\max}(P)} = 0.02773 > 0$, confining trajectories to the compact ball $\|\tilde{z}(t)\|_2 \le \sqrt{\frac{2}{\lambda_{\min}(P)}\max(W(0), C_F/\mu)}$. \hfill $\blacksquare$
\end{proposition}

\textbf{Proof of Proposition~\ref{prop:jury}:} Writing $D_{\delta t/2} = \mathrm{diag}(1, e^{-\gamma\delta t/2})$ and the linear St\"{o}rmer--Verlet matrix $S_{\delta t}^{\mathrm{lin}} = \begin{bmatrix} 1 - \frac{1}{2}\alpha_{\mathrm{eff}}\delta t^2 & \delta t \\ -\alpha_{\mathrm{eff}}\delta t(1 - \frac{1}{4}\alpha_{\mathrm{eff}}\delta t^2) & 1 - \frac{1}{2}\alpha_{\mathrm{eff}}\delta t^2 \end{bmatrix}$, direct multiplication $A(\delta t) = D_{\delta t/2}S_{\delta t}^{\mathrm{lin}}D_{\delta t/2}$ gives diagonal entries $A_{11} = 1 - \frac{1}{2}\alpha_{\mathrm{eff}}\delta t^2$ and $A_{22} = e^{-\gamma\delta t}(1 - \frac{1}{2}\alpha_{\mathrm{eff}}\delta t^2)$, whose sum is Eq.~\eqref{eq:exact_trace}, and $\det(A(\delta t)) = e^{-\gamma\delta t}\det(S_{\delta t}^{\mathrm{lin}}) = e^{-\gamma\delta t} < 1$. Dividing Jury's condition $|\mathrm{tr}(A(\delta t))| < 1 + e^{-\gamma\delta t}$ by $(1 + e^{-\gamma\delta t}) > 0$ yields $|1 - \frac{1}{2}\alpha_{\mathrm{eff}}\delta t^2| < 1 \iff 0 < \delta t < 2/\sqrt{\alpha_{\mathrm{eff}}}$. \hfill $\blacksquare$

\textbf{Note on Why Finite-Horizon ODE Error Bounds Do Not Replace Conformal Calibration:} Across all 10 trained seeds on FD001, the worst-case attractive perturbation of the nonlinear force is $L_{\mathrm{nl}}^{(-)} = \frac{0.65}{d}\|W_{\mathrm{pot},-}\|_2^2 + c_f\beta_q = 0.2242 \pm 0.0237 < \alpha_{\mathrm{eff}} = 0.35$, so the potential $V_\phi + U_\psi$ is strongly convex around the anchor ($\underline{\kappa} = \alpha_{\mathrm{eff}} - L_{\mathrm{nl}}^{(-)} = 0.1258 \pm 0.0237 > 0$). However, global two-trajectory contraction under the centered-stiffness LMI requires damping $\gamma \ge 0.42\text{ s}^{-1}$, whereas the default model uses $\gamma = 0.15\text{ s}^{-1}$. In the mechanical energy norm $\|z\|_E = \sqrt{\alpha_{\mathrm{eff}}\|q\|_2^2 + \|p\|_2^2}$ (where the linear generator $A_{\mathrm{lin}}$ has logarithmic norm $\mu_E(A_{\mathrm{lin}}) = 0$ and $\|e^{A_{\mathrm{lin}}s}\|_E \le 1$), Duhamel--Gr\"{o}nwall error bounds grow as $e^{\nu_E H_w}$ ($\nu_E = L_{\mathrm{nl,full}}/\sqrt{\alpha_{\mathrm{eff}}} = 0.6295\text{ s}^{-1}$) and, when multiplied by the readout Lipschitz constant $L_{g,\mathrm{spec}} = 115.30\text{ cycles/unit}$, yield RUL bounds ($247.1$ cycles on $h_w=0.35$) that exceed the $125$-cycle cap. Operational RUL safety guarantees must therefore rely on conformal calibration (Section~\ref{sec:method_conformal}).

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

with open("manuscript/main.tex", "w") as f:
    f.write(tex_content)
print("Wrote re-scoped manuscript/main.tex")
