# AldOpt
This is a collection of codes developed for Paulson et al. (2021).

## Instruction to reproduce paper results
1. Run `uptake_curve.py` to generate uptake curve figures for each ALD system and some .csv files needed by ald_opt.py (for plotting purposes only). This will produce Figure 1a and Figures S1a, S2a, and S3a.
2. Run `uptake_curve_trad.py` to generate traditional uptake curve figures for each ALD system. This will produce Fig. 1b, and Figures S1b, S2b, and S3b.
3. Run `ald_cost_sensitivity.py` to explore the sensitivity of the C_var1 and C_var2 cost function components for different gas timings and imposed measurement noise levels. This will generate Fig. 4.
4. Run `ald_opt_sensitivity_initcond.py` to explore the sensitivity of the expert systems optimization to initial gas timing guesses. Change `noise_imposed` on line 210 to explore different levels of imposed measurement noise. This will generate Fig. 5.
5. Run `ald_opt_sensitivity_nrep.py` to explore optimization sensitivity to different numbers of repeated cycles with the same gas timings under different imposed measurement noise. Change `noise_imposed` on line 207 to explore different levels of imposed measurement noise, and `acq` on line 201 to explore each optimization algorithm ('random' for random optimization, 'phys' for expert systems optimization, and 'EI' for Bayesian optimization). This will generate Fig. 6 and 7.
6. Run `ald_opt.py` to explore optimization performance for the 4 ALD systems and 3 optimization strategies with changing imposed measurement noise. This will generate Figures 8-10 and S5-S13.

## Other codes
* `core.py` and `sopt.py` contain common functions used for plotting, output, surrogate modeling, and optimization.
* `ALDmodel.py` contains the basic ALD model considering the interaction of two precursors with a growing surface.
* `physicsmodel.py` contains the expert systems optimization algorithm.
* `maximin_lhs_l2_10_4d.csv`, and `maximin_lhs_l2_20_4d.csv` are 4-dimensional optimized latin-hypercube designs for 10 and 20 samples, respectively.

## Required packages
* python/3.7.6
* matplotlib/3.1.1
* numpy/1.19.2
* pandas/1.0.3
* scikit-learn/0.24.1
* scipy/1.4.1
* seaborn/0.9.0

## Paper reference
Paulson, N.H., Yanguas-Gil, A., Abuomar, O.Y., Elam, J.W. “Intelligent agents for the optimization of atomic layer deposition,” ACS Applied Materials and Interfaces. (2021) Accepted