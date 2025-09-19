# How to use tuning_mcmc.ipynb
1. Install the required Python package in `tuning_mcmc.ipynb`, `emulator_fun.py`, `tuning_fun.py`, etc.
## HP tuning
### preprocessing
2. Prepare your PPE summary file as follows:
   1. have the following global attributes:
        - 'n_init': number of initial condition variables perturbed by PPE [int]
        - 'init_var': names of (varying) initial condition variables [string]
        - 'thresholds_eff0': thresholds for the constraint variables [float] (in theory can be optional but required in practice right now)
        - 'var_constraints': names of constraint variables [string]
        - 'n_param_*': number of parameters for each parameter group [int]
        - 'is_perturbed_*': whether the parameter group is perturbed by PPE [int]
   2. have the following dimensions:
        - ncases: number of cases run by target model
        - nppe: number of PPE members
        - nparams: number of parameters used by the model being trained
   3. have the following variables:
        - **param_names** (nparams): names of parameters [string] (optional but recommended)
        - **[init_var]_PPE** (nppe): initial condition used by PPE members [float]
        - **ppe_[var_constraints]** (nppe): constraint variables from each PPE member [float]
        - **params_PPE** (nppe, nparams): parameters used by the model being trained [float]
        - **tgt_[var_constraints]** (ncases): constraint variables from the target model [float]
        - **case_[init_var]** (ncases): initial condition used by the target model [float]
3. Pick a `transform_method` (in string or a list of strings):
   - `standard_scaler`
   - `standard_scaler_asinh`: smooth log-linear transition at `thresholds_eff0` specified in the summary file. Preferred for variables ranging across multiple orders of magnitude but insignificantly small values don't have a large impact on the training/sampling (unlike log transformation).
   - `standard_scaler_log`
   - `minmaxscale_asinh`: also need to specify `threshold_eff0`
   - `minmaxscale` <br>
4. Pick a ML architecture:
   - CRPS (preferred)
     - `build_reg_crps_model` in `tuning_fun.py`
   - multi-output with NLL uncertainty
     - `build_classreg_unc_model` in `tuning_fun.py`
     - predicts both the presence and the value and use negative log likelihood for uncertainty calculation in the loss function. <br>
  (**Note**: range and steps for hyperparameter tuning can be changed in their respective function.)
### validation
5. After running hyperparameter tuning and neural network training, go to the `validation` subsection in `HP tuning` and run `ef.plot_emulator_results` to see if the PPE emulator is working properly.
## MCMC:
6. Load the prior parameters (not for MCMC but necessary for better visualization and saving the complete distribution)
7. Specify the following MCMC sampling:
   - `nchain`, `num_burnin_steps`, `num_samples`
8. Default values should be fine but
   - `tau` for granular control over the soft presence gate (irrelevant for the current implementation using CRPS, which doesn't predict presence of water)
   - `inflate_factor` for manually inflating the log likelihood if the observation doesn't constrain the parameters at all, or deflating if the chains are stuck or not well mixed.
### Saving MCMC results:
9. Save the complete posterior distribution into a netCDF file. (**Note**: only the perturbed parameters are saved as specified by `params_train['param_interest_idx']`, not all parameters in `params_train['pnames']`.)
10. Save only the MAP, standard deviation, and inflated standard deviation of the posterior. Good enough if the posterior itself is Gaussian.
