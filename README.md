# Reliable Generation of Privacy-preserving Synthetic EHR Time Series via Diffusion Models
[![arxiv badge](https://img.shields.io/badge/arXiv-2310.15290-red)](https://arxiv.org/abs/2310.15290)

This repository contains the code implementation for TimeDiff, a diffusion model designed for the generation of mixed-type time series data in EHR. Our work is accepted by the Journal of the American Medical Informatics Association (JAMIA).

## How to Run TimeDiff
* Use `requirements.txt` to install dependencies
* `etdiff_train.py` is the script to initialize model training. After training is complete, it will automatically generate synthetic EHR time series data and store it in the designated directory set by the user. This directory can be specified with `--check_point_path` command line argument
* The specific implementation of the model can be found at `models/ETDiff`

## Baselines
* **EHR-M-GAN**: run `ehrmgan_train.py`; implementation is stored at `models/ehr_m_gan`
* **DSPD/CSPD**: run `dspd_train.py`; implementation is stored at `models/tsdiff`; to use CSPD rather than DSPD, use continuous diffusion for `--diffusion` argument
* **GT-GAN**: run `gtgan_train.py`; implementation is stored at `models/gt_gan`
* **TimeGAN**:  run `timegan_train.py`; implementation is stored at `models/time_gan`
* **RCGAN**: run `rcgan_train.py`; implementation is stored at `models/rc_gan`
* **C-RNN-GAN**: run `crnngan_train.py`; implementation is stored at `models/crnn_gan`
* **T-Forcing/P-Forcing**: run `pt_forcing_train.py`; implementation is stored at `models/p_or_t_forcing`; passing the command line argument `-adversarial` uses P-Forcing. Otherwise, T-Forcing is used
* **HALO**: please refer to `halo_preprocess.py` and the descriptions provided by the HALO authors in their manuscript to preprocess the input data. Run `halo_train.py` to train the model and `halo_generate.py` for sampling. Implementation is stored at `models/halo`

## Dependencies for the Baselines
We recommend using independent environments for each of the baseline models, as some of them have distinctively different requirements (one example is the version of Python for tensorflow 1.X). Please refer to the code repository for the baselines and Appendix A.4.4 in our paper for more details.

## Evaluation Metrics
* `eval_samples.py` is the script responsible for running all the evaluation metrics discussed in the paper; to run the code, pass the intended metric to use via `--metric` command line argument, as well as the training data (`--train_path`), testing data (`--test_path`), and synthetic data (`--sync_path`)
* Implementations for data utility metrics can be found at `evaluate/utility.py`
* Implementations for privacy risk metrics can be found at `evaluate/privacy.py`
* `to_summary_stats.py` turns time series data into summary statistics discussed in Appendix B.4.
* `visualize.py` creates time series visualizations for Appendix B.5. The mean and $\pm$ standard deviation visualizations are created with `eval_samples.py` by setting the metric argument as `--metric trajectory`
