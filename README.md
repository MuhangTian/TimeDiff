# Reliable Generation of Privacy-preserving Synthetic EHR Time Series via Diffusion Models
[![DOI](https://img.shields.io/badge/DOI-10.1093/jamia/ocae229-blue.svg)](https://doi.org/10.1093/jamia/ocae229)

This repository contains the code implementation for TimeDiff, a diffusion model designed for the generation of mixed-type time series data in EHR. Our work is accepted by the Journal of the American Medical Informatics Association (JAMIA).

## How to Run TimeDiff
* Use `requirements.txt` to install dependencies.
* `etdiff_train.py` is the script to initialize model training. After training is complete, it will automatically generate synthetic EHR time series data and store it in the designated directory set by the user. This directory can be specified with `--check_point_path` command line argument.
* The specific implementation of the model can be found at `models/ETDiff`.

## Data Pre-processing
Please see `preprocess` directory for Jupyter notebooks and Python scripts on how we preprocess EHR data for model training and evaluation.

## Baselines
***NOTE: Please set `PYTHONPATH` correctly in order to run the baseline models.***

### Dependencies for the Baselines
We recommend using independent environments for each of the baseline models, as some of them have distinctively different requirements (one example is the version of Python for tensorflow 1.X). Please refer to the code repository for the baselines and Appendix A.4.4 in our paper for more details.

* **EHR-M-GAN**: run `baselines/ehrmgan_train.py`; implementation is stored at `models/ehr_m_gan`.
* **DSPD/CSPD**: run `baselines/dspd_train.py`; implementation is stored at `models/tsdiff`; to use CSPD rather than DSPD, use continuous diffusion for `--diffusion` argument.
* **GT-GAN**: run `baselines/gtgan_train.py`; implementation is stored at `models/gt_gan`.
* **TimeGAN**:  run `baselines/timegan_train.py`; implementation is stored at `models/time_gan`.
* **RCGAN**: run `baselines/rcgan_train.py`; implementation is stored at `models/rc_gan`.
* **C-RNN-GAN**: run `baselines/crnngan_train.py`; implementation is stored at `models/crnn_gan`.
* **T-Forcing/P-Forcing**: run `baselines/pt_forcing_train.py`; implementation is stored at `models/p_or_t_forcing`; passing the command line argument `-adversarial` uses P-Forcing. Otherwise, T-Forcing is used.
* **HALO**: please refer to `baselines/halo_preprocess.py` and the descriptions provided by the HALO authors in their manuscript to preprocess the input data. Run `baselines/halo_train.py` to train the model and `baselines/halo_generate.py` for sampling. Implementation is stored at `models/halo`.

## Evaluation Metrics
* `eval_samples.py` is the script responsible for running all the evaluation metrics discussed in the paper; to run the code, pass the intended metric to use via `--metric` command line argument, as well as the training data (`--train_path`), testing data (`--test_path`), and synthetic data (`--sync_path`).
* Implementations for data utility metrics can be found at `evaluate/utility.py`.
* Implementations for privacy risk metrics can be found at `evaluate/privacy.py`.
