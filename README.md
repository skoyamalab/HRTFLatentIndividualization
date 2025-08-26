# HRTFLatentIndividualization
[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3921/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.6-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the official implementation of <strong>"Head-Related Transfer Function Individualization Using Anthropometric Features and Spatially Independent Latent Representation"</strong> in Proceedings of <em>IEEE Workshop on Applications of Signal Processing to Audio and Acoustics</em> [[Preprint](https://arxiv.org/abs/2508.16176)]. We propose a HRTF individualization method using the latent representation of HRTF magnitude obtained through an autoencoder conditioned on sound source positions, which makes it possible to combine multiple HRTF datasets with different measured source positions, and makes the network training tractable by reducing the number of parameters to be estimated from anthropometric parameters. Experimental evaluation shows that high estimation accuracy is achieved by the proposed method, compared to current DNN-based methods.

The Frequency Independent Autoencoder (FIAE) is based on [FSP-AE](https://github.com/ikets/FSP-AE).

If you use this code in your experiments, please cite [1] in your work.

## System Requirements
Tested on the following environment:
- OS: Ubuntu 22.04, 24.04
- GPU: NVIDIA RTX A6000 Ada (48GB VRAM)
- CUDA 12.4
- Python 3.9

## Installation (with pyenv-virtualenv)
1. Install `pyenv` and `pyenv-virtualenv`.
2. Clone the repository:
     ```bash
     git clone git@github.com:skoyamalab/HRTFLatentIndividualization.git
     cd HRTFLatentIndividualization
     ```
3. Setup the virtual environment:
     ```bash
     pyenv install 3.9.21
     pyenv virtualenv 3.9.21 hrtf_li
     pyenv activate hrtf_li
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
4. Download and extract HRTF datasets (you can find many at the [SOFA conventions website](https://www.sofaconventions.org/mediawiki/index.php/Files)).
     - Currently, only CIPIC [3] and HUTUBS [4,5] have been tested. If you want to work with a new dataset with anthropometric features, you will need to modify both `dataset.py` (for the autoencoder) and `multidataset.py` (for the individualization models) to support it. After that, you can update the configurations accordingly.
5. Organize your project directory as follows:
   ```
   ├── Dataset 1 (ex. CIPIC)
   ├── Dataset 2 (ex. HUTUBS)
   └── HRTFLatentIndividualization/
       ├── src/
       │   ├── <files within src>
       └── main.py
   ```

## Training the FIAE
**Steps:**
1.  In `src/models.py`, ensure the `utils` import is `src.utils`, not `utils`.
2.  In `src/configs.py`, modify the `database`, `sub_index`, and `idx_plot_list` variables to specify the datasets and the desired train/validation/test splits.
3.  **Run the training script:**
    ```bash
    nohup python3 -u main.py -n 2 -m "FIAE" -c 500239 -a outputs/out_20240918_FIAE_500239 > "outputs/out_20240918_FIAE_500239/log.txt" &
    ```
**Notes:**
- You can only use `-n > 1` (more than one GPU) when training on a single dataset.
- Only `-m "FIAE"` is currently supported.
- The `-c 500239` argument corresponds to the configuration number in `configs.py`.
- The `-a` flag specifies the artifacts directory; you can change this to any path.
- Training logs will be saved to `outputs/out_20240918_FIAE_500239/log.txt`.
- TensorBoard logs can be found in `outputs/out_20240918_FIAE_500239/logs/`.
- Models will be saved at epochs 500, 1000, 1400 (final), and the best performing model will also be saved.

## Training the HRTF Individualization Models
**Steps:**
1.  In `src/models.py`, ensure the `utils` import is `utils`, not `src.utils`.
2.  Modify the `sub_indicies` in `src.multidataset.py` to change the train/test/skip indices or to add a new dataset.
3.  **Run the training script:**
    ```bash
    python3 src/[proto/direct]_diffusion.py
    ```
    
    or
    
    ```bash
    python3 src/[proto/direct]_DNN.py
    ```
**Notes:**
- In the chosen script, make sure the `train_db_names` and `test_db_names` (or `db_name` in `direct_DNN.py`) reflect the datasets you want to use for training and testing.
- Ensure the `output_dir` variable points to the directory where you saved the autoencoder artifacts (e.g., `outputs/out_20240918_FIAE_500239`). The model name should correspond to the order of the names in `train_db_names` (e.g., `hrtf_approx_network.best_CIPIC_HUTUBS.net`).
- You can modify the training configurations in the `main` method of each script. The provided values are the recommended defaults for each model.
- The `save_prefix` and `training_configs` also allow you to specify where model artifacts, TensorBoard logs, and figures will be saved.

## Cite
```bibtex
@article{Niu:WASPAA2025,
  author    = {Ryan Niu and Shoichi Koyama and Tomohiko Nakamura},
  title     = {Head-Related Transfer Function Individualization Using Anthropometric Features and Spatially Independent Latent Representation},
  journal   = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics},
  year      = {2025},
  publisher = {IEEE},
  note      = {(accepted)}
}
```

## License
[CC-BY-4.0](https://github.com/ikets/FSP-AE/blob/main/LICENSE)

## References
[1] Ryan Niu, Shoichi Koyama, and Tomohiko Nakamura, <strong>"Head-Related Transfer Function Individualization Using Anthropometric Features and Spatially Independent Latent Representation,"</strong> <em>IEEE WASPAA</em>, Oct., 2025. (accepted) [[Preprint](https://arxiv.org/abs/2508.16176)] <br>
<!-- vol. XX, pp. xxxx-xxxx, 2025.  [[PDF]]()  -->

[2] Yuki Ito, Tomohiko Nakamura, Shoichi Koyama, Shuichi Sakamoto, and Hiroshi Saruwatari, <strong>"Spatial Upsampling of Head-Related Transfer Function Using Neural Network Conditioned on Source Position and Frequency,"</strong> <em>IEEE Open J. Signal Process.</em>, 2025.  (accepted) <br>

[3] Duda, R.O. & Thompson, D.M. & Avendaño, Carlos. (2001). The CIPIC HRTF database. IEEE ASSP Workshop on Applications of Signal Processing to Audio and Acoustics. 99 - 102. 10.1109/ASPAA.2001.969552. 

[4] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Jan Joschka Wohlgemuth, Fabian Seipel, Daniel Voss, Peter Grosche, and Stefan Weinzierl, “The HUTUBS head-related transfer function (HRTF) database,” 2019, url: http://dx.doi.org/10.14279/depositonce-8487 (accessed May 6, 2022).<br>

[5] Kanji Watanabe, Yukio Iwaya, Yôiti Suzuki, Shouichi Takane, and Sojun Sato, “Dataset of head-related transfer functions measured with a circular loudspeaker array,” <em>Acoust. Sci. Tech.</em>, vol. 35, no. 3, pp. 159–165, 2014.<br>

