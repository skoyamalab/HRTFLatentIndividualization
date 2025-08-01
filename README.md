# HRTFLatentIndividualization

## Abstract

A method for head-related transfer function (HRTF) individualization from the subject's anthropometric parameters is proposed. Due to the high cost of measurement, the number of subjects included in many HRTF datasets is limited, and the number of those that include anthropometric parameters is even smaller. Therefore, HRTF individualization based on deep neural networks (DNNs) is a challenging task. We propose a HRTF individualization method using the latent representation of HRTF magnitude obtained through an autoencoder conditioned on sound source positions, which makes it possible to combine multiple HRTF datasets with different measured source positions, and makes the network training tractable by reducing the number of parameters to be estimated from anthropometric parameters. Experimental evaluation shows that high estimation accuracy is achieved by the proposed method, compared to current DNN-based methods.

-----

## How to Run the Code

### Prerequisites:

1.  **Install dependencies:**

      * It is recommended to use a virtual Python environment.
      * Install the required packages using the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Download HRTF datasets:**

      * You can find various HRTF datasets at the [SOFA conventions website](https://www.sofaconventions.org/mediawiki/index.php/Files).

      * Organize your project directory as follows:

        ```
        ├── Dataset 1 (ex. CIPIC)
        ├── Dataset 2 (ex. HUTUBS)
        └── HRTFLatentIndividualization/
            ├── src/
            │   ├── <files within src>
            └── main.py
        ```

-----

### Training the FIAE AutoEncoder

1.  **Set the correct import:**

      * In `src/models.py`, ensure the `utils` import is `src.utils`, not `utils`.

2.  **Configure your training:**

      * In `src/configs.py`, modify the `database`, `sub_index`, and `idx_plot_list` variables to specify the datasets and the desired train/validation/test splits.

3.  **Run the training command:**

    ```bash
    nohup python3 -u main.py -n 2 -m "FIAE" -c 500239 -a outputs/out_20240918_FIAE_500239 > "outputs/out_20240918_FIAE_500239/log.txt" &
    ```

      * **Notes:**
          * You can only use `-n > 1` (more than one GPU) when training on a single dataset.
          * Only `-m "FIAE"` is currently supported.
          * The `-c 500239` argument corresponds to the configuration number in `configs.py`.
          * The `-a` flag specifies the artifacts directory; you can change this to any path.
          * Training logs will be saved to `outputs/out_20240918_FIAE_500239/log.txt`.
          * TensorBoard logs can be found in `outputs/out_20240918_FIAE_500239/logs/`.
          * Models will be saved at epochs 500, 1000, 1400 (final), and the best performing model will also be saved.

-----

### Training the HRTF Individualization Models

1.  **Set the correct import:**

      * In `src/models.py`, ensure the `utils` import is `utils`, not `src.utils`.

2.  **Configure your dataset splits:**

      * Modify the `sub_indicies` in `src.multidataset.py` to change the train/test/skip indices or to add a new dataset.

3.  **Run the appropriate training script:**

      * Execute one of the following scripts:
          * `python/[proto/direct]_diffusion.py`
          * `python/[proto/direct]_DNN.py`
      * **Notes:**
          * In the chosen script, make sure the `train_db_names` and `test_db_names` (or `db_name` in `direct_DNN.py`) reflect the datasets you want to use for training and testing.
          * Ensure the `output_dir` variable points to the directory where you saved the autoencoder artifacts (e.g., `outputs/out_20240918_FIAE_500239`). The model name should correspond to the order of the names in `train_db_names` (e.g., `hrtf_approx_network.best_CIPIC_HUTUBS.net`).
          * You can modify the training configurations in the `main` method of each script. The provided values are the recommended defaults for each model.
          * The `save_prefix` and `training_configs` also allow you to specify where model artifacts, TensorBoard logs, and figures will be saved.

-----

### Other

  * **Adding a new dataset:**
      * If you want to add a new dataset with anthropometric features, you will need to modify both `dataset.py` (for the autoencoder) and `multidataset.py` (for the individualization models) to support it. After that, you can update the configurations accordingly.
