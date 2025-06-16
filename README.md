# DuEnNet
DuEnNet: CNN and Swin Transformer Dual Encoder Network for Medial Image Segmentation

## Data
The preprocessed data of [Synapse Dataset](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [ACDC Dataset](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link) we used are provided by TransUNet's authors.

## Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies. We set the batch size as 8 for both Synapse and ACDC dataset on a GPU P100 (16G).

## Quantitative Results
### Synapse
Synapse multi-organ segmentation dataset contains 8 organs: aorta (Ao), gallbladder (GB), spleen (Spl), left kidney (LK), right kidney (RK), liver (Liv), pancreas (Panc), and stomach (St), with evaluation metrics Dice Similarity coefficient (DSC) and the 95th percentile Hausdorff Distance (HD95).

| Methods   | DSC   | HD95  | Ao    | GB    | LK    | RK    | Liv   | Panc  | Spl   | St    |
|:----------|:------|:------|:------|:------|:------|:------|:------|:------|:------|:------|
| UNet      | 76.85 | 39.70 | 89.07 | 69.72 | 77.77 | 68.60 | 93.43 | 53.98 | 86.67 | 75.58 |
| TransUNet | 77.48 | 31.69 | 87.23 | 63.13 | 81.87 | 77.02 | 94.08 | 55.86 | 85.08 | 75.62 |
| SwinUnet  | 79.13 | 21.55 | 85.47 | 66.53 | 83.28 | 79.61 | 94.29 | 56.58 | 90.66 | 76.60 |
| TransFuse | 80.63 | 23.13 | 86.70 | 68.03 | 84.26 | 78.87 | 93.55 | 62.33 | 89.78 | 81.53 |
| HiFormer  | 80.69 | 19.14 | 87.03 | 68.61 | 84.23 | 78.37 | 94.07 | 60.77 | 90.44 | 82.03 |
| Ours      | 82.80 | 17.66 | 86.91 | 72.46 | 85.51 | 82.48 | 94.83 | 68.30 | 91.75 | 80.17 |

### ACDC
Automated cardiac diagnosis challenge dataset contains 3 organs: left ventricle (LV), right ventricle (RV), and myocardium (Myo).

| Methods   | DSC   | HD95   | RV    | Myo   | LV    |
|:----------|:------|:-------|:------|:------|:------|
| UNet      | 88.33 | 3.8372 | 87.57 | 83.60 | 93.83 |
| TransUNet | 88.96 | 1.9955 | 87.49 | 83.70 | 95.69 |
| SwinUnet  | 89.27 | 2.8116 | 88.07 | 84.78 | 94.96 |
| TransFuse | 89.65 | 1.9794 | 87.75 | 85.38 | 95.82 |
| HiFormer  | 89.08 | 2.1642 | 87.64 | 84.36 | 95.25 |
| Ours      | 90.20 | 1.2995 | 89.62 | 85.88 | 95.11 |

## References
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
