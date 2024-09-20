# DuEnNet
DuEnNet: CNN and Swin Transformer Dual Encoder Network for Medial Image Segmentation


## Data
The preprocessed data of [Synapse Dataset](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [ACDC Dataset](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link) we used are provided by TransUNet's authors.


## Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies. We set the batch size as 8 for both Synapse and ACDC dataset on a GPU P100 (16G).


## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
