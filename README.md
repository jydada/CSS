# CSS
In this paper, we present a novel computerized spermatogenesis staging (CSS) pipeline for identifying Stages I-III from Stages IV-V. First, a multi-scale learning (MSL) model is employed to integrate local and global seminiferous tubule information to distinguish Stages I-V from Stages VI-XII. Then, a multi-task learning (MTL) model is developed to segment the multiple testicular cells (MTCs) without exhaustive requirement for human annotations. Finally, we develop 204-dimensional quantitative image-derived features for discriminating Stages I-III from Stages IV-V.

The code in this progject will reproduce the results in our paper submitted to Bioinformatics, "A novel pipeline for mouse testicular computerized spermatogenesis staging". Code is written in matlab and python.

# Outline

* Dataset
* Methods
* Experiment setup
* Contact information

# Dataset 
* Mouse testicular cross-sections
* Multi-organ Nucleus Segmentation (MoNuseg) Challenge
* Prostate lumen segmentation (PLSeg)
* Epithelium & stroma segmentation (EPSTSeg)


# Methods
The CSS pipeline comprises four parts: 
1) Seminiferous tubule segmentation is developed based on Channel and Spatial attention net (CSNet) at x10 resolution
![image](https://github.com/jydada/CSS/blob/main/seminiferous%20segmentation/STseg.jpg)
2) A multi-scale learning (MSL) model is developed to integrate local and global seminiferous tubule information to distinguish Stages I-V from Stages VI-XII
![image](https://github.com/jydada/CSS/blob/main/MSL-classification%20for%20Stages%20I-V/MSL.jpg)
3) A multi-task learning (MTL) model is developed to segment the multiple testicular cells (MTCs) without exhaustive requirement for human annotation
![image](https://github.com/jydada/CSS/blob/main/MTL-MTCs%20Segmentation/MTL_1.jpg)
4) We develop a novel set of image-derived features for discriminating Stages IV-V from Stages I-III
![image](https://github.com/jydada/CSS/blob/main/Image-derived%20features/feature.jpg)

# Experiment setup
* Experiment 1: Seminiferous tubule segmentation
* Experiment 2: Multi-scale learning (MSL) based SE-Resnet50 model for distinguish Stages I-V from Stages VI-XII
* Experiment 3: Multi-task learning (MTL) based CSNet model for segmenting the multiple testicular cells (MTCs) in a seminiferous tubule
* Experiment 4: Quantitative image-derived features extraction and classifer construction


# Contact information
If you have any questions, feel free to contact me.
Haoda Lu, Nanjing University of Information Science and Technology, Nanjing, China. Email: haoda@nuist.edu.cn
