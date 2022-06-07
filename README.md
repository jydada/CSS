# CSS-BOEHD-features
In this paper, we present a novel computerized spermatogenesis staging (CSS) pipeline for identifying Stages IV-V. First, a multi-task learning (MTL) model is developed to segment the multiple testicular cells (MTCs) without exhaustive requirement for human annotations. Then, a multi-scale learning (MSL) model is employed to integrate local and global seminiferous tubule information to distinguish Stages I-V from Stages VI-XII. Finally, we develop a new graph-based feature called bag-of-edge histogram and distance (BOEHD) for discriminating Stages IV-V from Stages I-III.

The code in this progject will reproduce the results in our paper submitted to Bioinformatics, "Multi-task learning based histomorphological analysis for mouse testicular cross-sections: towards computerized spermatogenesis staging". Code is written in matlab and python.

## Outline

* Downloading the histological images of mouse testicular cross-section
* Methods
* How to run code
* Contact information
* 
# Downloading the histological images of mouse testicular cross-section
Image data can be downloaded at here (3.37GB). If you don't want to downlad the data and just want to see the codes and resutls, that is okay, because all intermediate and final results are already included.


# Methods
The CSS pipeline comprises three parts: 1) A multi-task learning (MTL) model is developed to segment the multiple testicular cells (MTCs) without exhaustive requirement for human annotation; 2) A multi-scale learning (MSL) model is employed to integrate local and global seminiferous tubule information to distinguish Stages I-V from Stages VI-XII; 3) We develop a new graph-based feature called bag-of-edge histogram and distance (BOEHD) for discriminating Stages IV-V from Stages I-III.
# Downloading the histological images of mouse testicular cross-section

# Experiment setup
## Experiment 1: Seminiferous tubule segmentation
## Experiment 2: Multi-scale learning (MSL) based Resnet-50 model for distinguish Stages I-V from Stages VI-XII
## Experiment 3: Multi-task learning (MTL) based CSNet model for segmenting the multiple testicular cells (MTCs) in a seminiferous tubule
## Experiment 4: BOEHD features extraction and classifer construction


# Contact information
If you have any questions, feel free to contact me.
Haoda Lu, Nanjing University of Information Science and Technology, Nanjing, China. Email: jydada2018@gmail.com
