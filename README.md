<img align="left" width="200" src="uniform-logo.jpg" alt="UniFORM Logo">

# UniFORM: Towards Universal ImmunoFluorescence Normalization for Multiplex Tissue Imaging

<p align="center">
    Official GitHub documentation of the UniFORM Normalization Project
</p>



## :star2: About the Project

### :camera: Overview

UniFORM normalization pipeline operates both on pixel-level and feature-level, enabling robust, scalable preprocessing for high-dimensional CyCIF data. This pipeline allows users to normalize CyCIF images and data across multiple samples to facilitate comparative and quantitative analysis.



### :dart: Tech Stack

<ul>
    <li>numpy</li>
    <li>matplotlib</li>
    <li>pandas</li>
    <li>dask</li>
    <li>zarr</li>
    <li>scikit-image</li>
    <li>tifffile</li>
    <li>pyometiff</li>
    <li>scikit-learn</li>
    <li>scipy</li>
    <li>scikit-fda</li>
</ul>

## 	:toolbox: Getting Started

<!-- Installation -->
:gear: Installation
It is highly recommended to create a virtual environment for the UniFORM normalization. You can set up the environment using the provided YAML file:

You can install the latest version of the packages above using environment.yml
```bash
conda env create -f environment.yml
conda activate cycif-normalization-env
```


:test_tube: Workflow
```UniFORM-calculate-histogram.py``` -> ```UniFORM-landmark-finetuning.py (optional)``` --> ```UniFORM-normalization.py```

### Exploratory Data Analysis
To begin the normalization process, first analyze the data using:


