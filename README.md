<div align="center">

  <h1> UniFORM: Towards Universal ImmunoFluorescence Normalization for Multiplex Tissue Imaging </h1>
  
  <p>
    Official GitHub documentation of the UniFORM Normalization Project
  </p>
  
</div>


  

## :star2: About the Project


### :camera: Overview

UniFORM normalization pipeline both on pixel-level and feature-level, enabling robust, scalable preprocessing for high-dimensional CyCIF data. This pipeline allows users to normalize CyCIF images and data across multiple samples to facilitate comparative and quantitative analysis.


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
It is highly recommended to create a virtual environment for the CyCIF normalization. You can set up the environment using the provided YAML file:

You can install the latest version of the packages above using environment.yml
```bash
conda env create -f environment.yml
conda activate cycif-normalization-env
```
In case there's package comnflicts or incompatibility, you can use my environment setup
```bash
conda env create -f environment-mark-version.yml
conda activate cycif-normalization-env
```

:test_tube: Workflow
```exploratory_data_analysis.py``` -> ```normalize.py``` --> ```image_transformation.py```

### Exploratory Data Analysis
To begin the normalization process, first analyze the data using:



