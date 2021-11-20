# Session-based-Recommendation-via-Contrastive-Learning-on-Heterogeneous-Graph

This repo covers our implementation of our paper **Session-based Recommendation via Contrastive Learning on Heterogeneous Graph** accepted by **2021 IEEE Bigdata**

## Setup Environment

### Package Versions
Python = 3.7.9
Pytorch = 1.8.0
Pandas = 1.1.4

### Hardware Suggestions
GPU Memory Capacity > 12GB

## Prepare Contrastive Learning Procedures:
You can directly execute the corresponding files within a pycharm project.

### For Yelp
The raw data file can be put under /Yelp file

Converting json to csv files:
```
python Yelp_json2csv.py
```

Preprocessing and filtering the Yelp dataset:
```
python preprocessing_Yelp.py
```

Processing the meta-path for Yelp:
```
python utils/meta-path.py
```

### For Tmall
The raw datacan be put under /Tmall file 

Preprocessing and filtering Tmall dataset:
```
python Tmall/tmall_preprocess.py
```

Processing the meta-path for Tmall:
```
python mp_tmall.py
```

## Pre-training
You can directly execute pre_train.py in a pycharm project,
or you can use
```
python pre_train.py
```
but be sure the default values for parameters are settled correctly.

## Fine-tuning
You can directly execute finetune.py in a pycharm project,
or you can use
```
python finetune.py
```
but be sure the default values for parameters are settled correctly.



