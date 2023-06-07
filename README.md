# HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification

Official implementation for ACL 2023 accepted paper "HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification" . [[arXiv](https://arxiv.org/abs/2305.15182)][[pdf](https://arxiv.org/pdf/2305.15182.pdf)]

## Requirements

**It's hard to reproduce the results without the same devices and environment.** Although our code is highly compatible with python environments, we strongly recommend that you create a new environment according to our settings.

- Python == 3.7.13
- numpy == 1.21.5
- PyTorch == 1.11.0
- scikit-learn == 1.0.2
- transformers == 4.19.2
- numba == 0.56.2
- glove.6B.300d

## Data preparation

Please manage to acquire the original datasets and then run these scripts.

### Web Of Science (WOS)

The original dataset can be acquired in the repository of [HDLTex](https://github.com/kk7nc/HDLTex). Preprocess code could refer to the repository of [HiAGM](https://github.com/Alibaba-NLP/HiAGM). Please download the release of WOS-46985(version 2), open `WebOfScience/Meta-data/Data.xls` and convert it to `.txt` format (Click "Save as" in Office Excel). Then, run:
```shell
python data/preprocess_wos.py
```


### NyTimes (NYT)

The original dataset can be acquired [here](https://catalog.ldc.upenn.edu/LDC2008T19).  When you have fetched the original archive,  Unzip the files and make sure that the file path matches the indices in `data/idnewnyt_xxx.json`. Here we post our bash script in `nyt.sh` , which takes hours to accomplish the preprocessing (You could manage by your own way). 

```shell
bash nyt.sh
python data/preprocess_nyt.py
```

### RCV1-V2

The preprocessing code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement. It took us 1 data to receive a response.

```shell
python data/data_rcv1.py
python data/preprocess_rcv1.py
```

### Conduct experiments on your own data
In [HiAGM](https://github.com/Alibaba-NLP/HiAGM), an additional step is required to count the prior probabilities between parent and child labels by running `python helper/hiearchy_tree_statistic.py`. HiTIN only requires unweighted adjacency matrix of label hierarchies but we still retain this property and save the statistics in `data/DATASET_prob.json` as we also implement baseline methods including TextRCNN, BERT-based HiAGM. 

If you tend to evaluate these methods on your own dataset, please make sure to organize your data in the following format:
```
{
    "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}

where "doc_keyword" and "doc_topic" are optional.
```
then, replace the label name with your dataset's in line143~146 of `helper/hierarchy_tree_statistics.py` and run:
```shell
python helper/hierarchy_tree_statistic.py
```

> Thanks to the superior framework open-sourced by [NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier). You could also implement other methods in HTC or propose your own model.

## Train
The default parameters are not the best performing-hyper-parameters used to reproduce our results in the paper. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

To learn hyperparameters to be specified, please see: 
```
python train.py [-h] -cfg CONFIG_FILE [-b BATCH_SIZE] [-lr LEARNING_RATE]
                [-l2 L2RATE] [-p] [-k TREE_DEPTH] [-lm NUM_MLP_LAYERS]
                [-hd HIDDEN_DIM] [-fd FINAL_DROPOUT] [-tp {root,sum,avg,max}]
                [-hp HIERAR_PENALTY] [-ct CLASSIFICATION_THRESHOLD]
                [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]
                [--begin_time BEGIN_TIME]

optional arguments:
  -h, --help            show this help message and exit
  -cfg CONFIG_FILE, --config_file CONFIG_FILE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        input batch size for training (default: 32)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  -l2 L2RATE, --l2rate L2RATE
                        L2 penalty lambda (default: 0.01)
  -p, --load_pretrained
  -k TREE_DEPTH, --tree_depth TREE_DEPTH
                        The depth of coding tree to be constructed by CIRCA
                        (default: 2)
  -lm NUM_MLP_LAYERS, --num_mlp_layers NUM_MLP_LAYERS
                        Number of layers for MLP EXCLUDING the input one
                        (default: 2). 1 means linear model.
  -hd HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        Number of hidden units for HiTIN layer (default: 512)
  -fd FINAL_DROPOUT, --final_dropout FINAL_DROPOUT
                        Dropout rate for HiTIN layer (default: 0.5)
  -tp {root,sum,avg,max}, --tree_pooling_type {root,sum,avg,max}
                        Pool strategy for the whole tree in Eq.11. Could be
                        chosen from {root, sum, avg, max}.
  -hp HIERAR_PENALTY, --hierar_penalty HIERAR_PENALTY
                        The weight for L^R in Eq.14 (default: 1e-6).
  -ct CLASSIFICATION_THRESHOLD, --classification_threshold CLASSIFICATION_THRESHOLD
                        Threshold of binary classification. (default: 0.5)
  --log_dir LOG_DIR     Path to save log files (default: log).
  --ckpt_dir CKPT_DIR   Path to save checkpoints (default: ckpt).
  --begin_time BEGIN_TIME
                        The beginning time of a run, which prefixes the name
                        of log files.
```

We provide a lot of config files in `./config`. Before running, the last thing to do is modify the `YOUR_DATA_DIR`, `YOUR_BERT_DIR` in the json file.

An example of training HiTIN on RCV1 with **TextRCNN** as the text encoder:
```shell
python train.py -cfg config/tin-rcv1-v2.json -k 2 -b 64 -hd 512 -lr 1e-4 -tp sum
```

An example of training HiTIN on WOS with **BERT** as the text encoder:
```shell
python train.py -cfg config/tin-wos-bert.json -k 2 -b 12 -hd 768 -lr 1e-4 -tp sum
```

## Citation
If you found the provided code with our paper useful in your work, please **star** this repo and cite our paper!
```
@inproceedings{Zhu2023HiTINHT,
  title={HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification},
  author={He Zhu and Chong Zhang and Junjie Huang and Junran Wu and Ke Xu},
  year={2023}
}
```