# Multimodal Neural Machine Translation


### Dataset

 - [Multi30k Dataset](https://github.com/multi30k/dataset)













This is the implementation of **four different multi-modal neural machine translation models** described in the research papers [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf).
They are based on the [Pytorch](https://github.com/pytorch/pytorch) port of [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source (MIT) neural machine translation system.


Table of Contents
=================
  * [Requirements](#requirements)
  * [Features](#features)
  * [Multi-modal NMT Quickstart](#quickstart)
  * [Citation](#citation)

[Full Documentation](http://opennmt.net/OpenNMT-py/)
 
## Requirements

```
torchtext>=0.2.1
pytorch>=0.2
```

In case one of the two are missing or not up-to-date and assuming you installed pytorch using the conda package manager and torchtext using pip, you might want to run the following:

```bash
conda install -c soumith pytorch
pip install torchtext --upgrade
pip install -r requirements.txt
pip install pretrainedmodels
conda update pytorch
```

## Features

The following OpenNMT features are implemented:

- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard/Crayon logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)

Beta Features (committed):
- multi-GPU
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper
- Inference time loss functions.

## Multi-modal NMT Quickstart

### Step 0: Extract the image features for the Multi30k data set.

If you are using image features extracted by someone else, you can skip this step.

We assume you have downloaded the [Multi30k data set](http://www.statmt.org/wmt16/multimodal-task.html) and have the training, validation and test images locally (make sure you download the `test2016` test set). Together with the image files, you need text files with the image file names in the training, validation, and test sets, respectively. These are named `train_images.txt`,`val_images.txt`, and `test_images.txt`, and are part of the original Flickr30k data set. If you download them from the [WMT Multi-modal MT shared task website](http://www.statmt.org/wmt16/multimodal-task.html), you might need to adjust the file names accordingly.

In order to extract the image features, run the following script:

```bash
python extract_image_features.py --gpuid 0 --pretrained_cnn vgg19_bn --splits=train,valid,test --images_path ./path/to/flickr30k/images/ --train_fnames ./path/to/flickr30k/train_images.txt --valid_fnames ./path/to/flickr30k/val_images.txt --test_fnames ./path/to/flickr30k/test2016_images.txt
```

This will use GPU 0 to extract features with the pre-trained VGG19 with batch normalisation, for the training, validation and test sets of the Flickr30k. Change the name of the pre-trained CNN to any of the CNNs available under [this repository](https://github.com/Cadene/pretrained-models.pytorch), and the model will automatically use this CNN to extract features. **This script will extract both global and local visual features**.


### Step 1: Preprocess the data

That is the same way as you would do with a text-only NMT model. **Important**: *the preprocessing script only uses the textual portion of the multi-modal machine translation data set*!

In here, we assume you have downloaded the [Multi30k data set](http://www.statmt.org/wmt16/multimodal-task.html) and extracted the sentences in its training, validation and test sets. After pre-processing them (e.g. tokenising, lowercasing, and applying a [BPE model](https://github.com/rsennrich/subword-nmt)), feed the training and validation sets to the `preprocess.py` script, as below.

```bash
python preprocess.py -train_src ./path/to/flickr30k/train.norm.tok.lc.10000bpe.en -train_tgt./path/to/flickr30k/train.norm.tok.lc.10000bpe.de -valid_src ./path/to/flickr30k/val.norm.tok.lc.10000bpe.en -valid_tgt ./path/to/flickr30k/val.norm.tok.lc.10000bpe.de -save_data ./data/m30k
```


### Step 2: Train the model

To train a multi-modal NMT model, use the `train_mm.py` script. In addition to the parameters accepted by the standard `train.py` (that trains a text-only NMT model), this script expects the path to the training and validation image features, as well as the multi-modal model type (one of `imgd`, `imge`, `imgw`, or `src+img`).

For a complete description of the different multi-modal NMT model types, please refer to the papers where they are described [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf).

```bash
python train_mm.py -data data/m30k -save_model model_snapshots/IMGD_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/features/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/features/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgd
```

In case you want to continue training from a previous checkpoint, simply run (for example):

```bash
MODEL_SNAPSHOT=IMGD_ADAM_acc_60.79_ppl_8.38_e4.pt
python train_mm.py -data data/m30k -save_model model_snapshots/IMGD_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/features/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/features/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --multimodal_model_type imgd -train_from model_snapshots/${MODEL_SNAPSHOT}
```

As an example, if you wish to train a doubly-attentive NMT model (referred to as `src+img`), try the following command:

```bash
python train_mm.py -data data/m30k -save_model model_snapshots/NMT-src-img_ADAM -gpuid 0 -epochs 25 -batch_size 40 -path_to_train_img_feats /path/to/flickr30k/features/flickr30k_train_vgg19_bn_cnn_features.hdf5 -path_to_valid_img_feats /path/to/flickr30k/features/flickr30k_valid_vgg19_bn_cnn_features.hdf5 -optim adam -learning_rate 0.002 -use_nonlinear_projection --decoder_type doubly-attentive-rnn --multimodal_model_type src+img
```


### Step 3: Translate new sentences

To translate a new test set, simply use `translate_mm.py` similarly as you would use the original `translate.py` script, with the addition of the path to the file containing the test image features. In the example below, we translate the Multi30k test set used in the 2016 run of the WMT Multi-modal MT Shared Task.

```bash
MODEL_SNAPSHOT=IMGD_ADAM_acc_60.79_ppl_8.38_e4.pt
python translate_mm.py -src ~/exp/opennmt_imgd/data_multi30k/test2016.norm.tok.lc.bpe10000.en -model model_snapshots/${MODEL_SNAPSHOT} -path_to_test_img_feats ~/resources/multi30k/features/flickr30k_test_vgg19_bn_cnn_features.hdf5 -output model_snapshots/${MODEL_SNAPSHOT}.translations-test2016
```

## Citation

If you use the multi-modal NMT models in this repository, please consider citing the research papers where they are described [(1)](http://aclweb.org/anthology/D17-1105) and [(2)](https://aclweb.org/anthology/P/P17/P17-1175.pdf):

```
@InProceedings{CalixtoLiu2017EMNLP,
  Title                    = {{Incorporating Global Visual Features into Attention-Based Neural Machine Translation}},
  Author                   = {Iacer Calixto and Qun Liu},
  Booktitle                = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  Year                     = {2017},
  Address                  = {Copenhagen, Denmark},
  Url                      = {http://aclweb.org/anthology/D17-1105}
}
```

```
@InProceedings{CalixtoLiuCampbell2017ACL,
  author    = {Calixto, Iacer  and  Liu, Qun  and  Campbell, Nick},
  title     = {{Doubly-Attentive Decoder for Multi-modal Neural Machine Translation}},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1913--1924},
  url       = {http://aclweb.org/anthology/P17-1175}
}
```

If you use OpenNMT, please cite as below.

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
