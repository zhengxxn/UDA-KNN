# UDA-KNN

Code for our paper "Non-Parametric Unsupervised Domain Adaptation for Neural Machine Translation".


Please cite our paper if you find this repo helpful in your research:

```
comming soon
```


The implementation is build upon [fairseq](https://github.com/pytorch/fairseq), and heavily inspired by [knn-lm](https://github.com/urvashik/knnlm), many thanks to the authors for making their code avaliable.

Note: This code is a little messy now（But of course it works well), we will further refine it as soon as possible.

## Requirements and Installation

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0

You can install this project by
```
pip install --editable ./
```

## Instructions

We use an example to show how to use our code.

### Pre-trained Model and Data

The pre-trained translation model can be downloaded from [this site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md).
We use the De->En Single Model as general domain model.

The raw WMT19 News data for training our introduced adapters can be downloaded in [here](http://www.statmt.org/wmt19/translation-task.html),
while the raw multi-domain data can be downloaded in [here](https://github.com/roeeaharoni/unsupervised-domain-clusters). You should preprocess all data with moses toolkits and the bpe-codes provided by pre-trained model. 

For convenience, we also released the fairseq-preprocessed data, including the [wmt19 dataset](https://drive.google.com/file/d/1BlCc1Aw_q53gRinPA0eca8-WJA7_AY-p/view?usp=sharing), 
and the [multi-domain dataset](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view?usp=sharing)

### Train Adapters

we insert the adapters to pre-trained model and train that with below script:

```
PRETRAINED_MODEL_PATH=/path/to/pre-trained-de-en-model
DATA_PATH=/path/to/wmt19-data-bin
PROJECT_PATH=/path/to/uda-knn
MODEL_RECORD_PATH=/path/to/save/trained-model
TRAINING_RECORD_PATH=/path/to/save/training-record

mkdir -p $MODEL_RECORD_PATH
mkdir -p $TRAINING_RECORD_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 \
$PROJECT_PATH/fairseq_cli/train.py \
$DATA_PATH \
--no-progress-bar --log-interval 500 --log-format simple \
--arch transformer_wmt19_de_en --share-all-embeddings --encoder-append-adapter --encoder-embedding-append-adapter --only-update-adapter --adapter-ffn-dim 1024 \
--tensorboard-logdir $TRAINING_RECORD_PATH --save-dir $MODEL_RECORD_PATH \
--validate-interval 1 --validate-after-updates 10000 --validate-interval-updates 5000 --save-interval-updates 5000 --keep-interval-updates 1 \
--keep-best-checkpoints 1 --save-interval 1 --keep-last-epochs 1 --no-save-optimizer-state \
--train-subset train --valid-subset valid --source-lang de --target-lang en \
--max-tokens 10000 --update-freq 1 --max-epoch 200 \
--optimizer adam --adam-betas "(0.9, 0.98)" --min-lr 1e-09 --lr 0.0007 \
--warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --clip-norm 0.0 --weight-decay 0.0 \
--criterion label_smoothed_cross_entropy_with_denoising_approximate \
--denoising-approximate-loss-type mse --denoising-start-epoch 1 --denoising-approximate-loss-ratio 0.01 --denoising-approximate-loss-ratio-begin 0.0001 \
--label-smoothing 0.1 --update-denoising-with-adapter \
--task translation_and_denoising --denoising-mask-length span-poisson --denoising-replace-length 1 --denoising-mask-ratio 0.0 \
--train-task denoising_approximate --validation-task denoising_approximate --select-last-as-mask \
--fp16 \
--reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
--restore-file $PRETRAINED_MODEL_PATH
```

We also provide the well-trained [model](https://drive.google.com/file/d/1682Zzm9_WEWAI_p9j0-wli3t385OD5Ak/view?usp=sharing).

### Create Datastore

When the model (with adapters) is trained, you could load the model and use it to create datastore with below script (Please make sure that --activate-adapter):

```
DSTORE_SIZE=/token count of target-side data, you can find it in preprocess.log in the data binary folder 
MODEL_PATH=/path/to/model
DATA_PATH=/data binary path, which can be the copied pairs or parallel pairs (we automatically use the target-side)
DATASTORE_PATH=/path/to/save/datastore
PROJECT_PATH=/path/to/uda-knn

mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation_and_denoising --denoising-mask-length span-poisson --denoising-replace-length 1 --denoising-mask-ratio 0.0 \
    --valid-subset train --save-denoising-feature \
    --path $MODEL_PATH --activate-adapter \
    --max-tokens 8000 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH
```

### Inference

This part can refer to [this site](https://github.com/zhengxxn/adaptive-knn-mt), 
But please note that we fix k as 16 and temperature as 4 for IT, Medical, Law and as 40 for Koran respectively.

### Much more

The code for Back-Translation and analysis in our paper is also included in this repo, 
we will re-organize them as soon as possible.
