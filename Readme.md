# Multi-turn Response Selection

Paper: Multi-turn Response Selection using Dialogue Dependency Relations has been accepted by EMNLP 2020 ([linke](https://www.aclweb.org/anthology/2020.emnlp-main.150/)).

## Response selection

### Download the dataset

Download the discourse parsed datasets from [Google Drive](https://drive.google.com/file/d/1nDZsFtkuGJIIqv3-7vw21PBjSPmqoMTp/view?usp=sharing) and unzip into ~/code/data. 

### Getting Started

The code for this paper is based on [ParlAI](http://parl.ai). 

To build the environment with anaconda:
```bash
conda create -n Thread-Enc python=3.7 pytorch=1.1
conda activate Thread-Enc
pip install -r requirements.txt
```

Run the following commands to setup the code:
```bash
cd ~/code; 
python setup.py develop
```

Run this command to finetune Thread-bi encoder (e.g. DSTC7):

```bash
python3 -u examples/train_model.py --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model --shuffle True --eval-batchsize 4 --batchsize 32  --model transformer/parencoder --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 --text-truncate 360 -vp 3 -veps 0.5 --validation-metric accuracy --validation-metric-mode max --save-after-valid True --log_every_n_secs 20 --candidates batch --dict-tokenizer bpe  --dict-lower True --optimizer adamax --output-scaling 0.06 --variant xlm --reduction_type mean --share-encoders False --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 --learn-embeddings True --share-word-embeddings False --dict-endtoken __start__ -pyt par_dstc7 --fp16 False --par_type basic --par_num 4 --reduction-type mean --parencoder-type codes --model-file ./thread_bi_dstc7
```
Run this command to finetune Thread-poly encoder（e.g. DSTC7):
```bash
python3 -u examples/train_model.py --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model -pyt par_dstc7 --eval-batchsize 4 --batchsize 32  --model transformer/parpolyencoder --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 --text-truncate 360 -vp 3 -veps 0.5 --validation-metric accuracy --validation-metric-mode max --save-after-valid True --log_every_n_secs 20 --candidates batch --dict-tokenizer bpe  --dict-lower True --optimizer adamax --output-scaling 0.06 --variant xlm --reduction_type mean --share-encoders False --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 --learn-embeddings True --share-word-embeddings False --dict-endtoken __start__ --fp16 False --polyencoder-type codes --codes-attention-type basic --poly-n-codes 64 --poly-attention-type basic --polyencoder-attention-keys context --par_type basic --par_num 4 --reduction-type mean --parencoder-type codes --model-file ./thread_poly_dstc7
```

## Dependency Parsing

The new dataset we used to train the [dependency parsing model](https://github.com/shizhouxing/DialogueDiscourseParsing) is transformed from the dataset proposed in "[A large-scale corpus for conversation disentanglement](https://www.aclweb.org/anthology/P19-1374/)". The new dataset can be downloaded from [here](https://drive.google.com/file/d/1Bh515NoYHq1od7B0q9cax99jTCLDLhAq/view?usp=sharing). 

It should be noted that we only use this model to predict if there exists a dependency relation between two utterances and ignored the relation types. The "type" of each relation in our generated dataset is meaningless.