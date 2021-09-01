# Multi-turn Response Selection

Paper: Multi-turn Response Selection using Dialogue Dependency Relations has been accepted by EMNLP 2020 ([link](https://www.aclweb.org/anthology/2020.emnlp-main.150/)).

## Response selection

### Download the dataset

Download the discourse parsed DSTC7 datasets from [Google Drive](https://drive.google.com/file/d/1nDZsFtkuGJIIqv3-7vw21PBjSPmqoMTp/view?usp=sharing) and unzip into ~/code/data. 

The other other two dataset used in this paper can be downloaded here: 
* [UbuntuV2](https://adapt.seiee.sjtu.edu.cn/~angel/par_ubuntuv2.zip) or from [Google Drive](https://drive.google.com/file/d/1FNKt-CagID3tIVpcYb0B4HfCSTRkIzbi/view?usp=sharing)
* [DSTC8\*](https://adapt.seiee.sjtu.edu.cn/~angel/par_dstc8.zip) or from [Google Drive](https://drive.google.com/file/d/1WafMff_tumrIYPe6gMTRjrqngkXB6WPO/view?usp=sharing)

### Getting Started

The code for this paper is based on [ParlAI](http://parl.ai). The original codes this repository based on can be downloaded [here](https://drive.google.com/file/d/1xhKv-XY2aXKDcVoA_d3r0ES7bkETwhQk/view?usp=sharing). 

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

The comparison of baselines and our models on DSTC7 are shown as follows:
|                                 | hits@1 | hits@10 | hits@50 | MRR  |
|---------------------------------|--------|---------|---------|------|
| DAM (Zhou et al., 2018)         | 34.7   | 66.3    | -       | 35.6 |
| ESIM-18 (Dong and Huang, 2018)  | 50.1   | 78.3    | 95.4    | 59.3 |
| ESIM-19 (Chen and Wang, 2019)   | 64.5   | 90.2    | **99.4**    | 73.5 |
| Bi-Enc (Humeau et al., 2019)    | 70.9   | 90.6    | -       | 78.1 |
| Poly-Enc (Humeau et al., 2019)  | 70.9   | 91.5    | -       | 78.0 |
| Cross-Enc (Humeau et al., 2019) | 71.7   | 92.4    | -       | 79.0 |
| Thread-bi                       | **73.3**   | 92.5    | 99.3    | 80.2 |
| Thread-poly                     | 73.2   | **93.6**    | 99.1    | **80.4** |



## Dependency Parsing

The new dataset we used to train the [dependency parsing model](https://github.com/shizhouxing/DialogueDiscourseParsing) is transformed from the dataset proposed in "[A large-scale corpus for conversation disentanglement](https://www.aclweb.org/anthology/P19-1374/)". 

The new dataset can be downloaded [here](https://drive.google.com/file/d/1Bh515NoYHq1od7B0q9cax99jTCLDLhAq/view?usp=sharing). It includes:
* new_ubuntu_train.json: Transformed from the original training set, and also used as the training set in our paper.
* new_ubuntu_dev.json: Transformed from the original development set.
* new_ubuntu_test.json: Transformed from the orignial test set.
* new_ubuntu_final_test.json: The merge of new_ubuntu_dev.json and new_ubuntu_test.json. Used as the test set in our paper.


It should be noted that we only use this model to predict if there exists a dependency relation between two utterances and ignored the relation types. The "type" of each relation in our generated dataset is meaningless.
