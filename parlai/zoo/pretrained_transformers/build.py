#!/usr/bin/env python3

"""
Pretrained model allowing to get the same performances as in
https://arxiv.org/abs/1905.01969.
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    model_name = 'pretrained_transformers'
    mdir = os.path.join(get_model_dir(datapath), model_name)
    version = 'v3.0'
    if not built(mdir, version):
        opt = {'datapath': datapath}
        fnames = ['pretrained_transformers.tgz']
        download_models(opt, fnames, model_name, version=version, use_model_type=False)

if __name__ == '__main__':
    download("zoo:pretrained_transformers/bi_model_huge_reddit/model")