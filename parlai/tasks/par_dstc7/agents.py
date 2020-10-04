#!/usr/bin/env python3
"""
The Ubuntu dataset used for dstc 8.
"""

from parlai.core.teachers import FixedDialogTeacher
from .build import build

import json
import os
import random
import numpy


def get_partioned_text(texts, partitions, segment_type='basic',segment_num=5):
    '''

    :param texts: list of utterances
    :param patitions: idx of partitions by our algorithm
    :param segment_type: 'basic',
                        'without',
                        'random',
                        'distance'
    :return: text: num_partitions partition_mask
    '''
    text = []
    partition_mask = []
    if segment_type=='without':
        text = [' '.join(texts)]
        partition_mask = [1]

        text = text + ["None"]*(segment_num-1)
        partition_mask = partition_mask + [0]* (segment_num-1)
    elif segment_type=='random':
        total_utterance = len(texts)
        num_utterance_per_partition = int(total_utterance/segment_num)
        if num_utterance_per_partition == 0:
            num_utterance_per_partition=1

        for i in range(segment_num):
            idxs = numpy.sort(numpy.random.choice(total_utterance,size=num_utterance_per_partition,replace=False))
            temp=[]
            for idx in idxs:
                temp.append(texts[idx])
            text.append(' '.join(temp))
        partition_mask = [1] * segment_num
    elif segment_type=='distance':
        total_utterance = len(texts)

        num_utterance_per_partition = int(total_utterance/segment_num)
        if num_utterance_per_partition ==0:
            num_utterance_per_partition=1

        for i in range(segment_num):
            temp=''
            if num_utterance_per_partition * (i + 1) <= total_utterance:
                if i==0:
                    temp = ' '.join(texts[(-num_utterance_per_partition * (i + 1)):])
                else:
                    temp = ' '.join(texts[(-num_utterance_per_partition*(i+1)):(-num_utterance_per_partition*i)])
                text.append(temp)
                partition_mask.append(1)
            else:
                text.append('None')
                partition_mask.append(0)
    elif segment_type=='basic':
        num_partitions = len(partitions)
        for i in range(segment_num):
            if i < num_partitions:
                idxs = partitions[i]
                temp = []
                for idx in idxs:
                    temp.append(texts[idx])
                text.append(' '.join(temp))
                partition_mask.append(1)
            else:
                text.append('None')
                partition_mask.append(0)


    return text, partition_mask


class ParDSTC7Teacher(FixedDialogTeacher):
    """
    Teacher that corresponds to the default DSTC7 ubuntu track 1.

    The data hasn't been augmented by using the multi-turn utterances.
    """


    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.segment_type = opt['par_type']
        self.segment_num = opt['par_num']

        self.split = 'train'
        if 'valid' in opt['datatype']:
            self.split = 'dev'
        if 'test' in opt['datatype']:
            self.split = 'test'
        build(opt)

        basedir = os.path.join(opt['datapath'], 'par_dstc7')
        filepath = os.path.join(
            basedir, "ubuntuDSTC7st1.%s.withdiscourse.txt" % (self.split)
        )

        with open(filepath, 'r') as f:
            samples = []
            for n,line in enumerate(f):
                samples.append(json.loads(line))
            self.data = samples

        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Par_dstc7')
        # the rules of partition
        agent.add_argument(
            '--par_type',
            default='basic',
            type=str,
            choices=['basic', 'without', 'random', 'distance'],
            help="the ways we devide the partitions"
                 "basic: the rules we use reply_to predictions"
                 "without: don't divide into partitions"
                 "random: ramdomly divide into several(decide by the length of the dialogue) partitions"
                 "distance: divide(truncate) into partitions only by distance",
        )
        agent.add_argument(
            '--par_num',
            default=5,
            type=int,
            help="The number of partitions we consider"
                 "this parameter is activate when --par_type != without",
        )

    def get_suffix(self):
        return ""

    def _setup_data(self, datatype):
        pass

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)



    def get(self, episode_idx, entry_idx=0):
        rand = random.Random(episode_idx)
        episode = self.data[episode_idx]
        texts=[]
        for m in episode['edus']:
            texts.append("participant "+ str(m['speaker']) + ": "+m['text'])

        partitions = episode['partitions']

        text, partition_mask = get_partioned_text(texts,partitions,self.segment_type,self.segment_num)

        labels = [m["utterance"] for m in episode["options-for-correct-answers"]]

        labels_ids = [m['candidate-id'] for m in episode['options-for-correct-answers']]


        candidates = [m['utterance'] for m in episode['options-for-next']]
        candidate_ids = [m['candidate-id'] for m in episode["options-for-next"]]

        for i, m in enumerate(labels_ids):
            if m in candidate_ids:
                pass
            else:
                candidates.append(labels[i])
                candidate_ids.append(labels_ids[i])


        rand.shuffle(candidates)
        label_key = "labels" if self.split == "train" else "eval_labels"



        return {
            "text": text,
            label_key: labels,
            "label_candidates": candidates,
            "episode_done": True,
            "partition_mask":partition_mask

        }

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared



class ParDSTC7TeacherAugmentedSampled(ParDSTC7Teacher):
    """
    The dev and test set are the same, but the training set has been augmented using the
    other utterances.

    Moreover, only 16 candidates are used (including the right one)
    """

    def get_suffix(self):
        if self.split != "train":
            return ""
        return "_augmented"

    def get_nb_cands(self):
        return 16

    def get(self, episode_idx, entry_idx=0):
        sample = super().get(episode_idx, entry_idx)
        if self.split != 'train':
            return sample
        new_cands = [sample['labels'][0]]
        counter = 0
        while len(new_cands) < self.get_nb_cands():
            if sample['label_candidates'][counter] not in sample['labels']:
                new_cands.append(sample['label_candidates'][counter])
            counter += 1
        sample['label_candidates'] = new_cands
        return sample


class DefaultTeacher(ParDSTC7Teacher):
    pass

if __name__ == '__main__':
    texts=["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]

    partitions = [[11,12,13],[9,10],[5,6,7,9]]

    text,partition_mask = get_partioned_text(texts, partitions, segment_type='distance',segment_num=20)
    print(text,partition_mask)