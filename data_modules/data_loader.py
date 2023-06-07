#!/usr/bin/env python
# coding:utf-8

from data_modules.dataset import ClassificationDataset
from data_modules.collator import Collator
from torch.utils.data import DataLoader

import torch
import numpy as np
import random

def data_loaders(config, vocab, data={'train': None, 'val': None, 'test': None}, tokenizer=None):
    """
    get data loaders for training and evaluation
    :param config: helper.configure, Configure Object
    :param vocab: data_modules.vocab, Vocab Object
    :param data: on-memory data, Dict{'train': List[str] or None, ...}
    :param tokenizer: bert tokenizer for tokenizing input document
    :return: -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    on_memory = data['train'] is not None
    collate_fn = Collator(config, vocab)
    train_dataset = ClassificationDataset(config, vocab, stage='TRAIN', on_memory=on_memory, corpus_lines=data['train'], tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,  # using args
                              shuffle=True,
                              num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=False,
                              drop_last=True)

    val_dataset = ClassificationDataset(config, vocab, stage='VAL', on_memory=on_memory, corpus_lines=data['val'], tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,  # using args
                            shuffle=True,
                            num_workers=config.train.device_setting.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=False,
                            drop_last=True)

    test_dataset = ClassificationDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=data['test'], tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,  # using args
                             shuffle=True,
                             num_workers=config.train.device_setting.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=False,
                             drop_last=True)

    return train_loader, val_loader, test_loader
