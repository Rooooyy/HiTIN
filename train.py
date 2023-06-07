#!/usr/bin/env python
# coding:utf-8
import helper.logger as logger
from models.model import HiAGM
import torch
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss
from train_modules. trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
from helper.arg_parser import get_args

import time
import random
import numpy as np
import pprint
import warnings

from transformers import BertTokenizer
from helper.lr_schedulers import get_linear_schedule_with_warmup
from helper.adamw import AdamW

warnings.filterwarnings("ignore")


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.learning_rate,  # using args
            # lr=config.train.optimizer.learning_rate,
                                params=params,
                                weight_decay=args.l2rate)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config, args):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=50000)
    if config.text_encoder.type == "bert":
        tokenizer = BertTokenizer.from_pretrained(config.text_encoder.bert_model_dir)
    else:
        tokenizer = None

    # get data
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab, tokenizer=tokenizer)

    # build up model
    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')

    hiagm.to(config.train.device_setting.device)

    # Code for counting parameters
    # from thop import clever_format
    # print(hiagm)
    # def count_parameters(model):
    #     total = sum(p.numel() for p in model.parameters())
    #     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return total, trainable
    #
    # total_params, trainable_params = count_parameters(hiagm)
    # total_params, trainable_params = clever_format([total_params, trainable_params], "%.4f")
    # print("Total num of parameters: {}. Trainable parameters: {}".format(total_params, trainable_params))
    # sys.exit()

    # define training objective & optimizer
    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                   corpus_vocab.v2i['label'],
                                   # recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_penalty=args.hierar_penalty,  # using args
                                   recursive_constraint=config.train.loss.recursive_regularization.flag)
    if config.text_encoder.type == "bert":
        t_total = int(len(train_loader) * (config.train.end_epoch-config.train.start_epoch))

        param = list(hiagm.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in param if 'bert' in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.l2_rate, 'lr': config.train.optimizer.learning_rate},
            {'params': [p for n, p in param if 'bert' in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.train.optimizer.learning_rate},
            {'params': [p for n, p in param if 'bert' not in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.l2_rate, 'lr': args.learning_rate},
            {'params': [p for n, p in param if 'bert' not in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.learning_rate}
        ]
        warmup_steps = int(t_total * 0.1)
        optimizer = AdamW(grouped_parameters, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
    else:
        optimizer = set_optimizer(config, hiagm)
        scheduler = None

    # get epoch trainer
    trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      vocab=corpus_vocab,
                      config=config)

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    '''
        ckpt_dir
            begin-time_dataset_model
                best_micro/macro-model_type-training_params_(tin_params)
                                            
    '''
    # model_checkpoint = config.train.checkpoint.dir
    model_checkpoint = os.path.join(args.ckpt_dir, args.begin_time + config.train.checkpoint.dir)  # using args
    model_name = config.model.type
    if config.structure_encoder.type == "TIN":
        model_name += '_' + str(args.tree_depth) + '_' + str(args.hidden_dim) + '_' + args.tree_pooling_type + '_' + str(args.final_dropout) + '_' + str(args.hierar_penalty)
    wait = 0
    if not os.path.isdir(model_checkpoint):
        # os.mkdir(model_checkpoint)
        os.makedirs(model_checkpoint)
    elif args.load_pretrained:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = ''
        for model_file in dir_list[::-1]:  # best or latest ckpt
            if model_file.startswith('best'):
                continue
            else:
                latest_model_file = model_file
                break
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                       model=hiagm,
                                                       config=config,
                                                       optimizer=optimizer)
            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance[0], best_performance[1]))

    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        # epochs += 1
        start_time = time.time()
        trainer.train(train_loader, epoch)
        trainer.eval(train_loader, epoch, 'TRAIN')
        performance = trainer.eval(dev_loader, epoch, 'DEV')

        # record results for each epoch
        print("[Val] epoch: %d precision: %.4f\t recall: %.4f\t micro_f1: %.4f\t macro_f1: %.4f" \
                    % (epoch, performance['precision'], performance['recall'], performance['micro_f1'], performance['macro_f1']))
        # saving best model and check model
        if not (performance['micro_f1'] >= best_performance[0] or performance['macro_f1'] >= best_performance[1]):
            wait += 1
            # reduce LR on plateau
            if wait % config.train.optimizer.lr_patience == 0:
                logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
                trainer.update_lr()
            # early stopping
            if wait == config.train.optimizer.early_stopping:
                logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                               .format(wait))
                break

        if performance['micro_f1'] > best_performance[0]:
            wait = 0
            logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro_f1']))
            best_performance[0] = performance['micro_f1']
            best_epoch[0] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
        if performance['macro_f1'] > best_performance[1]:
            wait = 0
            logger.info('Improve Macro-F1 {}% --> {}%'.format(best_performance[1], performance['macro_f1']))
            best_performance[1] = performance['macro_f1']
            best_epoch[1] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

        # if epoch % 10 == 1:
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'model_type': config.model.type,
        #         'state_dict': hiagm.state_dict(),
        #         'best_performance': best_performance,
        #         'optimizer': optimize.state_dict()
        #     }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))
        # total_time += time.time() - start_time

    # print("Average training time per epoch: {} secs.".format(float(total_time) / epochs))

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimizer)
        # start_time = time.time()
        performance = trainer.eval(test_loader, best_epoch[0], 'TEST')
        # print("Inference time A : {} secs.".format(time.time() - start_time))
        # record best micro test performance
        print("Best micro-f1 on epoch: %d, [Test] performance↓\nmicro-f1: %.4f\nmacro-f1: %.4f" \
                    % (best_epoch[0], performance['micro_f1'], performance['macro_f1']))

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimizer)
        performance = trainer.eval(test_loader, best_epoch[1], 'TEST')
        # record best macro test performance
        print("Best macro-f1 on epoch: %d, [Test] performance↓\nmicro-f1: %.4f\nmacro-f1: %.4f" \
                    % (best_epoch[1], performance['micro_f1'], performance['macro_f1']))
    return


if __name__ == "__main__":
    args = get_args()
    pprint.pprint(vars(args))
    configs = Configure(config_json_file=args.config_file)
    configs.update(vars(args))

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')

    logger.Logger(configs)

    # if not os.path.isdir(configs.train.checkpoint.dir):
    #     os.mkdir(configs.train.checkpoint.dir)

    # train(config)
    train(configs, args)
