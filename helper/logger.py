#!/usr/bin/env python
# coding:utf-8
import os
import sys
import logging

logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}


def debug(msg):
    logging.debug(msg)
    # print('DEBUG: ', msg)


def info(msg):
    logging.info(msg)
    # print('INFO: ', msg)


def warning(msg):
    logging.warning(msg)
    # print('WARNING: ', msg)


def error(msg):
    logging.error(msg)
    # print('ERROR: ', msg)


def fatal(msg):
    logging.critical(msg)
    # print('FATAL: ', msg)


class Logger(object):
    def __init__(self, config):
        """
        set the logging module
        :param config: helper.configure, Configure object
        """
        super(Logger, self).__init__()
        assert config.log.level in logging_level.keys()

        self.logger = logging.getLogger('').handlers = []

        self.log_dir = os.path.join(config.log_dir,
                               config.model.type + '-' + config.structure_encoder.type + '-' + config.text_encoder.type,
                               config.data.dataset + '_' + str(config.batch_size) + '_' + str(
                                   config.learning_rate) + '_' + str(config.l2rate) + '_' + str(
                                   config.classification_threshold) + '_' + str(config.hierar_penalty))
        if config.structure_encoder.type == "TIN":
            self.log_file = os.path.join(self.log_dir, config.begin_time + str(config.tree_depth) + '_' + str(config.hidden_dim) + '_' + config.tree_pooling_type + '_' + str(config.final_dropout) + '.log')
        else:
            self.log_file = os.path.join(config.log_dir, config.begin_time + config.log.filename)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(filename=self.log_file,
                            level=logging_level[config.log.level],
                            format='%(asctime)s - %(levelname)s : %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')

        self.stdout_handler = logging.StreamHandler(sys.stdout)
        self.stdout_handler.setLevel(logging.ERROR)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s : %(message)s'))

        self.logger = logging.getLogger('')
        self.logger.addHandler(self.stdout_handler)




