import os
import sys
import time
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')
from logging import Formatter, StreamHandler, FileHandler, getLogger

import cv2
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from configuration import CFG


def read_img_from_path(path):
  im_bgr = cv2.imread(path)
  im_rgb = im_bgr[:, :, ::-1].copy()
  return im_rgb

def accuracy_metric(input, target):
  return accuracy_score(target.cpu(), input.cpu())

def seed_everything(seed=CFG.seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def read_img_from_path(path):
  im_bgr = cv2.imread(path)
  im_rgb = im_bgr[:, :, ::-1].copy()
  return im_rgb

def init_logger(log_file=CFG.log_path+'train.log'):
  logger = getLogger(__name__)
  logger.setLevel("INFO")
  log_fmt = Formatter(
        "%(asctime)s - [%(levelname)s][%(funcName)s] - %(message)s "
      ) 
  # log_fmt = Formatter("%(message)s") 

  stream_handler = StreamHandler()
  # stream_handler.setLevel("INFO")
  stream_handler.setFormatter(log_fmt)

  file_handler = FileHandler(filename=log_file)
  # file_handler.setLevel("INFO")
  file_handler.setFormatter(log_fmt)

  logger.addHandler(stream_handler)
  logger.addHandler(file_handler)
  logger.propagate = False
  
  return logger

LOGGER = init_logger()

def get_folds():
  train_df_merged = pd.read_csv(CFG.train_data_path+'merged.csv')
  train_df_2020 = train_df_merged.loc[train_df_merged.source == 2020]
  train_labels_2020 = train_df_2020['label']
  num_data_2020 = len(train_df_2020)

  # Randomly select 100 samples if debug mode is on
  if CFG.debug == True:
    sample_id = np.random.randint(num_data_2020, size=100)
    num_data_2020 = len(sample_id)
    
    train_df_2020 = train_df_2020.iloc[sample_id]
    train_labels_2020 = train_labels_2020.iloc[sample_id]
  
  skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
  skf.get_n_splits(np.arange(num_data_2020), train_labels_2020) # Note: you have to take in train labels because it is StratifiedKFold
  folds = [(idx_train, idx_valid) for i, (idx_train, idx_valid) in \
           enumerate(skf.split(np.arange(num_data_2020), train_labels_2020))]

  # If we just want to debug, no need to use 2019 data
  if CFG.use_2019==True:
    # Only add data from 2019 to the training folds to avoid contamination of the validaiton fold
    train_df_2019 = train_df_merged.loc[train_df_merged.source == 2019]
    train_labels_2019 = train_df_2019['label']
    num_data_2019 = len(train_df_2019)

    folds_2019 = [np.concatenate((idx_train, idx_valid)) for i, (idx_train,idx_valid) in \
                  enumerate(skf.split(np.arange(num_data_2019), train_labels_2019))]
    
    # Merge 2019 data into the training folds
    for i in range(CFG.n_folds):
      (idx_train, idx_valid) = folds[i]
      folds[i] = (np.concatenate((idx_train, train_df_2019.iloc[folds_2019[i]].index)), idx_valid)

  return folds