# -*- coding:utf-8 -*-
import datetime
import json
import math
import os
import pickle
import random
from itertools import chain
from time import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils import data
from tqdm import tqdm

from seq.seq import TCNModel, LSTMModel, AVGModel, AttentionModel


if __name__ == "__main__":
    with open("/Volumes/hedongfeng/data/vip/result_list.pkl", "rb") as f:
        result_list = pickle.load(f)
        print()