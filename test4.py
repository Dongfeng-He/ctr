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
    new_result_list = []
    with open("result_list.pkl", "rb") as f:
        result_list = pickle.load(f)
        for qingting_id, group_id, probs in result_list:
            group_id = group_id.to('cpu').detach().numpy().tolist()
            probs = np.array(probs).squeeze().tolist()
            new_result_list.append([qingting_id, group_id, probs])
    with open("new_result_list.pkl", "wb") as f:
        pickle.dump(new_result_list, f)