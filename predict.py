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


class MyDataset(data.Dataset):
    def __init__(self, tensors_list, batch_size=1, shuffle=True, drop_last=False):
        self.tensors_list = tensors_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = [i for i in range(len(tensors_list))]

    def __getitem__(self, index):
        return self.tensors_list[index]

    def __iter__(self):
        batch = []
        if self.shuffle:
            random.shuffle(self.indices)
        for idx in self.indices:
            batch.append(self.tensors_list[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.tensors_list) // self.batch_size


class DeepFM(torch.nn.Module):
    def __init__(self, field_size, feature_sizes, channel_size=80000, embedding_size=4, is_shallow_dropout=True, dropout_shallow=(0.5, 0.5),
                 deep_layers=(32, 32), is_deep_dropout=True, dropout_deep=(0.5, 0.5, 0.5), deep_layers_activation='relu',
                 is_batch_norm=False, random_seed=950104, use_fm=True, use_ffm=False, use_deep=True, use_cuda=True,
                 use_plain_emb=True, use_seq=True, use_lstm=False, use_tcn=False, use_avg=True, use_att=False,
                 seq_emb_size=64, seq_hidden_size=32, seq_pool="max", loss_func="rank",
                 ):
        super(DeepFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.channel_size = channel_size
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.is_batch_norm = is_batch_norm
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.use_deep = use_deep
        self.use_cuda = use_cuda
        self.use_plain_emb = use_plain_emb
        self.use_seq = use_seq
        self.use_lstm = use_lstm
        self.use_tcn = use_tcn
        self.use_avg = use_avg
        self.use_att = use_att
        self.seq_emb_size = seq_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.seq_pool = seq_pool
        self.sigmoid = nn.Sigmoid()
        self.loss_func = loss_func
        self.criterion = self.rank_loss if loss_func == "rank" else nn.BCELoss()
        torch.manual_seed(self.random_seed)

        """
        check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
        check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm and self.use_deep:
            print("The model is deepfm(fm+deep layers)")
        elif self.use_ffm and self.use_deep:
            print("The model is deepffm(ffm+deep layers)")
        elif self.use_fm:
            print("The model is fm only")
        elif self.use_ffm:
            print("The model is ffm only")
        elif self.use_deep:
            print("The model is deep layers only")
        else:
            print("You have to choose more than one of (fm, ffm, deep) models to use")
            exit(1)

        """
        bias
        """
        if self.use_fm or self.use_ffm:
            self.bias = torch.nn.Parameter(torch.randn(1))

        """
        seq part
        """
        if self.use_seq:
            print("Init seq part")
            # input_size 是所有专辑数量, pool 可选 last、max、avg、both
            if self.use_lstm:
                self.seq_model = LSTMModel(self.channel_size, num_inputs=self.seq_emb_size, hidden_size=self.seq_hidden_size, pool=self.seq_pool)
                self.seq_output_size = self.seq_hidden_size * 4 if self.seq_pool == "both" else self.seq_hidden_size * 2
            elif self.use_tcn:
                self.seq_model = TCNModel(self.channel_size, num_inputs=self.seq_emb_size, num_channels=(self.seq_hidden_size, self.seq_hidden_size, self.seq_hidden_size, self.seq_hidden_size), kernel_size=2, dropout=0.2, pool=self.seq_pool)
                self.seq_output_size = self.seq_hidden_size * 2 if self.seq_pool == "both" else self.seq_hidden_size
            elif self.use_avg:
                self.seq_model = AVGModel(self.channel_size, num_inputs=self.seq_emb_size)
                self.seq_output_size = self.seq_emb_size
            else:
                self.seq_model = AttentionModel(self.embedding_size * 4, self.channel_size, num_inputs=self.seq_emb_size, hidden_units=(self.seq_emb_size, self.seq_hidden_size), dropout=0.2, activation="tanh", weight_norm=True)
                self.seq_output_size = self.seq_emb_size
            print("Init seq part succeed")
        else:
            self.seq_output_size = 0

        """
        fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init fm part succeed")

        """
        ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.ffm_second_order_embeddings = nn.ModuleList([nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init ffm part succeed")

        """
        deep part
        """
        if self.use_deep:
            print("Init deep part")
            if not self.use_fm and not self.use_ffm:
                self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
            # 要加上 seq_emb 的输出维度
            self.linear_1 = nn.Linear(self.field_size * self.embedding_size + self.seq_output_size, deep_layers[0])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_'+str(i+1), nn.Linear(self.deep_layers[i-1], self.deep_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_'+str(i+1)+'_dropout', nn.Dropout(self.dropout_deep[i+1]))
            print("Init deep part succeed")

        """
        projection part
        """
        if self.use_fm and self.use_deep:
            concat_size = len(self.feature_sizes) + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            concat_size = len(self.feature_sizes) + self.embedding_size
        else:
            concat_size = self.deep_layers[-1]
        self.concat_projection = nn.Linear(concat_size, 1)
        print("Init succeed")

    def rank_loss(self, batch_preds, batch_stds):
        batch_top1_pros_pred = F.softmax(batch_preds, dim=0)
        batch_top1_pros_std = F.softmax(batch_stds, dim=0)
        batch_top1_pros_pred_log = torch.log(batch_top1_pros_pred)
        product = batch_top1_pros_std * batch_top1_pros_pred_log
        loss = -torch.sum(product, dim=0) / batch_preds.size(0)
        return loss

    def forward(self, Xi, Xp, Xv, X_seq, label=None):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k
        :param X_seq: seq input tensor, batch_size * seq_len * 1
        :return: the last output
        """
        # X_seq 做完变换再扩充
        product_num, _ = Xp.shape
        Xi = Xi.unsqueeze(0).unsqueeze(2).repeat(product_num, 1, 1)
        Xp = Xp.unsqueeze(2)
        Xi = torch.cat([Xi, Xp], 1)
        Xv = Xv.unsqueeze(0).repeat(product_num, 1).float()
        X_seq = X_seq.unsqueeze(0)
        """
        fm part
        """
        # Xi是 batch_size * k * 1 k是特征的数量（39）
        # fm_first_order_emb_arr 是 39 个 tensor 的list，每个 tensor 是 batch_size * 1 (1 是 emb_size)
        # fm_second_order_emb_arr 是 39 个 tensor 的list，每个 tensor 是 batch_size * 4 (4 是 emb_size)
        # ffm_second_order_emb_arr 是 39 个 list 的list，每个 list 是 field_size 个 tensor，每个 tensor 是 batch_size * 4
        # vip_emb = torch.cat(fm_second_order_emb_arr[-4:], 1)
        if self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)
            # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
            vip_emb = torch.cat(fm_second_order_emb_arr[-2:], 1)
            fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
            fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb    # (x+y)^2
            fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
            fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)    # x^2 + y^2
            fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
            if self.is_shallow_dropout:
                fm_second_order = self.fm_second_order_dropout(fm_second_order)

        """
        ffm part
        """
        if self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr, 1)
            if self.is_shallow_dropout:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs] for i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_wij_arr = []
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    ffm_wij_arr.append(ffm_second_order_emb_arr[i][j] * ffm_second_order_emb_arr[j][i])
            ffm_second_order = sum(ffm_wij_arr)
            if self.is_shallow_dropout:
                ffm_second_order = self.ffm_second_order_dropout(ffm_second_order)

        """
        deep part
        """
        if self.use_deep:
            if self.use_plain_emb:
                deep_emb = torch.cat([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)], 1)
            else:
                if self.use_fm:
                    deep_emb = torch.cat(fm_second_order_emb_arr, 1)
                elif self.use_ffm:
                    deep_emb = torch.cat([sum(ffm_second_order_embs) for ffm_second_order_embs in ffm_second_order_emb_arr], 1)

            if self.use_seq:
                if self.use_avg or self.use_lstm or self.use_tcn:
                    seq_emb = self.seq_model(X_seq)
                    seq_emb = seq_emb.repeat(product_num, 1)
                else:
                    X_seq = X_seq.repeat(product_num, 1)
                    seq_emb = self.seq_model(vip_emb, X_seq)
                deep_emb = torch.cat([deep_emb, seq_emb], 1)

            if self.deep_layers_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = F.tanh
            else:
                activation = F.relu
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
        """
        projection
        """
        if self.use_fm and self.use_deep:
            concat_input = torch.cat([fm_first_order, fm_second_order, x_deep], 1)
        elif self.use_fm:
            concat_input = torch.cat([fm_first_order, fm_second_order], 1)
        else:
            concat_input = torch.cat([x_deep], 1)

        output = self.concat_projection(concat_input)
        output = self.sigmoid(output)

        if label is not None:
            label = label.float()
            if self.loss_func == "rank":
                label *= 10
            loss = self.criterion(output, label)
            return loss
        else:
            return output


class Trainer:
    def __init__(self, epochs=1, batch_size=4, seed=1, use_ratio=1., split_ratio=0.8, lr=3e-4, weight_decay=0.0001,
                 optimizer="adam", lr_schedule="", warmup_steps=2000, use_grad_clip=False, max_grad=1.0,
                 use_apex=True, output_model=False, emb_dir="emb/", data_dir="/Volumes/移动硬盘/数据/会员商品/",
                 model_save_dir="model/", debug_mode=False, use_seq_emb=True, use_seq_cnt=True, embedding_size=10,
                 is_shallow_dropout=True, dropout_shallow=(0.5, 0.5), deep_layers=(32, 32), is_deep_dropout=True,
                 dropout_deep=(0.5, 0.5, 0.5), deep_layers_activation='relu', is_batch_norm=False, use_plain_emb=True,
                 use_lstm=False, use_tcn=False, use_avg=False, use_att=True, seq_emb_size=64, seq_hidden_size=32,
                 seq_pool="max", dense_product_feature=False, loss_func="rank", log_name="train_log",
                 ):
        self.device = torch.device('cuda')
        self.output_model = output_model
        self.debug_mode = debug_mode
        self.seed = seed
        self.seed_everything()
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_ratio = use_ratio
        self.split_ratio = split_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.use_grad_clip = use_grad_clip
        self.max_grad = max_grad
        self.use_apex = use_apex
        self.emb_dir = emb_dir
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir

        self.use_seq_emb = use_seq_emb
        self.use_seq_cnt = use_seq_cnt
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.is_batch_norm = is_batch_norm
        self.use_plain_emb = use_plain_emb
        self.use_lstm = use_lstm
        self.use_tcn = use_tcn
        self.use_avg = use_avg
        self.use_att = use_att
        self.seq_emb_size = seq_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.seq_pool = seq_pool
        self.dense_product_feature = dense_product_feature
        self.loss_func = loss_func
        self.log_name = log_name
        self.trade_duration_dict = {"7d": 0, "31d": 1, "93d": 2, "186d": 3, "365d": 4, "372d": 4}
        with open(os.path.join(self.emb_dir, "emb_index_dict.pkl"), "rb") as f:
            self.emb_index_dict = pickle.load(f)
        with open(os.path.join(self.emb_dir, "bucket_dict.pkl"), "rb") as f:
            self.bucket_dict = pickle.load(f)

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def is_float(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    def feature_indexing(self, feature, feature_dict):
        if feature in feature_dict:
            return feature_dict[feature]
        else:
            return 0

    def feature_bucketing(self, feature, bucket):
        for i, threshold in enumerate(bucket):
            if feature <= threshold:
                return i
        return len(bucket)

    def convert_date(self, date):
        year, month, day = map(lambda x: int(x), date.split("-"))
        return datetime.date(year, month, day)

    def compute_feature_size(self):
        feature_sizes = [
                         # 离散特征
                         len(self.emb_index_dict["gender"]) + 1,
                         len(self.emb_index_dict["age"]) + 1,
                         len(self.emb_index_dict["os"]) + 1,
                         len(self.emb_index_dict["brand"]) + 1,
                         len(self.emb_index_dict["model"]) + 1,
                         len(self.emb_index_dict["country"]) + 1,
                         len(self.emb_index_dict["province"]) + 1,
                         len(self.emb_index_dict["city"]) + 1,
                         len(self.emb_index_dict["first_buy"]) + 1,
                         len(self.emb_index_dict["zhibo_flag"]) + 1,
                         # 连续特征
                         len(self.bucket_dict["sum_fee_bucket"]) + 2,
                         len(self.bucket_dict["stay_day_bucket"]) + 2
                         ]
        if self.use_seq_cnt:
            feature_sizes += [
                          # 可能删除的收听序列相关特征
                          len(self.bucket_dict["program_cnt_bucket"]) + 1,
                          len(self.bucket_dict["channel_cnt_bucket"]) + 1,
                          len(self.bucket_dict["cate_cnt_bucket"]) + 1
                          ]
        feature_sizes += [
                         # 最后拼接的商品特征
                         5, # 时长
                         2, # 连续
                         ]
        return feature_sizes

    def load_data(self):
        with open(os.path.join(self.data_dir, "test_vip_feature1.tsv"), "r") as f:
            data_list = []
            for i, line in enumerate(f):
                if self.debug_mode and i == 100:
                    break
                if (i + 1) % 10000 == 0:
                    print("已处理 %d" % (i + 1))
                field_list = line.strip().split("\t")
                if len(field_list) != 19:
                    continue
                device_id, gender, age, device_os, brand, model, isp, country, province, city, play_list, trade_date, trade_fee, continuous_flag, first_buy, sum_fee, first_date, zhibo_flag, group_id = field_list
                if self.is_float(group_id):
                    group_id = int(group_id)
                else:
                    continue
                if trade_date == "":
                    first_buy = 1
                else:
                    first_buy = 0
                trade_date = "2019-11-01"
                brand = brand.lower()
                model = model.lower()
                products_list = []
                # 类别数据 emb_size 是 dict_len + 1
                for duration, con in [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [4, 0], [4, 1]]:
                    products_list.append([duration, con])
                feature_list = [self.feature_indexing(gender, self.emb_index_dict["gender"]),
                                self.feature_indexing(age, self.emb_index_dict["age"]),
                                self.feature_indexing(device_os, self.emb_index_dict["os"]),    # key 不是 device_os
                                self.feature_indexing(brand, self.emb_index_dict["brand"]),
                                self.feature_indexing(model, self.emb_index_dict["model"]),
                                self.feature_indexing(country, self.emb_index_dict["country"]),
                                self.feature_indexing(province, self.emb_index_dict["province"]),
                                self.feature_indexing(city, self.emb_index_dict["city"]),
                                self.feature_indexing(first_buy, self.emb_index_dict["first_buy"]),
                                self.feature_indexing(zhibo_flag, self.emb_index_dict["zhibo_flag"])
                                ]
                # 连续型数据 feature_size 是 bucket_len + 2
                # TODO: trade_date改成今天
                if self.is_float(sum_fee):
                    feature_list.append(self.feature_bucketing(float(sum_fee), self.bucket_dict["sum_fee_bucket"]))
                else:
                    feature_list.append(len(self.bucket_dict["sum_fee_bucket"]) + 1)
                if trade_date and first_date:
                    trade_date = self.convert_date(trade_date)
                    first_date = self.convert_date(first_date)
                    stay_day = (trade_date - first_date).days
                    if 0 < stay_day < 4000:
                        feature_list.append(self.feature_bucketing(stay_day, self.bucket_dict["stay_day_bucket"]))
                    else:
                        feature_list.append(len(self.bucket_dict["stay_day_bucket"]) + 1)
                else:
                    feature_list.append(len(self.bucket_dict["stay_day_bucket"]) + 1)
                if play_list:
                    play_list = json.loads(play_list)
                    channel_list = []
                    cate_list = []
                    for item in play_list:
                        split_item = item.split("|")
                        if len(split_item) != 3:
                            continue
                        cate, channel, program = split_item
                        channel_list.append(self.feature_indexing(channel, self.emb_index_dict["channel"]))
                        cate_list.append(self.feature_indexing(cate, self.emb_index_dict["cate"]))
                    if self.use_seq_cnt:
                        feature_list.append(self.feature_bucketing(len(play_list), self.bucket_dict["program_cnt_bucket"]))
                        feature_list.append(self.feature_bucketing(len(set(channel_list)), self.bucket_dict["channel_cnt_bucket"]))
                        feature_list.append(self.feature_bucketing(len(set(cate_list)), self.bucket_dict["cate_cnt_bucket"]))
                else:
                    # 专辑和分类 emb_size 是 dict_len + 2 (0: 未知，len + 1: 白板填充)
                    channel_list = [len(self.emb_index_dict["channel"]) + 1] * 1
                    cate_list = [len(self.emb_index_dict["cate"]) + 1] * 1
                    if self.use_seq_cnt:
                        feature_list.extend([0, 0, 0])
                value_list = [1] * (len(feature_list) + len(products_list[0]))
                data_list.append([feature_list, value_list, channel_list, cate_list, products_list, group_id, device_id])
        return data_list

    def create_dataloader(self):
        data_list = self.load_data()
        tensors_list = []
        for result in data_list:
            tensors = list(torch.tensor(result[i], dtype=torch.long) for i in range(len(result) - 1))
            tensors.append(result[-1])
            tensors_list.append(tensors)
        test_dataset = MyDataset(tensors_list, batch_size=self.batch_size, shuffle=False)
        return test_dataset

    def evaluate(self, prob_list, label_list):
        # 粗粒度的 auc
        total_probs = list(chain(*prob_list))
        total_lables = list(chain(*label_list))
        macro_auc_score = roc_auc_score(total_lables, total_probs)
        # 细粒度的 auc
        micro_auc_score = 0
        for probs, labels in zip(prob_list, label_list):
            micro_auc_score += roc_auc_score(labels, probs)
        micro_auc_score /= len(label_list)
        # 分类准确率
        total_logits = []
        total_label_logits = []
        for probs, labels in zip(prob_list, label_list):
            total_logits.append(np.argmax(probs))
            total_label_logits.append(np.argmax(labels))
        acc_score = accuracy_score(total_label_logits, total_logits)
        return macro_auc_score, micro_auc_score, acc_score

    def predict(self):
        # 加载 dataloader
        test_dataset = self.create_dataloader()
        # 计算 特征维度
        feature_sizes = self.compute_feature_size()
        # 训练
        self.seed_everything()
        # 加载预训练模型
        model = DeepFM(field_size=len(feature_sizes), feature_sizes=feature_sizes, channel_size=len(self.emb_index_dict["channel"]) + 2,
                       embedding_size=self.embedding_size, is_shallow_dropout=self.is_shallow_dropout,
                       dropout_shallow=self.dropout_shallow, deep_layers=self.deep_layers, is_deep_dropout=self.is_deep_dropout,
                       dropout_deep=self.dropout_deep, deep_layers_activation=self.deep_layers_activation,  is_batch_norm=self.is_batch_norm,
                       random_seed=self.seed, use_fm=True, use_ffm=False, use_deep=True, use_cuda=True, use_plain_emb=self.use_plain_emb,
                       use_seq=self.use_seq_emb, use_lstm=self.use_lstm, use_tcn=self.use_tcn, use_avg=self.use_avg, use_att=self.use_att,
                       seq_emb_size=self.seq_emb_size, seq_hidden_size=self.seq_hidden_size, seq_pool=self.seq_pool, loss_func=self.loss_func
                        )
        # 开始验证
        if os.path.exists("/Volumes/hedongfeng/data/vip/"):
            model.load_state_dict(torch.load("/Volumes/hedongfeng/model/vip/model_92_89_72.bin", map_location="cpu"))
        else:
            model.load_state_dict(torch.load("/root/ctr/avg/model_92_90_72.bin"))
        result_list = []
        model.eval()
        for j, large_batch_data in enumerate(tqdm(test_dataset)):
            for test_batch_data in large_batch_data:
                if torch.cuda.is_available():
                    feature_list = test_batch_data[0].to(self.device)
                    value_list = test_batch_data[1].to(self.device)
                    channel_list = test_batch_data[2].to(self.device)
                    cate_list = test_batch_data[3].to(self.device)
                    products_list = test_batch_data[4].to(self.device)
                    group_id = test_batch_data[5].to(self.device)
                    qingting_id = test_batch_data[6]
                else:
                    feature_list = test_batch_data[0]
                    value_list = test_batch_data[1]
                    channel_list = test_batch_data[2]
                    cate_list = test_batch_data[3]
                    products_list = test_batch_data[4]
                    group_id = test_batch_data[5]
                    qingting_id = test_batch_data[6]
                probs = model(feature_list, products_list, value_list, channel_list)
                probs = probs.to('cpu').detach().numpy().tolist()
                result_list.append([qingting_id, group_id, probs])
        with open("result_list.pkl", "wb") as f:
            pickle.dump(result_list, f)


if __name__ == "__main__":
    if os.path.exists("/Volumes/hedongfeng/data/vip/"):
        data_dir = "/Volumes/hedongfeng/data/vip/"
    else:
        data_dir = "/root/ctr/"
    # weight_decay 可能会造成权重过小 nan
    # 每次改变了 load_data (比如 use_seq_cnt=True)，都要重新删掉 pkl 文件，重新加载
    trainer = Trainer(epochs=10, batch_size=16, seed=1, use_ratio=1, split_ratio=1, lr=3e-4, weight_decay=0.000,
                      optimizer="adam", lr_schedule="", warmup_steps=2000, use_grad_clip=True, max_grad=1.0,
                      use_apex=False, output_model=True, emb_dir="emb/", data_dir=data_dir,
                      model_save_dir="avg/", debug_mode=False, use_seq_emb=True, use_seq_cnt=True, embedding_size=10,
                      is_shallow_dropout=True, dropout_shallow=(0.5, 0.5), deep_layers=(32, 32), is_deep_dropout=True,
                      dropout_deep=(0.5, 0.5, 0.5), deep_layers_activation='relu', is_batch_norm=False, use_plain_emb=True,
                      use_lstm=False, use_tcn=False, use_avg=True, use_att=False, seq_emb_size=64, seq_hidden_size=32,
                      seq_pool="both", dense_product_feature=False, loss_func="bce", log_name="avg_modified")
    trainer.predict()



