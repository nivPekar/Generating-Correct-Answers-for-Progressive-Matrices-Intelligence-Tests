# -*- coding: utf-8 -*-


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.blocks import *


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class CEM_3res(nn.Module):
    def __init__(self, num_attr=10, num_rule=6, sample=False,dropout=True, with_meta=False, do_contrast=False, force_bias=False,
                 g_func='conv'):
        super(CEM_3res, self).__init__()
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.sample = sample
        self.do_contrast = do_contrast


        if dropout:
            _dropout = {
                'high': 0.1,
                'mid': 0.1,
                'low': 0.1,
                'mlp': 0.,
            }
        else:
            _dropout = {
                'high': 0.,
                'mid': 0.,
                'low': 0.,
                'mlp': 0.,
            }
        # Perception
        self.high_dim = 64
        self.perception_net_high = nn.Sequential(nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
                                                 nn.BatchNorm2d(32),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout2d(0.5),
                                                 nn.Conv2d(32, self.high_dim, kernel_size=3, stride=2, padding=1,
                                                           bias=False),
                                                 nn.BatchNorm2d(self.high_dim),
                                                 nn.ReLU(inplace=True))

        self.mid_dim = 128
        self.perception_net_mid = nn.Sequential(
            nn.Conv2d(self.high_dim, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['mid']),
            nn.Conv2d(64, self.mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
            )

        self.low_dim = 256
        self.perception_net_low = nn.Sequential(
            nn.Conv2d(self.mid_dim, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['low']),
            nn.Conv2d(128, self.low_dim, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.low_dim),
            nn.ReLU(inplace=True)
            )



        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.g_func = g_func
        if g_func == 'sum':
            raise ValueError('Multi_Res_Match_Deep expects g_func to be "conv"')

        elif g_func == 'conv':



            self.g_function_high = nn.Sequential(Reshape(shape=(-1, 3 * self.high_dim, 20, 20)),
                                                 conv3x3(3 * self.high_dim, self.high_dim),
                                                 ResBlock(self.high_dim, self.high_dim),
                                                 ResBlock(self.high_dim, self.high_dim))
            self.g_function_mid = nn.Sequential(Reshape(shape=(-1, 3 * self.mid_dim, 5, 5)),
                                                conv3x3(3 * self.mid_dim, self.mid_dim),
                                                ResBlock(self.mid_dim, self.mid_dim),
                                                ResBlock(self.mid_dim, self.mid_dim))
            self.g_function_low = nn.Sequential(Reshape(shape=(-1, 3 * self.low_dim, 1, 1)),
                                                conv1x1(3 * self.low_dim, self.low_dim),
                                                ResBlock1x1(self.low_dim, self.low_dim),
                                                ResBlock1x1(self.low_dim, self.low_dim))



            self.p_embed_function_high = nn.Sequential(Reshape(shape=(-1, 3 * self.high_dim, 20, 20)),
                                                 conv3x3(3 * self.high_dim, self.high_dim),
                                                 ResBlock(self.high_dim, self.high_dim))

            self.p_embed_function_mid = nn.Sequential(Reshape(shape=(-1, 3 * self.mid_dim, 5, 5)),
                                                conv3x3(3 * self.mid_dim, self.mid_dim),
                                                ResBlock(self.mid_dim, self.mid_dim))

            self.p_embed_function_low = nn.Sequential(Reshape(shape=(-1, 3 * self.low_dim, 1, 1)),
                                                conv3x3(3 * self.low_dim, self.low_dim),
                                                ResBlock(self.low_dim, self.low_dim))


        self.choice_comp_p_embed_high = nn.Sequential(Reshape(shape=(-1, 2 * self.high_dim, 20, 20)),
                                                 conv3x3(2 * self.high_dim, self.high_dim)
                                                      , ResBlock(self.high_dim, self.high_dim))

        self.choice_comp_p_embed_mid = nn.Sequential(Reshape(shape=(-1, 2 * self.mid_dim, 5, 5)),
                                                     conv3x3(2 * self.mid_dim, self.mid_dim),
                                                     ResBlock(self.mid_dim, self.mid_dim))

        self.choice_comp_p_embed_low = nn.Sequential(Reshape(shape=(-1, 2 * self.low_dim, 1, 1)),
                                                    conv3x3(2 * self.low_dim, self.low_dim),
                                                     ResBlock(self.low_dim, self.low_dim))


        #self.reduce_func = reduce_func

        self.conv_row_high = conv3x3(self.high_dim, self.high_dim)
        self.bn_row_high = nn.BatchNorm2d(self.high_dim)
        self.conv_col_high = conv3x3(self.high_dim, self.high_dim)
        self.bn_col_high = nn.BatchNorm2d(self.high_dim, )

        self.conv_row_mid = conv3x3(self.mid_dim, self.mid_dim)
        self.bn_row_mid = nn.BatchNorm2d(self.mid_dim)
        self.conv_col_mid = conv3x3(self.mid_dim, self.mid_dim)
        self.bn_col_mid = nn.BatchNorm2d(self.mid_dim)

        self.conv_row_low = conv3x3(self.low_dim, self.low_dim)
        self.bn_row_low = nn.BatchNorm2d(self.low_dim)
        self.conv_col_low = conv3x3(self.low_dim, self.low_dim)
        self.bn_col_low = nn.BatchNorm2d(self.low_dim)


        if not force_bias and 'dist' in ['dist', 'prod']:
            self.bn_row_high.register_parameter('bias', None)
            self.bn_col_high.register_parameter('bias', None)

            self.bn_row_mid.register_parameter('bias', None)
            self.bn_col_mid.register_parameter('bias', None)

            self.bn_row_low.register_parameter('bias', None)
            self.bn_col_low.register_parameter('bias', None)



        self.res1_high = ResBlock(self.high_dim, 2 * self.high_dim, stride=2,
                                  downsample=nn.Sequential(conv1x1(self.high_dim, 2 * self.high_dim, stride=2),
                                                           nn.BatchNorm2d(2 * self.high_dim)
                                                           )
                                  )


        self.res2_high = ResBlock(2 * self.high_dim, 128, stride=2,
                                  downsample=nn.Sequential(conv1x1(2 * self.high_dim, 128, stride=2),
                                                           nn.BatchNorm2d(128)
                                                           )
                                  )

        self.res1_mid = ResBlock(self.mid_dim, 2 * self.mid_dim, stride=2,
                                 downsample=nn.Sequential(conv1x1(self.mid_dim, 2 * self.mid_dim, stride=2),
                                                          nn.BatchNorm2d(2 * self.mid_dim)
                                                          )
                                 )


        self.res2_mid = ResBlock(2 * self.mid_dim, 128, stride=2,
                                 downsample=nn.Sequential(conv1x1(2 * self.mid_dim, 128, stride=2),
                                                          nn.BatchNorm2d(128)
                                                          )
                                 )


        self.mlp_dim_low = 128
        self.res1_low = nn.Sequential(conv1x1(self.low_dim, self.mlp_dim_low),
                                           nn.BatchNorm2d(self.mlp_dim_low),
                                           nn.ReLU(inplace=True))
        self.res2_low = ResBlock1x1(self.mlp_dim_low, self.mlp_dim_low)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



        self.mlp = nn.Sequential(nn.Linear(128*3, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(_dropout['mlp']),
                                 nn.Linear(256, 128, bias=False),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 1, bias=True))

        self.with_meta = with_meta
        if self.with_meta:
            self.meta_res1_high = ResBlock(self.high_dim, 2 * self.high_dim, stride=2,
                                      downsample=nn.Sequential(conv1x1(self.high_dim, 2 * self.high_dim, stride=2),
                                                               nn.BatchNorm2d(2 * self.high_dim)
                                                               )
                                      )
            self.meta_res2_high = ResBlock(2 * self.high_dim, 128, stride=2,
                                      downsample=nn.Sequential(conv1x1(2 * self.high_dim, 128, stride=2),
                                                               nn.BatchNorm2d(128)
                                                               )
                                      )
            self.meta_res1_mid = ResBlock(self.mid_dim, 2 * self.mid_dim, stride=2,
                                      downsample=nn.Sequential(conv1x1(self.mid_dim, 2 * self.mid_dim, stride=2),
                                                               nn.BatchNorm2d(2 * self.mid_dim)
                                                               )
                                      )
            self.meta_res2_mid = ResBlock(2 * self.mid_dim, 128, stride=2,
                                      downsample=nn.Sequential(conv1x1(2 * self.mid_dim, 128, stride=2),
                                                               nn.BatchNorm2d(128)
                                                               )
                                      )
            self.mlp_dim_low = 128
            self.meta_res1_low = nn.Sequential(conv1x1(self.low_dim, self.mlp_dim_low),
                                          nn.BatchNorm2d(self.mlp_dim_low),
                                          nn.ReLU(inplace=True))
            self.meta_res2_low = ResBlock1x1(self.mlp_dim_low, self.mlp_dim_low)


            self.mlp_meta = nn.Sequential(nn.Linear(128*3, 128, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, 12, bias=False))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def compute_meta_loss(self, meta_output,meta_target):
        meta_pred = torch.chunk(meta_output, chunks=12, dim=1)
        meta_target = torch.chunk(meta_target, chunks=12, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 12):
            meta_target_loss += F.binary_cross_entropy(F.sigmoid(meta_pred[idx]), meta_target[idx])
        return meta_target_loss / 12.


    def triples_just_2(self, input_features):
        N, _, C, H, W = input_features.shape

        row1_features = input_features[:, 0:3, :, :, :]  # N, 3, 64, 20, 20
        row2_features = input_features[:, 3:6, :, :, :]  # N, 3, 64, 20, 20


        col1_features = input_features[:, 0:8:3, :, :, :]  # N, 3, 64, 20, 20
        col2_features = input_features[:, 1:8:3, :, :, :]  # N, 3, 64, 20, 20

        return row1_features, row2_features, col1_features, col2_features



    def triples(self, input_features):
        N, _, C, H, W = input_features.shape
        choices_features = input_features[:, 8:, :, :, :].unsqueeze(2)  # N, 8, 64, 20, 20 -> N, 8, 1, 64, 20, 20

        row1_features = input_features[:, 0:3, :, :, :]  # N, 3, 64, 20, 20
        row2_features = input_features[:, 3:6, :, :, :]  # N, 3, 64, 20, 20
        row3_pre = input_features[:, 6:8, :, :, :].unsqueeze(1).expand(N, 8, 2, C, H,
                                                                       W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        row3_features = torch.cat((row3_pre, choices_features), dim=2).view(N * 8, 3, C, H, W)

        col1_features = input_features[:, 0:8:3, :, :, :]  # N, 3, 64, 20, 20
        col2_features = input_features[:, 1:8:3, :, :, :]  # N, 3, 64, 20, 20
        col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, 8, 2, C, H,
                                                                         W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        col3_features = torch.cat((col3_pre, choices_features), dim=2).view(N * 8, 3, C, H, W)
        return row1_features, row2_features, row3_features, col1_features, col2_features, col3_features

    def reduce(self, row_features, col_features, N):
        _, C, H, W = row_features.shape

        row1 = row_features[:N, :, :, :].unsqueeze(1)
        row2 = row_features[N:2 * N, :, :, :].unsqueeze(1)



        dist12 = (row1 - row2).pow(2)
        final_row_features = 1 - (dist12)

        col1 = col_features[:N, :, :, :].unsqueeze(1)
        col2 = col_features[N:2 * N, :, :, :].unsqueeze(1)


        dist12 = (col1 - col2).pow(2)
        final_col_features = 1 - (dist12)


        return final_row_features, final_col_features

    def forward(self, x):
        x = x.view(-1, 16, 80, 80)
        N, _, H, W = x.shape

        # Perception Branch
        input_features_high = self.perception_net_high(x.view(-1, 80, 80).unsqueeze(1))
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)


        # High res
        row1_cat_high, row2_cat_high, col1_cat_high, col2_cat_high = \
            self.triples_just_2(input_features_high.view(N, 16, self.high_dim, 20, 20))

        row_feats_high = self.g_function_high(torch.cat((row1_cat_high, row2_cat_high), dim=0))
        row_feats_high = self.bn_row_high(self.conv_row_high(row_feats_high))
        col_feats_high = self.g_function_high(torch.cat((col1_cat_high, col2_cat_high), dim=0))
        col_feats_high = self.bn_col_high(self.conv_col_high(col_feats_high))

        final_row_features_high, final_col_features_high = self.reduce(row_feats_high, col_feats_high, N)

        # mid res
        row1_cat_mid, row2_cat_mid, col1_cat_mid, col2_cat_mid = \
            self.triples_just_2(input_features_mid.view(N, 16, self.mid_dim, 5, 5))

        row_feats_mid = self.g_function_mid(torch.cat((row1_cat_mid, row2_cat_mid), dim=0))
        row_feats_mid = self.bn_row_mid(self.conv_row_mid(row_feats_mid))
        col_feats_mid = self.g_function_mid(torch.cat((col1_cat_mid, col2_cat_mid), dim=0))
        col_feats_mid = self.bn_col_mid(self.conv_col_mid(col_feats_mid))

        final_row_features_mid, final_col_features_mid = self.reduce(row_feats_mid, col_feats_mid, N)  # N, 8, 256, 5, 5

        # Low res
        row1_cat_low, row2_cat_low, col1_cat_low, col2_cat_low = \
            self.triples_just_2(input_features_low.view(N, 16, self.low_dim, 1, 1))

        row_feats_low = self.g_function_low(torch.cat((row1_cat_low, row2_cat_low), dim=0))
        row_feats_low = self.bn_row_low(self.conv_row_low(row_feats_low))
        col_feats_low = self.g_function_low(torch.cat((col1_cat_low, col2_cat_low), dim=0))
        col_feats_low = self.bn_col_low(self.conv_col_low(col_feats_low))


        final_row_features_low, final_col_features_low = self.reduce(row_feats_low, col_feats_low, N)  # N, 8, 256, 5, 5



        # ======== meta target predict ==========
        if self.with_meta:
            meta_res1_out_row_high = self.meta_res1_high(final_row_features_high.view(N, self.high_dim, 20, 20))
            out_meta_row_high = self.meta_res2_high(meta_res1_out_row_high.view(N, 2 * self.high_dim, 10, 10))
            final_meta_row_high = self.avgpool(out_meta_row_high)
            final_meta_row_high = final_meta_row_high.view(-1, 128)
            meta_res1_out_col_high = self.meta_res1_high(final_col_features_high.view(N, self.high_dim, 20, 20))
            out_meta_col_high = self.meta_res2_high(meta_res1_out_col_high.view(N, 2 * self.high_dim, 10, 10))
            final_meta_col_high = self.avgpool(out_meta_col_high)
            final_meta_col_high = final_meta_col_high.view(-1, 128)
            final_meta_high = final_meta_row_high + final_meta_col_high

            meta_res1_out_row_mid = self.meta_res1_mid(final_row_features_mid.view(N, self.mid_dim, 5, 5))
            out_meta_row_mid = self.meta_res2_mid(meta_res1_out_row_mid.view(N, 2 * self.mid_dim, 3, 3))
            final_meta_row_mid = self.avgpool(out_meta_row_mid)
            final_meta_row_mid = final_meta_row_mid.view(-1, 128)
            meta_res1_out_col_mid = self.meta_res1_mid(final_col_features_mid.view(N, self.mid_dim, 5, 5))
            out_meta_col_mid = self.meta_res2_mid(meta_res1_out_col_mid.view(N, 2 * self.mid_dim, 3, 3))
            final_meta_col_mid = self.avgpool(out_meta_col_mid)
            final_meta_col_mid = final_meta_col_mid.view(-1, 128)
            final_meta_mid = final_meta_row_mid + final_meta_col_mid

            meta_res1_out_row_low = self.meta_res1_low(final_row_features_low.view(N, self.low_dim, 1, 1))
            out_meta_row_low = self.meta_res2_low(meta_res1_out_row_low.view(N, self.mlp_dim_low, 1, 1))
            final_meta_row_low = self.avgpool(out_meta_row_low)
            final_meta_row_low = final_meta_row_low.view(-1, 128)
            meta_res1_out_col_low = self.meta_res1_low(final_col_features_low.view(N, self.low_dim, 1, 1))
            out_meta_col_low = self.meta_res2_low(meta_res1_out_col_low.view(N, self.mlp_dim_low, 1, 1))
            final_meta_col_low = self.avgpool(out_meta_col_low)
            final_meta_col_low = final_meta_col_low.view(-1, 128)
            final_meta_low = final_meta_row_low + final_meta_col_low


            final_meta = torch.cat((final_meta_low, final_meta_mid, final_meta_high), dim=1)
            final_meta = self.mlp_meta(final_meta)
        else:
            final_meta = None

        # ------------------- problem embedding ------------------------
        input_features_high = input_features_high.view(N, 16, self.high_dim, 20, 20)
        row3_pre_high = input_features_high[:, 6:8, :, :, :] #.unsqueeze(1)
        col3_pre_high = input_features_high[:, 2:8:3, :, :, :] #.unsqueeze(1)

        input_features_mid = input_features_mid.view(N, 16, self.mid_dim, 5, 5)
        row3_pre_mid = input_features_mid[:, 6:8, :, :, :]  # .unsqueeze(1)
        col3_pre_mid = input_features_mid[:, 2:8:3, :, :, :]  # .unsqueeze(1)

        input_features_low = input_features_low.view(N, 16, self.low_dim, 1, 1)
        row3_pre_low = input_features_low[:, 6:8, :, :, :] #.unsqueeze(1)
        col3_pre_low = input_features_low[:, 2:8:3, :, :, :]  #.unsqueeze(1)


        p_embed_row_high = torch.cat((row3_pre_high, final_row_features_high), dim=1)
        p_embed_row_mid = torch.cat((row3_pre_mid , final_row_features_mid), dim=1)
        p_embed_row_low = torch.cat((row3_pre_low, final_row_features_low), dim=1)


        p_embed_col_high = torch.cat((col3_pre_high, final_col_features_high), dim=1)
        p_embed_col_mid = torch.cat((col3_pre_mid , final_col_features_mid ), dim=1)
        p_embed_col_low = torch.cat((col3_pre_low, final_col_features_low), dim=1)


        p_embed_row_high = self.p_embed_function_high(p_embed_row_high)
        p_embed_col_high = self.p_embed_function_high(p_embed_col_high)
        p_embed_high = p_embed_row_high + p_embed_col_high

        p_embed_row_mid = self.p_embed_function_mid(p_embed_row_mid)
        p_embed_col_mid = self.p_embed_function_mid(p_embed_col_mid)
        p_embed_mid = p_embed_row_mid + p_embed_col_mid

        p_embed_row_low = self.p_embed_function_low(p_embed_row_low)
        p_embed_col_low = self.p_embed_function_low(p_embed_col_low)
        p_embed_low = p_embed_row_low + p_embed_col_low

        # -----------------------------------------------------------------


        p_embed_low_to_return = p_embed_low
        p_embed_mid_to_return = p_embed_mid
        p_embed_high_to_return = p_embed_high


        p_embed_high = p_embed_high.unsqueeze(1).unsqueeze(1).expand(N, 8, 1, self.high_dim, 20, 20)
        p_embed_mid = p_embed_mid.unsqueeze(1).unsqueeze(1).expand(N, 8, 1, self.mid_dim, 5, 5)
        p_embed_low = p_embed_low.unsqueeze(1).unsqueeze(1).expand(N, 8, 1, self.low_dim, 1, 1)

        choices_features_high = input_features_high[:, 8:, :, :, :].unsqueeze(2)
        choices_features_mid = input_features_mid[:, 8:, :, :, :].unsqueeze(2)
        choices_features_low = input_features_low[:, 8:, :, :, :].unsqueeze(2)

        res1_in_high = torch.cat((p_embed_high, choices_features_high), dim=2)
        res1_in_mid = torch.cat((p_embed_mid, choices_features_mid), dim=2)
        res1_in_low = torch.cat((p_embed_low, choices_features_low), dim=2)

        res1_in_high = self.choice_comp_p_embed_high(res1_in_high)
        res1_in_mid = self.choice_comp_p_embed_mid(res1_in_mid)
        res1_in_low = self.choice_comp_p_embed_low(res1_in_low)




        # Combine


        res1_out_high = self.res1_high(res1_in_high.view(N * 8, self.high_dim, 20, 20))
        res2_in_high = res1_out_high.view(N, 8, 2 * self.high_dim, 10, 10)
        out_high = self.res2_high(res2_in_high.view(N * 8, 2 * self.high_dim, 10, 10))
        final_high = self.avgpool(out_high)
        final_high = final_high.view(-1, 128)

        res1_out_mid = self.res1_mid(res1_in_mid.view(N * 8, self.mid_dim, 5, 5))
        res2_in_mid = res1_out_mid.view(N, 8, 2 * self.mid_dim, 3, 3)
        out_mid = self.res2_mid(res2_in_mid.view(N * 8, 2 * self.mid_dim, 3, 3))
        final_mid = self.avgpool(out_mid)
        final_mid = final_mid.view(-1, 128)

        res1_out_low = self.res1_low(res1_in_low.view(N * 8, self.low_dim, 1, 1))
        res2_in_low = res1_out_low.view(N, 8, self.mlp_dim_low, 1, 1)
        out_low = self.res2_low(res2_in_low.view(N * 8, self.mlp_dim_low, 1, 1))
        final_low = self.avgpool(out_low)
        final_low = final_low.view(-1, self.mlp_dim_low)





        # MLP

        final = torch.cat((final_low, final_mid, final_high), dim=1)
        final = self.mlp(final)
        return final.view(-1, 8), final_meta, p_embed_high_to_return, p_embed_mid_to_return, p_embed_low_to_return
