import torch
import torch.nn as nn
from util import consts
from . import config
from ..model_meta import MetaType, model
from module import layers, common, encoder


@model("meta_din_smooth", MetaType.ModelBuilder)
class DIN(nn.Module):
    """
      模型主体
      功能是用户最近的历史40个购买物品是xxx时，购买y的概率是多少
    """

    def __init__(self, model_conf):
        super(DIN, self).__init__()
        assert isinstance(model_conf, config.ModelConfig)
        # 1.特征维度，就是输入的特征有多少个类
        self.feature_dim = model_conf.feature_dim
        self.id_dimension = model_conf.id_dimension
        self.mlp_dims = model_conf.mlp_dims
        self.dropout = model_conf.dropout
        self._id_encoder = encoder.IDEncoder(model_conf.id_vocab,
                                             model_conf.id_dimension)
        self._position_embedding = encoder.IDEncoder(
            model_conf.id_vocab,  #多少个词
            model_conf.id_dimension  #每个词的维度
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension],
            [torch.nn.Tanh]  #一层全连接层
        )
        self._target_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * 2,
            [torch.nn.Tanh, None]  #两层全连接层
        )
        # 3.注意力计算层（论文核心）
        self.AttentionActivate = layers.AttentionPoolingLayer(
            self.id_dimension, self.dropout)
        self._meta_classifier_param_list = common.HyperNetwork_FC(
            self.id_dimension, [self.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
            batch=True,
            model_conf=model_conf)
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, features):
        """
            x输入(behaviors*40,ads*1) ->（输入维度） batch*(behaviors+ads) 
        """
        with torch.no_grad():
            click_seq = features[consts.FIELD_CLK_SEQUENCE]
            batch_size = int(click_seq.shape[0])
            mask = torch.not_equal(click_seq, 0).unsqueeze(-1)
            seq_length = torch.maximum(
                torch.sum(mask.to(torch.int32), dim=1) -
                1,  #按行求和 -1 后续编码user_state时用到 将user_state编码到最后一个点击物品的位置上
                torch.Tensor([0]).to(device=mask.device))
            seq_length = seq_length.to(torch.long).squeeze(-1)
            positions = torch.arange(0,
                                     int(click_seq.shape[1]),
                                     dtype=torch.int32).to(
                                         click_seq.device)  #L  元素为0~L-1
            positions = torch.tile(positions.unsqueeze(0), [batch_size, 1])
            sequence_length = click_seq.shape[1]
        #target
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        pos_weights = 1 / (1 + self.alpha * (sequence_length - 1 - positions)
                           )  # 权重公式 1/(1+α(L-1-pos))
        pos_weights = pos_weights * mask.squeeze(-1).to(
            torch.float32)  # 遮盖填充位置
        pos_weights = pos_weights.unsqueeze(-1)  # B * L * 1
        # 4.对推荐目标进行向量嵌入
        query_ad = target_embed.unsqueeze(1)
        # 5.对用户行为进行embeding，注意这里的维度为(batch*历史行为长度*embedding长度)
        user_behavior = self._id_encoder(click_seq)
        hist_pos_embed = self._position_embedding(positions)
        hist_embed = user_behavior + hist_pos_embed
        weighted_hist_embed = hist_embed * pos_weights

        weighted_hist_embed = self._seq_trans(weighted_hist_embed)

        user_behavior_mask = user_behavior.mul(mask)
        user_interest = self.AttentionActivate(query_ad, user_behavior_mask,
                                               mask)

        user_embedding = self._meta_classifier_param_list(
            user_interest, weighted_hist_embed, user_interest.size(0),
            seq_length)
        return torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
