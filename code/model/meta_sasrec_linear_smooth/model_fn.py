import torch
import torch.nn as nn
from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model
import logging

logger = logging.getLogger(__name__)


@model("meta_sasrec_smooth", MetaType.ModelBuilder)
class SasRecWithPositionAwareWeights(nn.Module):

    def __init__(self, model_conf):
        super(SasRecWithPositionAwareWeights, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)

        self._position_embedding = encoder.IDEncoder(
            model_conf.id_vocab,  #多少个词
            model_conf.id_dimension  #每个词的维度
        )

        self._id_encoder = encoder.IDEncoder(model_conf.id_vocab,
                                             model_conf.id_dimension)

        self._target_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * 2,
            [torch.nn.Tanh, None]  #两层全连接层
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension],
            [torch.nn.Tanh]  #一层全连接层
        )

        self._transformer = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4 * model_conf.id_dimension,
            dropout=0)

        initializer.default_weight_init(
            self._transformer.self_attn.in_proj_weight)  #初始化qkv权重
        initializer.default_weight_init(
            self._transformer.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer.self_attn.in_proj_bias)
        initializer.default_bias_init(
            self._transformer.self_attn.out_proj.bias)

        initializer.default_weight_init(
            self._transformer.linear1.weight)  #第一个线形层用来上升纬度
        initializer.default_bias_init(self._transformer.linear1.bias)
        initializer.default_weight_init(
            self._transformer.linear2.weight)  #第二个线形层用来降低纬度
        initializer.default_bias_init(self._transformer.linear2.bias)

        self._meta_classifier_param_list = common.HyperNetwork_FC(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
            batch=True,
            model_conf=model_conf)
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, features, fig1=False):
        with torch.no_grad():
            click_seq = features[consts.FIELD_CLK_SEQUENCE]
            batch_size = int(click_seq.shape[0])
            positions = torch.arange(0,
                                     int(click_seq.shape[1]),
                                     dtype=torch.float32).to(click_seq.device)
            positions = torch.tile(positions.unsqueeze(0), [batch_size, 1])
            mask = torch.not_equal(click_seq, 0)
            seq_length = torch.maximum(
                torch.sum(mask.to(torch.int32), dim=1) - 1,
                torch.tensor([0], device=mask.device))
            seq_length = seq_length.to(torch.long)
            sequence_length = click_seq.shape[1]

        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        pos_weights = 1 / (1 + self.alpha * (sequence_length - 1 - positions)
                           )  # 权重公式 1/(1+α(L-1-pos))
        pos_weights = pos_weights * mask.to(torch.float32)  # 遮盖填充位置
        pos_weights = pos_weights.unsqueeze(-1)  # B * L * 1
        # Weighted embeddings
        hist_embed = self._id_encoder(click_seq)
        hist_pos_embed = self._position_embedding(positions.long())
        hist_embed = hist_embed + hist_pos_embed
        weighted_hist_embed = hist_embed * pos_weights  # Apply weights

        # Transformer encoding
        atten_embed = self._transformer(
            torch.swapaxes(weighted_hist_embed, 0, 1))
        user_state = torch.swapaxes(atten_embed, 0, 1)[range(batch_size),
                                                       seq_length, :]

        user_embedding = self._meta_classifier_param_list(
            user_state, weighted_hist_embed,
            user_state.size()[0], seq_length)

        return torch.sum(user_embedding * target_embed, dim=1, keepdim=True)