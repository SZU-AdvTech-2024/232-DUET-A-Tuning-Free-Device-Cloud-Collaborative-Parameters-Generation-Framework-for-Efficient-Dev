from typing import Dict
import torch
from torch import nn
from ..model_meta import MetaType, model
from . import config
from util import consts
from module import encoder, common
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@model("meta_gru4rec", MetaType.ModelBuilder)
class GRU4Rec(nn.Module):

    def __init__(self, model_conf):
        super(GRU4Rec, self).__init__()
        assert isinstance(model_conf, config.ModelConfig)
        self.id_dimension = model_conf.id_dimension
        self.gru_layers_num = model_conf.gru_layers_num
        self.gru = nn.GRU(input_size=self.id_dimension,
                          hidden_size=self.id_dimension,
                          num_layers=self.gru_layers_num,
                          batch_first=True,
                          bias=False)

        self._id_encoder = encoder.IDEncoder(model_conf.id_vocab,
                                             model_conf.id_dimension)
        self._target_trans = common.StackedDense(model_conf.id_dimension,
                                                 [model_conf.id_dimension] * 2,
                                                 [torch.nn.Tanh, None])
        self._target_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * 2,
            [torch.nn.Tanh, None]  #两层全连接层
        )
        self._position_embedding = encoder.IDEncoder(
            model_conf.id_vocab,  #多少个词
            model_conf.id_dimension  #每个词的维度
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension],
            [torch.nn.Tanh]  #一层全连接层
        )
        self.apply(self._init_weights)
        self._meta_classifier_param_list = common.HyperNetwork_FC(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
            batch=True,
            model_conf=model_conf)
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def _init_weights(self, module):
        """
        Initializes the weight value for the given module.

        Args:
        module (nn.Module): The module whose weights need to be initialized.
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)

    def forward(self, features):
        with torch.no_grad():
            item_seq = features[consts.FIELD_CLK_SEQUENCE]
            batch_size = int(item_seq.shape[0])
            mask = torch.not_equal(item_seq, 0)  #不等于0的位置为1，等于0的位置为0
            # B
            seq_length = torch.maximum(
                torch.sum(
                    mask.to(torch.int32), dim=1
                ),  #按行求和 -1 后续编码user_state时用到 将user_state编码到最后一个点击物品的位置上
                torch.tensor([0], dtype=torch.int32).to(device=mask.device))
            positions = torch.arange(0,
                                     int(item_seq.shape[1]),
                                     dtype=torch.int32).to(
                                         item_seq.device)  #L  元素为0~L-1
            positions = torch.tile(positions.unsqueeze(0), [batch_size, 1])
            sequence_length = item_seq.shape[1]

        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        pos_weights = self.alpha * torch.exp(
            (positions + 1 - sequence_length) /
            self.beta)  # 权重公式 exp(-r_i / beta)
        pos_weights = pos_weights * mask.to(torch.float32)  # 遮盖填充位置
        pos_weights = pos_weights.unsqueeze(-1)  # B * L * 1

        seq_emb = self._id_encoder(item_seq)
        packed_emb = pack_padded_sequence(seq_emb,
                                          seq_length.cpu(),
                                          batch_first=True,
                                          enforce_sorted=False)
        packed_output, _ = self.gru(packed_emb)
        seq_emb_gru, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 将有效长度转为最后一个时间步索引
        last_indices = (seq_length - 1).view(-1, 1,
                                             1).expand(-1, 1, seq_emb.size(-1))
        last_indices = last_indices.long()
        # 提取最后一个时间步的隐状态
        user_state = torch.gather(seq_emb_gru, dim=1,
                                  index=last_indices).squeeze(1)

        hist_pos_embed = self._position_embedding(positions)
        hist_embed = seq_emb + hist_pos_embed
        weighted_hist_embed = hist_embed * pos_weights
        # weighted_hist_embed = self._seq_trans(weighted_hist_embed)

        user_embedding = self._meta_classifier_param_list(
            user_state,  #来自click_seq
            weighted_hist_embed,  #来自click_seq
            user_state.size()[0],
            seq_length - 1)

        return torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
