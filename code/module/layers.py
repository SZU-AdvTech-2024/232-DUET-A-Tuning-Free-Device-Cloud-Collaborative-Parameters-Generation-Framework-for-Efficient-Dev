import torch
import torch.nn as nn


class Dice(nn.Module):
    # Dice激活函数
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, )))
        self.epsilon = 1e-9

    def forward(self, x):

        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1 - p) + x.mul(p)

        return x


class ActivationUnit(nn.Module):
    """
    激活函数单元
    功能是计算用户购买行为与推荐目标之间的注意力系数，比如说用户虽然用户买了这个东西，但是这个东西实际上和推荐目标之间没啥关系，也不重要，所以要乘以一个小权重
    """

    def __init__(self, embedding_dim, dropout=0.2, fc_dims=[32, 16]):
        super(ActivationUnit, self).__init__()
        # 1.初始化fc层
        fc_layers = []
        # 2.输入特征维度
        input_dim = embedding_dim * 4
        # 3.fc层内容：全连接层（4*embedding,32）—>激活函数->dropout->全连接层（32,16）->.....->全连接层（16,1）
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        # 4.将上面定义的fc层，整合到sequential中
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior):
        """
            :param query:targe目标的embedding ->（输入维度） batch*1*embed 
            :param user_behavior:行为特征矩阵 ->（输入维度） batch*seq_len*embed
            :return out:预测目标与历史行为之间的注意力系数
        """
        # 1.获取用户历史行为序列长度
        seq_len = user_behavior.shape[1]
        # 2.序列长度*embedding
        queries = torch.cat([query] * seq_len, dim=1)
        # 3.前面的把四个embedding合并成一个（4*embedding）的向量，
        #  第一个向量是目标商品的向量，第二个向量是用户行为的向量，
        #  至于第三个和第四个则是他们的相减和相乘（这里猜测是为了添加一点非线性数据用于全连接层，充分训练）
        attn_input = torch.cat([
            queries, user_behavior, queries - user_behavior,
            queries * user_behavior
        ],
                               dim=-1)
        out = self.fc(attn_input)
        return out


class AttentionPoolingLayer(nn.Module):
    """
      注意力序列层
      功能是计算用户行为与预测目标之间的系数，并将所有的向量进行相加，这里的目的是计算出用户的兴趣的能力向量
    """

    def __init__(self, embedding_dim, dropout):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim=embedding_dim,
                                          dropout=dropout)

    def forward(self, query_ad, user_behavior, mask):
        """
          :param query_ad:targe目标x的embedding   -> （输入维度） batch*1*embed
          :param user_behavior:行为特征矩阵     -> （输入维度） batch*seq_len*embed
          :param mask:被padding为0的行为置为false  -> （输入维度） batch*seq_len*1
          :return output:用户行为向量之和，反应用户的爱好
        """
        # 1.计算目标和历史行为之间的相关性
        attns = self.active_unit(query_ad, user_behavior)
        # 2.注意力系数乘以行为
        output = user_behavior.mul(attns.mul(mask))
        # 3.历史行为向量相加
        output = user_behavior.sum(dim=1)
        return output
