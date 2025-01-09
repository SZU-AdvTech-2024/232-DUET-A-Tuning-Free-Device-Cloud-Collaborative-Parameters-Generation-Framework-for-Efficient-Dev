from ..model_meta import MetaType, model


class ModelConfig(object):

    def __init__(self):
        self.feature_dim = 3706
        self.id_dimension = 8
        self.mlp_dims = [64, 32]
        self.dropout = 0.2
        self.mlp_layers = 2
        self.id_vocab = 6100

    @staticmethod
    @model("meta_din_linear", MetaType.ConfigParser)
    def parse(json_obj):
        conf = ModelConfig()
        conf.feature_dim = json_obj.get("feature_dim")
        conf.id_dimension = json_obj.get("id_dimension")
        conf.mlp_dims = json_obj.get("mlp_dims")
        conf.dropout = json_obj.get("dropout")
        conf.mlp_layers = json_obj.get("mlp_layers")
        conf.id_vocab = json_obj.get("id_vocab")

        return conf
