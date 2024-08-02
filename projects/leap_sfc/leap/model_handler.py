from .types import Head, Feature, HookName, Vector, SAE, Model, Transcoder


class ModelHandler:
    """Handles a transformer model and associated SAEs"""

    model: Model
    saes: dict[HookName, SAE]
    transcoders: dict[HookName, Transcoder]

    def __init__(
        self,
        model: Model,
        saes: dict[HookName, SAE],
        transcoders: dict[HookName, Transcoder],
    ):
        self.model = model
        self.saes = saes
        self.transcoders = transcoders

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    def get_sae_for_head(self, head: Head) -> SAE | Transcoder:
        """Get the SAE for a given head"""
        if head.head_type == "mlp":
            return self.transcoders[head.hook_name_in]
        elif head.head_type == "att":
            return self.saes[head.hook_name_in]
        else:
            raise ValueError(f"Invalid head type: {head.head_type}")

    def get_n_features_at_head(self, head: Head) -> int:
        """Get the number of features at a head"""
        # NOTE: W_enc is of shape [d_in, d_sae]
        if head.head_type == "mlp":
            return self.transcoders[head.hook_name_in].W_enc.shape[1]
        elif head.head_type == "att":
            return self.saes[head.hook_name_in].W_enc.shape[1]
        else:
            raise ValueError(f"Invalid head type: {head.head_type}")

    def get_encoder_weight(self, feature: Feature) -> Vector:
        # NOTE: W_enc is of shape [d_in, d_sae]
        if feature.head_type == "mlp":
            return self.transcoders[feature.hook_name_in].W_enc[:, feature.feature_id]
        elif feature.head_type == "att":
            return self.saes[feature.hook_name_in].W_enc[:, feature.feature_id]
        else:
            raise ValueError(f"Invalid head type: {feature.head_type}")

    def get_decoder_weight(self, feature: Feature) -> Vector:
        # NOTE: W_dec is of shape [d_sae, d_out]
        """Returns the decoder weight for a given node"""
        if feature.head_type == "mlp":
            return self.transcoders[feature.hook_name_in].W_dec[feature.feature_id]
        elif feature.head_type == "att":
            return self.saes[feature.hook_name_in].W_dec[feature.feature_id]
        else:
            raise ValueError(f"Invalid head type: {feature.head_type}")
