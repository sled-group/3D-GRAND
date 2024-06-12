import torch
import torch.nn as nn

from llava.model.multimodal_encoder.three_detr_model.models.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torch.nn.utils.rnn import pad_sequence
from llava.model.multimodal_encoder.mask3d_model.position_embedding import (
    PositionEmbeddingCoordsSine,
)
from torch.nn.init import xavier_uniform_


class SimpleBBoxHead(nn.Module):
    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 1024,
    ):
        super().__init__()

        self.activation = nn.ReLU()

        # # round up to the nearest multiple of 4
        # new_vision_feat_dim_in = (vision_feat_dim_in + 3) // 4 * 4
        # self.vision_projection_mlp = nn.Sequential(
        #     nn.Linear(vision_feat_dim_in, new_vision_feat_dim_in),
        #     self.activation,
        #     nn.Linear(new_vision_feat_dim_in, new_vision_feat_dim_in),
        #     self.activation,
        #     nn.Linear(new_vision_feat_dim_in, new_vision_feat_dim_in),
        # )

        # encoder_layer = TransformerEncoderLayer(
        #     d_model=new_vision_feat_dim_in,
        #     nhead=4,
        #     dim_feedforward=dim_feedforward,
        #     dropout=0.0,
        #     activation="relu",
        #     normalize_before=False,
        # )
        # self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

        self.box_mlp = nn.Sequential(
            nn.Linear(vision_feat_dim_in * num_vision_feat + lm_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, 6),
        )

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_

        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, d_latents]

        Returns:
            _type_: _description_
        """

        # pre_encoder_vision_feat = self.vision_projection_mlp(
        #     vision_features_before_mm_projection
        # )  # (B, num_latents, new_vision_feat_dim_in)

        # # get padding mask by checking where zero vectors are
        # src_key_padding_mask = vision_features_before_mm_projection.eq(0).all(
        #     dim=-1
        # )  # (B, num_latents)

        # # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        # # note that vision_features_before_mm_projection already contains positional embeddings
        # _, encoder_output, _ = self.encoder(
        #     src=pre_encoder_vision_feat.permute(1, 0, 2),
        #     src_key_padding_mask=src_key_padding_mask,
        # )  # [num_latents, B, d_latents]

        # encoder_output = encoder_output.permute(1, 0, 2)  # [B, num_latents, d_latents]

        bbox_preds = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            # vision_feat = encoder_output[batch_idx].flatten()  # (1024 * 96,)
            vision_feat = vision_features_before_mm_projection[batch_idx].flatten()  # (1024 * 96,)
            for i in range(len(grd_token_hidden_states)):
                langauge_feat = grd_token_hidden_states[i]  # (D,)
                concat_feat = torch.cat((vision_feat, langauge_feat), dim=-1)
                bbox_pred = self.box_mlp(concat_feat)
                bbox_preds.append(bbox_pred)

        bbox_preds = torch.stack(bbox_preds, dim=0)  # (N, 6)
        return bbox_preds


class BBoxHead(nn.Module):
    """A simple MLP head for bounding box regression"""

    def __init__(self, lm_feat_dim_in: int, vision_feat_dim_in: int, dim_feedforward: int = 128):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=vision_feat_dim_in,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            # normalize_before=False,
        )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)

        decoder_layer = TransformerDecoderLayer(
            d_model=vision_feat_dim_in,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            normalize_before=False,
        )

        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=4, return_intermediate=False
        )

        self.language_projection = nn.Sequential(
            nn.Linear(lm_feat_dim_in, vision_feat_dim_in),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, vision_feat_dim_in),
        )

        self.activation = nn.GELU()
        self.box_mlp = nn.Sequential(
            nn.Linear(vision_feat_dim_in, 256),
            self.activation,
            nn.Linear(256, 256),
            self.activation,
            nn.Linear(256, 6),
        )

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_

        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, d_latents]

        Returns:
            _type_: _description_
        """

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        # note that vision_features_before_mm_projection already contains positional embeddings
        _, encoder_output, _ = self.encoder(
            src=vision_features_before_mm_projection.permute(1, 0, 2)
        )  # [num_latents, B, d_latents]

        # we need to mask out the attention between different ground tokens
        # because each ground token is independent of each other

        # Pad the list of hidden states to the longest sample
        grd_token_hidden_states_padded = pad_sequence(
            grd_token_hidden_states_list, batch_first=True, padding_value=0
        )  # (B, N', D), where N' is the number of ground tokens in the sample with the most ground tokens in the batch
        # Create a mask for the padding tokens, True means there will be no attention
        tgt_key_padding_mask = grd_token_hidden_states_padded.eq(0).all(dim=-1)  # (B, N')
        tgt_mask = self.create_diag_mask(grd_token_hidden_states_padded.shape[1]).to(
            grd_token_hidden_states_padded.device
        )  # (N', N')

        # decoder expects: npoints x batch x channel
        language_projected = self.language_projection(
            grd_token_hidden_states_padded
        )  # (B, N', d_latents)
        decoder_output, decoder_attns = self.decoder(
            tgt=language_projected.permute(1, 0, 2),  # [N', B, d_latents]
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # output, attns, output shape: [N', B, d_latents]

        # predict the bounding boxes
        bbox_preds = self.box_mlp(decoder_output)  # (N', B, 6)
        # flatten the first two dimensions, remove padded locations
        bbox_preds = bbox_preds.permute(1, 0, 2)  # (B, N', 6)
        # discard the padded locations
        bbox_preds = bbox_preds[~tgt_key_padding_mask]  # (num_boxes_in_batch, 6)

        return bbox_preds

    @staticmethod
    def create_diag_mask(size):
        # for transformer, a binary ``True`` value indicates that the corresponding position is NOT
        # allowed to attend, while a ``False`` value indicates that the position is allowed to attend.
        mask = torch.ones(size, size, dtype=torch.bool)
        mask.fill_diagonal_(0)
        return mask


class BBoxHeadForGroundTruthBboxRegressionV2(nn.Module):
    """A simple MLP head for bounding box regression"""

    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 1024,
    ):
        super().__init__()

        # round up to the nearest multiple of 4
        new_vision_feat_dim_in = (vision_feat_dim_in + 3) // 4 * 4
        self.vision_projection_mlp = nn.Sequential(
            nn.Linear(vision_feat_dim_in, new_vision_feat_dim_in),
        )

        self.activation = nn.ReLU()
        self.language_projection_mlp = nn.Sequential(
            nn.Linear(lm_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, new_vision_feat_dim_in),
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=new_vision_feat_dim_in,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            normalize_before=True,
        )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)

        self.activation = nn.ReLU()
        self.box_mlp = nn.Sequential(
            nn.Linear(new_vision_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, 6),
        )

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_

        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, d_latents]

        Returns:
            _type_: _description_
        """

        bbox_preds = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            # vision_feat = encoder_output[batch_idx].flatten()  # (1024 * 96,)
            vision_feat = vision_features_before_mm_projection[batch_idx].unsqueeze(
                0
            )  # (1, num_vision_feat, vision_feat_dim_in)
            vision_feat = self.vision_projection_mlp(
                vision_feat
            )  # (1, num_vision_feat, new_vision_feat_dim_in)
            for i in range(len(grd_token_hidden_states)):
                language_feat = grd_token_hidden_states[i]  # (D,)

                language_feat = self.language_projection_mlp(
                    language_feat
                )  # (new_vision_feat_dim_in,)

                language_feat = language_feat[None, None, :]  # (1, 1, new_vision_feat_dim_in)

                language_concat_vision_feat = torch.cat(
                    (language_feat, vision_feat), dim=1
                )  # (1, 1 + new_vision_feat_dim_in, new_vision_feat_dim_in)

                # # nn.MultiHeadAttention in encoder expects seqlen x batch x channel features
                _, encoder_output, _ = self.encoder(
                    src=language_concat_vision_feat.permute(1, 0, 2)
                )  # [1 + new_vision_feat_dim_in, 1, new_vision_feat_dim_in]

                fused_feat = encoder_output[0][0]  # (new_vision_feat_dim_in,)

                bbox_pred = self.box_mlp(fused_feat)  # (6,)
                bbox_preds.append(bbox_pred)

        bbox_preds = torch.stack(bbox_preds, dim=0)  # (N, 6)
        return bbox_preds


class BBoxHeadForGroundTruthBboxRegressionV1(nn.Module):
    """A simple MLP head for bounding box regression"""

    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 1024,
    ):
        super().__init__()

        self.bbox_pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=10,
            pos_type="fourier",
        )
        self.obj_class_embedding = nn.Embedding(
            265, 64
        )  # 265 classes in ScanNet, learnable embedding size 64

        self.activation = nn.ReLU()

        encoder_layer = TransformerEncoderLayer(
            d_model=10 * 2 + 64,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            normalize_before=False,
        )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.box_mlp = nn.Sequential(
            nn.Linear((10 * 2 + 64) * num_vision_feat + lm_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, 6),
        )

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_

        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, 6 + 1]

        Returns:
            _type_: _description_
        """
        # get bbox position embeddings
        # xyz is batch x npoints x 3
        min_xyz_pos_embeddings = self.bbox_pos_embedding(
            xyz=vision_features_before_mm_projection[:, :, 0:3]
        )  # (B, 96, num_latents)
        min_xyz_pos_embeddings = min_xyz_pos_embeddings.permute(0, 2, 1)  # (B, num_latents, 96)
        max_xyz_pos_embeddings = self.bbox_pos_embedding(
            xyz=vision_features_before_mm_projection[:, :, 3:6]
        )  # (B, 96, num_latents)
        max_xyz_pos_embeddings = max_xyz_pos_embeddings.permute(0, 2, 1)  # (B, num_latents, 96)
        # get the object class embeddings
        obj_classes = vision_features_before_mm_projection[:, :, -1].long()
        obj_class_embeddings = self.obj_class_embedding(obj_classes)  # (B, num_latents, 64)

        vision_feat = torch.concat(
            (min_xyz_pos_embeddings, max_xyz_pos_embeddings, obj_class_embeddings), dim=-1
        )  # (B, num_vision_feat, 96*2+64)

        # get padding mask by checking where zero vectors are
        src_key_padding_mask = vision_features_before_mm_projection.eq(0).all(
            dim=-1
        )  # (B, num_latents)

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        # note that vision_features_before_mm_projection already contains positional embeddings
        _, encoder_output, _ = self.encoder(
            src=vision_feat.permute(1, 0, 2),
            src_key_padding_mask=src_key_padding_mask,
        )  # [num_latents, B, d_latents]

        encoder_output = encoder_output.permute(1, 0, 2)  # [B, num_latents, d_latents]

        bbox_preds = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            vision_feat = encoder_output[batch_idx].flatten()  # (1024 * 96,)
            # vision_feat = vision_features_before_mm_projection[batch_idx].flatten()  # (1024 * 96,)
            for i in range(len(grd_token_hidden_states)):
                langauge_feat = grd_token_hidden_states[i]  # (D,)
                concat_feat = torch.cat((vision_feat, langauge_feat), dim=-1)
                bbox_pred = self.box_mlp(concat_feat)
                bbox_preds.append(bbox_pred)

        bbox_preds = torch.stack(bbox_preds, dim=0)  # (N, 6)
        return bbox_preds


class BBoxHeadForGroundTruthBboxSelectionTransformerLateFusion(nn.Module):
    """A simple MLP head for bounding box selection, for training on CE loss"""

    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 2048,
    ):
        super().__init__()

        class_emb_dim = 256
        pos_emb_dim = 16
        self.bbox_pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=pos_emb_dim,
            pos_type="fourier",
        )
        self.obj_class_embedding = nn.Embedding(
            265, class_emb_dim
        )  # 265 classes in ScanNet, learnable embedding size 64

        self.activation = nn.GELU()
        self.language_vision_fusion_mlp = nn.Sequential(
            nn.Linear(class_emb_dim + pos_emb_dim + lm_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
        )

        # encoder_layer = TransformerEncoderLayer(
        #     d_model=dim_feedforward,
        #     nhead=8,
        #     dim_feedforward=dim_feedforward,
        #     dropout=0.0,
        #     activation="relu",
        #     normalize_before=True,
        # )
        # self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=class_emb_dim + pos_emb_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.scoring_mlp = nn.Sequential(
            nn.Linear(dim_feedforward, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_

        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, 6 + 1]

        Returns:
            _type_: _description_
        """

        # get bbox position embeddings
        # xyz is batch x npoints x 3
        # get the center of the bbox
        bbox_center = (
            vision_features_before_mm_projection[:, :, 0:3]
            + vision_features_before_mm_projection[:, :, 3:6]
        ) / 2.0
        bbox_pos_embeddings = self.bbox_pos_embedding(
            xyz=bbox_center
        )  # (B, pos_emb_dim, num_latents)
        bbox_pos_embeddings = bbox_pos_embeddings.permute(0, 2, 1)  # (B, num_latents, pos_emb_dim)
        # get the object class embeddings
        obj_classes = vision_features_before_mm_projection[:, :, -1].long()
        obj_class_embeddings = self.obj_class_embedding(
            obj_classes
        )  # (B, num_latents, class_emb_dim)

        vision_feat = torch.concat(
            (obj_class_embeddings, bbox_pos_embeddings), dim=-1
        )  # (B, class_emb_dim + pos_emb_dim, class_emb_dim)

        # get padding mask by checking where zero vectors are
        src_key_padding_mask = vision_features_before_mm_projection.eq(0).all(
            dim=-1
        )  # (B, num_latents)

        bbox_scores = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            # vision_feat = vision_features_before_mm_projection[
            #     batch_idx
            # ]  # (num_latents, d_latents)
            cur_vision_feat = vision_feat[batch_idx]  # (num_latents, class_emb_dim)
            cur_vision_feat = cur_vision_feat.unsqueeze(0)  # (1, num_latents, class_emb_dim)
            cur_vision_feat = cur_vision_feat.permute(1, 0, 2)  # (num_latents, 1, class_emb_dim)
            # nn.MultiHeadAttention in encoder expects seqlen x batch x channel features
            cur_encoder_output = self.encoder(
                cur_vision_feat,
                src_key_padding_mask=src_key_padding_mask[batch_idx].unsqueeze(0),
            )  # [num_latents, 1, class_emb_dim]
            cur_encoder_output = cur_encoder_output.squeeze(1)  # (num_latents, class_emb_dim)

            for i in range(len(grd_token_hidden_states)):
                langauge_feat = grd_token_hidden_states[i]  # (lm_feat_dim_in,)
                # concat the language feat with each vision feat
                langauge_feat_repeat = langauge_feat.repeat(
                    cur_encoder_output.shape[0], 1
                )  # (num_latents, lm_feat_dim_in)
                concat_feat = torch.cat(
                    (cur_encoder_output, langauge_feat_repeat), dim=-1
                )  # (num_latents, class_emb_dim + lm_feat_dim_in)
                fused_feat = self.language_vision_fusion_mlp(
                    concat_feat
                )  # (num_latents, dim_feedforward)

                bbox_score = self.scoring_mlp(fused_feat).squeeze(-1)  # (num_latents,)
                bbox_scores.append(bbox_score)  # (num_latents)

        bbox_scores = torch.stack(bbox_scores, dim=0)  # (N, num_latents)
        return bbox_scores


class BBoxHeadForGroundTruthBboxSelectionTransformerEarlyFusion(nn.Module):
    """A simple MLP head for bounding box selection, for training on CE loss"""

    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 2048,
    ):
        super().__init__()

        class_emb_dim = 256
        pos_emb_dim = 16
        self.bbox_pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=pos_emb_dim,
            pos_type="fourier",
        )
        self.obj_class_embedding = nn.Embedding(
            265, class_emb_dim
        )  # 265 classes in ScanNet, learnable embedding size class_emb_dim

        self.activation = nn.GELU()
        self.language_projection_mlp = nn.Sequential(
            nn.Linear(lm_feat_dim_in, class_emb_dim),
        )

        # encoder_layer = TransformerEncoderLayer(
        #     d_model=dim_feedforward,
        #     nhead=8,
        #     dim_feedforward=dim_feedforward,
        #     dropout=0.0,
        #     activation="relu",
        #     normalize_before=True,
        # )
        # self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=class_emb_dim + pos_emb_dim + class_emb_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.scoring_mlp = nn.Sequential(
            nn.Linear(class_emb_dim + pos_emb_dim + class_emb_dim, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_

        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, 6 + 1]

        Returns:
            _type_: _description_
        """

        # get bbox position embeddings
        # xyz is batch x npoints x 3
        # get the center of the bbox
        bbox_center = (
            vision_features_before_mm_projection[:, :, 0:3]
            + vision_features_before_mm_projection[:, :, 3:6]
        ) / 2.0
        bbox_pos_embeddings = self.bbox_pos_embedding(
            xyz=bbox_center
        )  # (B, pos_emb_dim, num_latents)
        bbox_pos_embeddings = bbox_pos_embeddings.permute(0, 2, 1)  # (B, num_latents, pos_emb_dim)
        # get the object class embeddings
        obj_classes = vision_features_before_mm_projection[:, :, -1].long()
        obj_class_embeddings = self.obj_class_embedding(
            obj_classes
        )  # (B, num_latents, class_emb_dim)

        vision_feat = torch.concat(
            (obj_class_embeddings, bbox_pos_embeddings), dim=-1
        )  # (B, num_latents, class_emb_dim)

        # get padding mask by checking where zero vectors are
        src_key_padding_mask = vision_features_before_mm_projection.eq(0).all(
            dim=-1
        )  # (B, num_latents)

        bbox_scores = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            # vision_feat = vision_features_before_mm_projection[
            #     batch_idx
            # ]  # (num_latents, d_latents)
            cur_vision_feat = vision_feat[batch_idx]  # (num_latents, class_emb_dim + pos_emb_dim)

            for i in range(len(grd_token_hidden_states)):
                langauge_feat = grd_token_hidden_states[i]  # (lm_feat_dim_in,)
                langauge_feat = self.language_projection_mlp(langauge_feat)  # (lm_feat_dim_in,)
                langauge_feat_repeat = langauge_feat.repeat(
                    cur_vision_feat.shape[0], 1
                )  # (num_latents, lm_feat_dim_in)

                concat_feat = torch.cat(
                    (cur_vision_feat, langauge_feat_repeat), dim=-1
                )  # (num_latents, class_emb_dim + pos_emb_dim + lm_feat_dim_in)

                concat_feat = concat_feat.unsqueeze(
                    0
                )  # (1, num_latents, class_emb_dim + pos_emb_dim + lm_feat_dim_in)
                concat_feat = concat_feat.permute(
                    1, 0, 2
                )  # (num_latents, 1, class_emb_dim + pos_emb_dim + lm_feat_dim_in)
                # nn.MultiHeadAttention in encoder expects seqlen x batch x channel features
                cur_encoder_output = self.encoder(
                    concat_feat,
                    src_key_padding_mask=src_key_padding_mask[batch_idx].unsqueeze(0),
                )  # [num_latents, 1, class_emb_dim + pos_emb_dim + lm_feat_dim_in]
                cur_encoder_output = cur_encoder_output.squeeze(1)  # (num_latents, class_emb_dim)

                bbox_score = self.scoring_mlp(cur_encoder_output).squeeze(-1)  # (num_latents,)
                bbox_scores.append(bbox_score)  # (num_latents)

        bbox_scores = torch.stack(bbox_scores, dim=0)  # (N, num_latents)
        return bbox_scores


class BBoxHeadForGroundTruthBboxSelectionMLPPosEmbAndFusionOneHot(nn.Module):
    """A simple MLP head for bounding box selection, for training on CE loss"""

    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 4096,
    ):
        super().__init__()

        self.class_emb_dim = class_emb_dim = 265  # 265 classes in ScanRefer
        pos_emb_dim = 16
        self.bbox_pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=pos_emb_dim,
            pos_type="fourier",
        )

        self.activation = nn.ReLU()
        self.language_vision_fusion_mlp = nn.Sequential(
            nn.Linear(class_emb_dim + pos_emb_dim + lm_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
        )

        self.scoring_mlp = nn.Sequential(
            nn.Linear(dim_feedforward, 1),
        )

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_
        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, 6 + 1]
        Returns:
            _type_: _description_
        """

        # get bbox position embeddings
        # xyz is batch x npoints x 3
        # get the center of the bbox
        bbox_center = (
            vision_features_before_mm_projection[:, :, 0:3]
            + vision_features_before_mm_projection[:, :, 3:6]
        ) / 2.0
        bbox_pos_embeddings = self.bbox_pos_embedding(
            xyz=bbox_center
        )  # (B, pos_emb_dim, num_latents)
        bbox_pos_embeddings = bbox_pos_embeddings.permute(0, 2, 1)  # (B, num_latents, pos_emb_dim)
        # get the object class embeddings, one-hot encoding of self.class_emb_dim classes
        obj_classes = vision_features_before_mm_projection[:, :, -1].long()
        obj_class_embeddings = torch.eye(
            self.class_emb_dim,
            device=vision_features_before_mm_projection.device,
            dtype=vision_features_before_mm_projection.dtype,
        )[
            obj_classes
        ]  # (B, num_latents, class_emb_dim)

        vision_feat = torch.concat(
            (obj_class_embeddings, bbox_pos_embeddings), dim=-1
        )  # (B, num_latents, class_emb_dim + pos_emb_dim)

        # get padding mask by checking where zero vectors are
        src_key_padding_mask = vision_features_before_mm_projection.eq(0).all(
            dim=-1
        )  # (B, num_latents)
        # for the padded locations, we set the vision_feat to be zero
        vision_feat[src_key_padding_mask] = 0

        bbox_scores = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            cur_vision_feat = vision_feat[batch_idx]  # (num_latents, d_latents)
            for i in range(len(grd_token_hidden_states)):
                langauge_feat = grd_token_hidden_states[i]  # (lm_feat_dim_in),)
                # concat the language feat with each vision feat
                langauge_feat_repeat = langauge_feat.repeat(
                    cur_vision_feat.shape[0], 1
                )  # (num_latents, lm_feat_dim_in)
                concat_feat = torch.cat(
                    (cur_vision_feat, langauge_feat_repeat), dim=-1
                )  # (num_latents, d_latents + lm_feat_dim_in)
                fused_feat = self.language_vision_fusion_mlp(concat_feat)
                bbox_score = self.scoring_mlp(fused_feat).squeeze(-1)  # (num_latents,)
                bbox_scores.append(bbox_score)  # (num_latents)

        bbox_scores = torch.stack(bbox_scores, dim=0)  # (N, num_latents)
        return bbox_scores


class BBoxHeadForGroundTruthBboxSelectionMLPFusionBoxCoordsAndClassID(nn.Module):
    """A simple MLP head for bounding box selection, for training on CE loss"""

    def __init__(
        self,
        lm_feat_dim_in: int,
        vision_feat_dim_in: int,
        num_vision_feat: int,
        dim_feedforward: int = 1024,
    ):
        super().__init__()

        self.activation = nn.ReLU()
        self.language_vision_fusion_mlp = nn.Sequential(
            nn.Linear(vision_feat_dim_in + lm_feat_dim_in, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_feedforward),
        )

        self.scoring_mlp = nn.Sequential(
            nn.Linear(dim_feedforward, 1),
        )

    def forward(
        self,
        grd_token_hidden_states_list: list[torch.Tensor],
        vision_features_before_mm_projection: torch.Tensor,
    ):
        """_summary_
        Args:
            grd_token_hidden_states_list (list[torch.Tensor]): each element in this list
                contains the hidden states of the ground tokens in one sample, list[[varying N, D]]
            vision_features_before_mm_projection (torch.Tensor): [B, num_latents, 6 + 1]
        Returns:
            _type_: _description_
        """

        bbox_scores = []
        for batch_idx, grd_token_hidden_states in enumerate(grd_token_hidden_states_list):
            vision_feat = vision_features_before_mm_projection[
                batch_idx
            ]  # (num_latents, d_latents)
            for i in range(len(grd_token_hidden_states)):
                langauge_feat = grd_token_hidden_states[i]  # (lm_feat_dim_in),)
                # concat the language feat with each vision feat
                langauge_feat_repeat = langauge_feat.repeat(
                    vision_feat.shape[0], 1
                )  # (num_latents, lm_feat_dim_in)
                concat_feat = torch.cat(
                    (vision_feat, langauge_feat_repeat), dim=-1
                )  # (num_latents, d_latents + lm_feat_dim_in)
                fused_feat = self.language_vision_fusion_mlp(concat_feat)
                bbox_score = self.scoring_mlp(fused_feat).squeeze(-1)  # (num_latents,)
                bbox_scores.append(bbox_score)  # (num_latents)

        bbox_scores = torch.stack(bbox_scores, dim=0)  # (N, num_latents)
        return bbox_scores
