from typing import List

import torch
from torch import nn
from torch import Tensor

from models.transformer_encoder import CaptionEncoder
from models.representation import RepresentationNetwork
from models.generation import DRAW


class SLIM(nn.Module):

    def __init__(self,
                 param: dict,
                 pretrain_draw: bool = False,
                 pretrain_caption_encoder: bool = False):
        """
        Parameters
        ----------

        """
        super(SLIM, self).__init__()

        self.prtrn_d = pretrain_draw
        self.prtrn_c = pretrain_caption_encoder

        x1 = not pretrain_draw
        x2 = not pretrain_caption_encoder
        assert x1 or x2, "You can pretain only one module at a time"  # noqa" E5011

        vwp_size = param["vwp_size"]  # camera angle
        vwp_embd_sz = param["vwp_embd_sz"]  # camera angel embedder output size

        self.viewpoint_embeddings = nn.Linear(vwp_size, vwp_embd_sz)

        if not pretrain_draw:
            vocab_size = param["vocab_size"]
            cptn_embs_sz = param["cptn_embs_sz"]  # caption text embedding size

            self.embedding = nn.Embedding(vocab_size, cptn_embs_sz)
            self.caption_encoder = CaptionEncoder(vocab_size=vocab_size,
                                                  embd_size=cptn_embs_sz)

            if not pretrain_caption_encoder:
                scnr_size = param["scnr_size"]  # scene repvector dim
                scnr_hidden_dim = param["scnr_hidden_dim"]  # scene rep. hidden
                input_size = cptn_embs_sz + vwp_embd_sz

                self.rep_model = RepresentationNetwork(
                    input_size=input_size,
                    hidden_dim=scnr_hidden_dim,
                    output_size=scnr_size)

            self.dropout = nn.Dropout(0.5)

        if not pretrain_caption_encoder:
            # Image
            image_width = param["image_width"]
            image_height = param["image_height"]
            image_color = param["image_color"]

            # DRAW param
            iter_num = param["iter_num"]
            draw_h_size = param["draw_h_size"]
            draw_z_size = param["z_size"]
            cond_size = vwp_embd_sz
            if not pretrain_draw:
                cond_size += scnr_size

            self.gen_model = DRAW(
                imw=image_width,
                imh=image_height,
                imc=image_color,
                iter_num=iter_num,
                h_size=draw_h_size,
                z_size=draw_z_size,
                cond_size=cond_size,
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="sigmoid")
                if hasattr(m.weight, "bias") and m.weight.bias is not None:
                    nn.init.constant_(m.weight.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch: List[Tensor]) -> Tensor:

        # Sizes:
        # ------
        # batch_size: B
        # image shape: imh, imw, imc = 32, 32, 3
        # other_scenes_number: N=9
        # sequence_length: T
        # Caption encoder output size: CE
        # Viewpoints encoder output size: VE
        # Scene encoder output size: SE

        img = batch[0]  # (B, imc, imh, imw)
        views = batch[1]  # (B, N=9, 2)
        img_view = batch[2]  # (B)
        tokens = batch[3]  # (B, N=9, T)

        if not self.prtrn_d:
            # Caption tokens embedding
            tokens_embedd = self.dropout(self.embedding(tokens))
            tokens_embedd = self.caption_encoder(tokens_embedd)
            # (B, N=9, T, CE)

            if not self.prtrn_c:
                # Camera angels encoding
                vw_embedd = self.viewpoint_embeddings(views)  # (B, N, VE)

                # Scenes representation
                sentence_embedd = tokens_embedd.mean(2)
                r = self.rep_model(cpt_embs=sentence_embedd,
                                   viewpoints=vw_embedd[:, 1:])  # (B, CE)
            else:
                return tokens_embedd  # pretraining caption encoding

        if not self.prtrn_c:
            cond = img_view
            if not self.prtrn_d:
                cond = torch.cat((cond, r), dim=1)

            # Image generation training
            output = self.gen_model(x=img, cond=cond)

        return output

    def generate(self, batch: List[Tensor]) -> Tensor:

        img = batch[0]  # (B, 3, 32, 32)
        views = batch[1]  # (B, 10, 2)
        tokens = batch[2]  # (B, 9, T)

        vw_embedd = self.viewpoint_embeddings(views)  # (B, 10, VE)
        tokens_embedd = self.embedding(tokens)
        tokens_embedd = self.caption_encoder(tokens_embedd)  # (B, 9, T, CE)
        r = self.rep_model(cpt_embs=tokens_embedd.mean(2),
                           viewpoints=vw_embedd[:, 1:])  # (B, CE)

        image = self.gen_model.generate(x=img,
                                        cond=torch.cat((r, vw_embedd[:, 0]),
                                                       dim=1))

        return image
