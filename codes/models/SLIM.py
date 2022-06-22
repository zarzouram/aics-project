from typing import List, Optional

import torch
from torch import nn
from torch import Tensor

from models.transformer_encoder import CaptionEncoder
from models.representation import RepresentationNetwork
from models.generation import DRAW


class SLIM(nn.Module):

    def __init__(
            self,
            params: dict,
            pretrain: Optional[str] = None,  # draw or caption_encoder
    ):
        """
        Parameters
        ----------

        """
        super(SLIM, self).__init__()

        self.pretrain = pretrain

        vwp_size = params["vwp_size"]  # camera angle
        vwp_embd_sz = params["vwp_embd_sz"]  # angel encoder output size
        if pretrain is None or pretrain == "caption_encoder":
            # Captions word encoder
            vocab_size = params["vocab_size"]
            cptn_embs_sz = params[
                "cptn_embs_sz"]  # caption text embedding size
            self.embedding = nn.Embedding(vocab_size, cptn_embs_sz)
            self.caption_encoder = CaptionEncoder(vocab_size=vocab_size,
                                                  embd_size=cptn_embs_sz)

            if pretrain is None:
                # Scene representation netwerk
                scnr_size = params["scnr_size"]  # scene repvector dim
                scnr_hidden_dim = params[
                    "scnr_hidden_dim"]  # scene rep. hidden
                input_size = cptn_embs_sz + vwp_embd_sz

                self.viewpoint_encoder = nn.Linear(vwp_size, vwp_embd_sz)
                self.rep_model = RepresentationNetwork(
                    input_size=input_size,
                    hidden_dim=scnr_hidden_dim,
                    output_size=scnr_size)

            self.dropout = nn.Dropout(0.5)

        if pretrain is None or pretrain.find("draw") == -1:
            # Image generation network
            image_width = params["image_width"]
            image_height = params["image_height"]
            image_color = params["image_color"]

            # DRAW param
            iter_num = params["draw_iter_num"]
            draw_h_size = params["draw_h_size"]
            draw_z_size = params["draw_z_size"]
            cond_size = vwp_embd_sz
            if pretrain is None:
                cond_size += scnr_size

            self.viewpoint_target_encoder = nn.Linear(vwp_size, vwp_embd_sz)
            self.gen_model = DRAW(imw=image_width,
                                  imh=image_height,
                                  imc=image_color,
                                  iter_num=iter_num,
                                  h_size=draw_h_size,
                                  z_size=draw_z_size,
                                  cond_size=cond_size,
                                  initc_tunning=params["draw_initc_tunning"])

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

    def forward(self, batch: List[str, Tensor]) -> Tensor:

        # Sizes:
        # ------
        # batch_size: B
        # image shape: imh, imw, imc = 32, 32, 3
        # other_scenes_number: N=9
        # sequence_length: T
        # Caption encoder output size: CE
        # Viewpoints encoder output size: VE
        # Scene encoder output size: SE

        if self.pretrain is None:
            img = batch[0]  # (B, imc, imh, imw)
            views = batch[1]  # (B, N=9, 2)
            img_view = batch[2]  # (B)
            tokens = batch[3]  # (B, N=9, T)
        elif self.pretrain.find("draw") != 1:
            img = batch[0]  # (B*10, imc, imh, imw)
            views = batch[1]  # (B*10, 2)
        else:
            tokens = batch[3]  # (B*10, T)

        if self.pretrain is None or self.pretrain == "caption_encoder":
            # Caption tokens embedding
            tokens_embedd = self.dropout(self.embedding(tokens))
            tokens_embedd = self.caption_encoder(tokens_embedd)
            # (B, N=9, T, CE)

            if self.pretrain is None:
                # Camera angels encoding
                vw_embedd = self.viewpoint_encoder(views)  # (B, N, VE)

                # Scenes representation
                sentence_embedd = tokens_embedd.mean(2)
                r = self.rep_model(cpt_embs=sentence_embedd,
                                   viewpoints=vw_embedd[:, 1:])  # (B, CE)
            else:
                return tokens_embedd  # pretraining caption encoding

        if not self.prtrn_c:
            cond = self.viewpoint_target_encoder(img_view)  # (B, VE)
            if not self.prtrn_d:
                cond = torch.cat((cond, r), dim=1)

            # Image generation training
            output = self.gen_model(x=img, cond=cond)

        return output

    def generate(self, batch: List[Tensor]) -> Tensor:

        img = batch[0]  # (B, imc, imh, imw)
        views = batch[1]  # (B, N=9, 2)
        img_view = batch[2]  # (B)
        tokens = batch[3]  # (B, N=9, T)

        if not self.prtrn_d:
            vw_embedd = self.viewpoint_encoder(views)  # (B, 10, VE)
            tokens_embedd = self.embedding(tokens)
            tokens_embedd = self.caption_encoder(
                tokens_embedd)  # (B, 9, T, CE)
            r = self.rep_model(cpt_embs=tokens_embedd.mean(2),
                               viewpoints=vw_embedd[:, 1:])  # (B, CE)

        cond = self.viewpoint_target_encoder(img_view)  # (B, VE)
        if not self.prtrn_d:
            cond = torch.cat((cond, r), dim=1)

        output = self.gen_model.generate(x=img, cond=cond)

        return output
