from typing import List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor

from codes.models.transformer_encoder import CaptionEncoder
from codes.models.representation import RepresentationNetwork
from codes.models.generation import DRAW


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
            self.caption_encoder = CaptionEncoder(vocab_size=vocab_size,
                                                  embd_size=cptn_embs_sz)

            if pretrain is None:
                # Scene representation netwerk
                scnr_dim = params["scnr_dim"]  # scene repvector dim
                scnr_h_dim = params["scnr_hidden_dim"]  # scene rep. hidden
                input_size = cptn_embs_sz + vwp_embd_sz

                self.viewpoint_encoder = nn.Linear(vwp_size, vwp_embd_sz)
                self.rep_model = RepresentationNetwork(input_size=input_size,
                                                       hidden_dim=scnr_h_dim,
                                                       output_size=scnr_dim)

        if pretrain is None or pretrain.find("draw") != -1:
            # Image generation network
            image_width = params["image_width"]
            image_height = params["image_height"]
            image_color = params["image_color"]

            # DRAW param
            iter_num = params["draw_iter_num"]
            draw_h_dim = params["draw_h_dim"]
            draw_z_dim = params["draw_z_dim"]
            cond_dim = vwp_embd_sz
            if pretrain is None:
                cond_dim += scnr_dim

            self.target_viewpoint_encoder = nn.Linear(vwp_size, vwp_embd_sz)
            self.gen_model = DRAW(imw=image_width,
                                  imh=image_height,
                                  imc=image_color,
                                  iter_num=iter_num,
                                  h_dim=draw_h_dim,
                                  z_dim=draw_z_dim,
                                  cond_dim=cond_dim,
                                  initc_tunning=params["draw_initc_tunning"])

        self.init_weights()

    def init_weights(self):
        if hasattr(self, "viewpoint_encoder"):
            nn.init.xavier_uniform_(self.viewpoint_encoder.weight.data)
        if hasattr(self, "target_viewpoint_encoder"):
            nn.init.xavier_uniform_(self.target_viewpoint_encoder.weight.data)

    def freeze_draw(self, freeze=True):
        for p in self.target_viewpoint_encoder.parameters():
            p.requires_grad = not freeze

        for p in self.gen_model.parameters():
            p.requires_grad = not freeze

    def load_pretrained(self, pretrained: str):
        pretrained_state = torch.load(pretrained,
                                      map_location=torch.device("cpu"))
        pretrained_state = pretrained_state["model"]
        state = self.state_dict()
        for pretrained_name, pretrained_param in pretrained_state.items():
            if pretrained_name in state:
                state[pretrained_name] = pretrained_param

        self.load_state_dict(state)

    def forward(self, batch: List[Tensor]) -> Tuple[Tensor]:

        # Sizes:
        # ------
        # batch_size: B
        # image shape: imh, imw, imc = 32, 32, 3
        # scenes_number: N=10 or 9
        # sequence_length: T
        # Caption encoder output size: CE
        # Viewpoints encoder output size: VE
        # Scene encoder output size: SE

        if self.pretrain is None:
            images, img_view, other_views, tokens, _ = batch
        elif self.pretrain == "draw":
            images, img_view = batch
        else:
            raise NotImplementedError
        # images: scene image, (B, 3, imh, imh)
        # img_view: the view angle of the image (B, 2)
        # other_views: the other nine angles views (B, 9, 2)
        # token: scene descriptiom of the the other nine angles views (B, 9, T)

        if self.pretrain is None or self.pretrain == "caption_encoder":
            # scene description
            tokens_embedd, _attns = self.caption_encoder(tokens)
            # (B, 9, T, CE)

            if self.pretrain is None:
                # Camera angels encoding
                vw_embedd = self.viewpoint_encoder(other_views)  # (B, 9, VE)
                # Scenes representation
                sentence_embedd = tokens_embedd.mean(2)
                r = self.rep_model(cpt_embs=sentence_embedd,
                                   viewpoints=vw_embedd[:, 1:])  # (B, CE)
            else:
                return tokens_embedd  # pretraining caption encoding

        if self.pretrain is None or self.pretrain == "draw":
            cond = self.target_viewpoint_encoder(img_view)  # (B, VE)
            if self.pretrain is None:
                cond = torch.cat((cond, r), dim=1)

            # Image generation training
            output = self.gen_model(x=images, cond=cond)

        return output

    def generate(self, batch: List[Tensor]) -> Tuple[Tensor]:

        if self.pretrain is None:
            images, img_view, other_views, tokens, _ = batch
        elif self.pretrain == "draw":
            images, img_view = batch
        else:
            raise NotImplementedError

        if self.pretrain is None:
            vw_embedd = self.viewpoint_encoder(other_views)  # (B, 10, VE)
            tokens_embedd, attns = self.caption_encoder(
                tokens)  # (B, 9, T, CE)
            r = self.rep_model(cpt_embs=tokens_embedd.mean(2),
                               viewpoints=vw_embedd[:, 1:])  # (B, CE)

        cond = self.target_viewpoint_encoder(img_view)  # (B, VE)
        if self.pretrain is None:
            cond = torch.cat((cond, r), dim=1)

        output = self.gen_model.generate(x=images, cond=cond)

        if self.pretrain is None:
            return output, attns
        if self.pretrain == "draw":
            return output
        else:
            raise NotImplementedError
