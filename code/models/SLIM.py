from typing import List

import gc

import torch
from torch import nn
from torch import Tensor

from layers.text_encoding import TextEncoding
from models.representation import RepresentationNetwork
from models.generation import DRAW


class SLIM(nn.Module):
    def __init__(self, param: dict):
        """
        Parameters
        ----------

        """
        super(SLIM, self).__init__()

        views_emb_size = param["views_emb_size"]
        views_enc_size = param["views_enc_size"]
        scene_rep_size = param["scene_rep_size"]
        caption_embs_size = param["caption_embs_size"]

        image_width = param["image_width"]
        image_height = param["image_height"]
        image_color = param["image_color"]
        iter_num = param["iter_num"]
        draw_h_size = param["draw_h_size"]

        z_size = param["z_size"]

        self.model_param = nn.Parameter(torch.empty(0))

        self.embd = TextEncoding(hidden_size=caption_embs_size)

        self.rep_model = RepresentationNetwork(
            caption_embs_size=caption_embs_size,
            viewpoints_size=views_emb_size,
            viewpoints_embs_size=views_enc_size,
            r_size=scene_rep_size)

        self.gen_model = DRAW(
            img_w=image_width,
            img_h=image_height,
            img_c=image_color,
            iter_num=iter_num,
            h_size=draw_h_size,
            z_size=z_size,
            cond_size=scene_rep_size + views_emb_size,
        )

    def forward(self, batch: List[Tensor]) -> Tensor:

        # Sizes:
        # ------
        # batch_size: B
        # captions_number: N
        # sequence_length: T
        # bert hidden layer size: H
        # Caption encoder output size: CE
        # Viewpoints encoder output size: VE
        device = self.model_param.device

        img = batch[0].to(device)  # (B, 3, 64, 64)
        view_imgr = batch[1].to(device)  # (B, 1)
        views_other = batch[2].to(device)  # (B, N, 9)
        captions = batch[3].tolist()  # (B*N, 9)

        views_size = views_other.size(-1)
        scene_input_num = views_other.size(-2)

        captions_emds = self.embd(captions)
        r = self.rep_model(cpt_embs=captions_emds,
                           viewpoints=views_other.view(-1, views_size),
                           n=scene_input_num)

        output = self.gen_model(x=img, cond=torch.cat((r, view_imgr), dim=1))

        if device.type == "cuda":
            del img
            del view_imgr
            del views_other
            # torch.cuda.empty_cache()
            gc.collect()

        return output

    def generate(self, batch: List[Tensor]) -> Tensor:
        device = self.model_param.device

        img = batch[0].to(device)  # (B, 3, 64, 64)
        view_imgr = batch[1].to(device)  # (B, 1)
        views_other = batch[2].to(device)  # (B, N, 9)
        captions = batch[3].tolist()  # (B*N, 9)

        # B = img.size(0)
        views_size = views_other.size(-1)
        scene_input_num = views_other.size(-2)

        captions_emds = self.embd(captions)
        r = self.rep_model(cpt_embs=captions_emds,
                           viewpoints=views_other.view(-1, views_size),
                           n=scene_input_num)

        image = self.gen_model.generate(x=img,
                                        cond=torch.cat((r, view_imgr), dim=1))

        return image
