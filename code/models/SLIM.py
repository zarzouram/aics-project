from typing import List

import gc

import torch
from torch import nn
from torch import Tensor

from models.representation import RepresentationNetwork
from models.generation import DRAW


class SLIM(nn.Module):
    def __init__(self, param: dict):
        """
        Parameters
        ----------

        """
        super(SLIM, self).__init__()

        bert_model_dir = param["bert_model_dir"]
        caption_embs_size = param["caption_embs_size"]
        views_emb_size = param["views_emb_size"]
        views_enc_size = param["views_enc_size"]
        scene_rep_size = param["scene_rep_size"]

        image_width = param["image_width"]
        image_height = param["image_height"]
        image_color = param["image_color"]
        iter_num = param["iter_num"]
        N = param["N"]
        draw_encoder_size = param["draw_encoder_size"]
        draw_decoder_size = param["draw_decoder_size"]
        z_size = param["z_size"]

        self.model_param = nn.Parameter(torch.empty(0))

        self.rep_model = RepresentationNetwork(
            bert_dir=bert_model_dir,
            caption_embs_size=caption_embs_size,
            viewpoints_size=views_emb_size,
            viewpoints_embs_size=views_enc_size,
            r_size=scene_rep_size)

        self.gen_model = DRAW(
            img_w=image_width,
            img_h=image_height,
            img_c=image_color,
            iter_num=iter_num,
            read_N=N,
            write_N=N,
            encoder_size=draw_encoder_size,
            decoder_size=draw_decoder_size,
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

        img = torch.squeeze(batch[0]).to(device)  # (B, 3, 64, 64)
        view_imgr = torch.squeeze(batch[1]).to(device)
        views_other = torch.squeeze(batch[2]).to(device)
        tokens_id = torch.squeeze(batch[3]).to(device)
        tokens_type_id = torch.squeeze(batch[4]).to(device)
        attention_mask = torch.squeeze(batch[5]).to(device)

        # bert model accepts tensors in shape of (batch_size, sequence_length),
        # refrom the sizes from (B, N, T) to (B*N, T)
        B = img.size(0)
        seq_len = tokens_id.size(-1)
        views_size = views_other.size(-1)
        # _, _, image_sizes = img.size()
        scene_input_num = views_other.size(-2)
        r = self.rep_model(
            bert_input_ids=tokens_id.view(-1, seq_len),
            bert_attention_mask=attention_mask.view(-1, seq_len),
            bert_token_type_ids=tokens_type_id.view(-1, seq_len),
            viewpoints=views_other.view(-1, views_size),
            n=scene_input_num)

        loss = self.gen_model.loss(x=img.view(B, -1),
                                   cond=torch.cat((r, view_imgr), dim=1))

        if device.type == "cuda":
            del img
            del view_imgr
            del views_other
            del tokens_id
            del tokens_type_id
            del attention_mask
            torch.cuda.empty_cache()
            gc.collect()

        return loss
