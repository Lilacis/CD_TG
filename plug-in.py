from encoder import CLIPModel
from module import ADaIN, AttentionModule, FFAttention

import torch
from torch import nn
import torch.nn.functional as F

class BaseLineNet(nn.Module):
    def __init__(self, backbone, clip):
        super(BaseLineNet, self).__init__()
        
        ######### The Encoder of Baseline model #############
        self.encoder = backbone  # you should modify it as appropriate

        ############# Initialize CD-TG module ############
        if clip == 'RN50':
            self.clip = CLIPModel(model_name="RN50", model_path="/path/to/your/RN50.pt")
        elif clip == 'ViT-B-32':
            self.clip = CLIPModel(model_name="ViT-B-32", model_path="/path/to/your/ViT-B-32.pt")
        elif clip == 'ViT-B-16':
            self.clip = CLIPModel(model_name="ViT-B-16", model_path="/path/to/your/ViT-B-16.pt")
        else:
            raise Exception('Unavailable clip: %s' % clip)

        self.ADaIN = ADaIN(n_output=512, strength=0.3)
        self.attention = AttentionModule()
        self.FAN1 = FFAttention()

    def forward(self, img_s, mask_s, img_q, idx, model_name, train):
       
        feature_s = self.encoder(img_s)
        feature_q = self.encoder(img_q)

        ################### ADaIN #####################
        # Aligning features of support images using ADaIN
        # when testing and finetune, close the ADaIN
        if train:
            feature_s = self.ADaIN(feature_s) 

        ################ Leveraging the text_guidance into CD-FSS ####################
        bdx = torch.zeros(img_q.shape[0], dtype=torch.long)
        text_embedding_f = self.clip.forward(idx, model_name)
        text_embedding_b = self.clip.forward(bdx, 'background')

        feature_st, feature_qt, FP_t, BP_t = self.generate_prototype(text_embedding_f, text_embedding_b, feature_s,
                                                                                               feature_q, mask_s)
        feature_st = torch.cat(feature_st, dim=0)
        FP_t = torch.mean(torch.cat(FP_t, dim=0), dim=0, keepdim=True)
        BP_t = torch.mean(torch.cat(BP_t, dim=0), dim=0, keepdim=True)
        
        # foreground(target class) and background prototypes FP and BP from baseline method
        FP = self.FAN1(FP_t, FP)
        BP = self.FAN1(BP_t, BP)
        
        ######## The decoder of Baseline model ###########
        pred_mask = self.decoder(feature_s, feature_q, FP, BP, mask_s) # you should modify it as appropriate

        return pred_mask

    def generate_prototype(self, text_embedding, text_embedding_b, img_s, img_q, support_mask):
        """
        generate prototype
        :param text_embedding: (B, C)
        :param text_embedding_b: (B, C)
        :param img_s: {(B, C, H, W)}
        :param img_q: (B, C, H, W)
        :param support_mask: {(B, 1, H, W)}
        :return: prototype, {(1, C)}
        """
        eps = 1e-6
        img_st_list = []
        proto_flist = []
        proto_blist = []
         ## [layernum, batchsize, C, H, W]
        for k in range(len(img_s)):
            mask = F.interpolate(support_mask[k].unsqueeze(0).float(), img_s[k].size()[2:], mode='bilinear',
                                 align_corners=True)
            mask = mask.permute(1, 0, 2, 3)
            bg_mask = 1 - mask
            support_features = self.attention(img_s[k], text_embedding)
            support_features_b = self.attention(img_s[k], text_embedding_b)
            img_st_list.append(support_features)

            support_features_ = support_features * mask
            support_features_b = support_features_b * bg_mask

            ### prototype
            proto_f = support_features_.sum((2, 3))
            label_sum = mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            proto_flist.append(proto_f)
            proto_b = support_features_b.sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b / (label_sum + eps)
            proto_blist.append(proto_b)

        img_qt = self.attention(img_q, text_embedding)

        return img_st_list, img_qt, proto_flist, proto_blist
