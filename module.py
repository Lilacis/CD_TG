import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random


class FFAttention(nn.Module):
    def __init__(self, weight_image=0.7, weight_text=0.3):
        super(FFAttention, self).__init__()
        self.weight_image = weight_image
        self.weight_text = weight_text

    def forward(self, text_features, image_features):
        fused_feature = (
            self.weight_image * image_features + self.weight_text * text_features
        )
        return fused_feature


class AttentionModule(nn.Module):
    def __init__(self, embed_size, num_heads=8):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads)

        self.text_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, image, text):
        """
        image: tensor of shape [B, 512, H, W] (image feature map)
        text: tensor of shape [B, 512] (text embedding)
        Returns: tensor of shape [B, 512, H, W] after attention mechanism
        """
        B, C, H, W = image.shape
        image = image.view(B, C, H * W)
        text = self.text_mlp(text)

        text = text.unsqueeze(-1).expand(-1, -1, H * W)
        image = image.permute(2, 0, 1)
        text = text.permute(2, 0, 1)

        # Residual connection
        attn_output, _ = self.attn(image, text, text)
        attn_output = attn_output + image
        attn_output = self.norm(attn_output) 

        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

        return attn_output


class ADaIN(nn.Module):
    def __init__(self, n_output, strength, eps=1e-5, disabled=False):
        super(ADaIN, self).__init__()
        self.eps = eps
        self.n_output = n_output
        self.disabled = disabled
        self.strength = strength

    def generate_medical_style_from_content(self, content):
        """
        Style generation based on medical image features that 
        simulate features of medical images (e.g., noise, contrast variation, etc.).

        Args.
        content (Tensor): the input content image [B, C, H, W].
        strength (float): the strength of the style, used to control the degree of transformation

        Returns: torch.
        torch.Tensor: the generated medical style image [B, C, H, W]
        """
        transformation_type = random.choice(['noise', 'contrast', 'boundary', 'low_contrast'])

        if transformation_type == 'noise':
            return self.apply_noise(content, self.strength)
        elif transformation_type == 'contrast':
            return self.apply_contrast(content, self.strength)
        elif transformation_type == 'boundary':
            return self.apply_boundary(content, self.strength)
        elif transformation_type == 'low_contrast':
            return self.apply_low_contrast(content, self.strength)

    def apply_noise(self, content, strength):
        """
        Noise is added to the content image to simulate the noise characteristics in medical images.
        """
        noise = torch.randn_like(content) * strength
        return content + noise

    def apply_contrast(self, content, strength):
        """
        Simulates contrast variations in medical images, such as the contrast of bones and tissues in CT images.
        """
        return F.relu(content * (1 + strength))

    def apply_boundary(self, content, strength):
        """
        Simulate boundary features in medical images to enhance boundary details (e.g., boundaries of tumors or tissues).
        Use high-frequency feature extraction instead of traditional edge detection.
        """
        high_freq_features = self.high_frequency_extraction(content)
        return content + strength * high_freq_features

    def high_frequency_extraction(self, content):
        """
        Convolution is used to extract high-frequency features and enhance details and edge information.
        """
        conv = nn.Conv2d(content.shape[1], content.shape[1], kernel_size=3, padding=1, groups=content.shape[1],
                         bias=False)
        device = content.device
        conv = conv.to(device)
        high_freq_features = conv(content)
        return high_freq_features

    def apply_low_contrast(self, content, strength):
        """
        Simulates low-contrast areas of a medical image to reduce the contrast of the image.
        :param content: input image (Tensor, [B, C, H, W])
        :param strength: the strength of the low contrast (0-1, the higher the value, the more pronounced the low contrast effect)
        :return: low contrast image
        """
        mean = torch.mean(content, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        content_low_contrast = mean + strength * (content - mean)

        return content_low_contrast

    def forward(self, content, style=None):
        """
        Args.
        content (Tensor): feature map for content image [B, C, H, W], optional, defaults to None
        style (Tensor): feature map for style image [B, C, H, W], optional, default is None

        If no style is provided, the style generated based on the content is used.
        """
        if self.disabled:
            return content
        if style is None:
            style = self.generate_medical_style_from_content(content)

        style_mean = style.mean([2, 3], keepdim=True)  # [B, C, 1, 1]
        style_std = style.std([2, 3], keepdim=True)   # [B, C, 1, 1]

    
        content_mean = content.mean([2, 3], keepdim=True)  # [B, C, 1, 1]
        content_std = content.std([2, 3], keepdim=True)   # [B, C, 1, 1]
        content_normalized = (content - content_mean) / (content_std + self.eps)
        stylized_content = content_normalized * style_std + style_mean

        return stylized_content




