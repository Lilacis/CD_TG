import json
from torchvision.transforms import ToPILImage
import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
import warnings
import open_clip
warnings.filterwarnings("ignore")


class CLIPModel:
    def __init__(self, model_name="ViT-B-32", model_path="/media/cs4007/disk2/ViT-B-32.pt", r=8, lora_alpha=32):
        super(CLIPModel, self).__init__()
        """
        Initialize Clip Model
        :param model_name: Name of the CLip, e.g. “ViT-B-32” or “RN50”.
        :param model_path: Local model weight paths
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_path = model_path
        self.to_pil = ToPILImage()
        self.model, self.preprocess, self.tokenizer = self.load_open_clip()
        self.model.to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.add_lora_to_clip(r, lora_alpha)

    def load_category_file(self, model_name):
        """Dynamically select category_file based on model name"""
        category_file_map = {
            'lung': 'clip_prompts/lung.json',
            'isic': 'clip_prompts/isic.json',
            'pascal': 'clip_prompts/pascal.json',
            'coco': 'clip_prompts/coco.json',
            'wbc': 'clip_prompts/wbc.json',
            'background': 'clip_prompts/background.json',
            'CHAOST2': 'clip_prompts/chaost2.json'
        }
        category_file = category_file_map.get(model_name, 'clip_prompts/default.json')

        with open(category_file, 'r') as f:
            category_data = json.load(f)

        self.category_descriptions = {cat['id']: cat.get('caption', cat['name']) for cat in category_data['categories']}

    def add_lora_to_clip(self, r, lora_alpha):
        """Adding LoRA to ViT only"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none"
        )

        if "ViT" in self.model_name:
            lora_config.target_modules = ["attn.out_proj"]

            self.model.visual = get_peft_model(self.model.visual, lora_config)
            self.model.transformer = get_peft_model(self.model.transformer, lora_config)

    def load_open_clip(self):
        """Loading the OpenCLIP Local Model"""
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.model_path,
        )

        tokenizer = open_clip.get_tokenizer(self.model_name)
        return model, preprocess, tokenizer

    def forward(self, category_ids: torch.Tensor, model: str, context_length: int = 77):
        """
        Args:
            images (torch.Tensor): [B, 3, H, W].
            category_ids (torch.Tensor): [B] tensor of category IDs.
            context_length (int, optional): Length of the context. Defaults to 77.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: image features and text features.
        """

        self.load_category_file(model)
        texts = [self.category_descriptions.get(cat_id.item(), 'No description available') for cat_id in category_ids]
        text_inputs = self.tokenizer(texts, context_length=context_length).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        # 适配不同架构的 feature 维度
        if "RN" in self.model_name:
            text_features = text_features[:, :512]

        return text_features



