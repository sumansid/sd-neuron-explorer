"""Image statistics and a CLIP "attribute lens" for steered images.

Pixel metrics (on [B,3,H,W] tensors in [0,1]):
    sat          mean HSV saturation
    colorful     Hasler & Suesstrunk colorfulness
    lum          mean luminance (Rec. 601)
    contrast     std of luminance
The CLIP lens scores every image against a vocabulary of visual attributes; a steered image's shift against its
unsteered baseline, averaged over prompts and compared with random-direction controls, is what the neuron writes.
"""
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

CLIP_ID = "openai/clip-vit-base-patch32"

ATTRIBUTES = [
    # tone / colour
    "a black and white photo", "a grayscale image", "a monochrome image", "a sepia toned photo",
    "a very colorful saturated image", "a pastel colored image", "a very dark image", "a very bright overexposed image",
    "a high contrast image", "a low contrast washed out image", "a red tinted image", "a blue tinted image",
    "a green tinted image", "a yellow tinted image", "a purple tinted image", "an orange tinted image",
    "neon lights", "golden hour lighting", "a photo at night", "a foggy misty scene", "a snowy scene",
    "a rainy scene", "fire and flames", "underwater", "a sunset",
    # medium / style
    "a photograph", "an oil painting", "a watercolor painting", "a pencil sketch", "a cartoon", "anime style art",
    "pixel art", "a 3d render", "a comic book illustration", "a vintage photograph", "a polaroid photo",
    "a blurry out of focus image", "a sharp detailed image", "film grain", "a glitchy distorted image",
    "an abstract pattern", "a collage", "a poster with text", "a screenshot of a website",
    # content
    "a close up macro photo", "a wide landscape shot", "an aerial view", "a portrait of a person",
    "a crowd of many people", "an animal", "a cat", "a dog", "a bird", "flowers", "trees and forest",
    "mountains", "the ocean", "a city skyline", "a building interior", "a car", "food on a plate",
    "a robot", "a skull", "a baby", "an old person", "a child", "a woman", "a man", "a hand",
    "many eyes", "repeated tiling pattern", "a mirror reflection", "a symmetrical image",
    "text and letters", "a map", "a diagram", "a cube", "a sphere", "stripes", "polka dots",
    "a wedding", "a war scene", "a horror scene", "a cute kawaii image", "a luxury product photo",
    "an empty scene with nothing in it", "noise",
    # specific landmarks / objects (golden-gate style probes)
    "the Golden Gate Bridge", "a suspension bridge", "the Eiffel Tower", "a pyramid", "a clock", "a cup of coffee",
]


def pixel_metrics(img):
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    mx, mn = img.max(1).values, img.min(1).values
    sat = ((mx - mn) / (mx + 1e-6)).mean((1, 2))
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = torch.sqrt(rg.std((1, 2)) ** 2 + yb.std((1, 2)) ** 2) + 0.3 * torch.sqrt(rg.mean((1, 2)) ** 2 + yb.mean((1, 2)) ** 2)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return {"sat": sat, "colorful": colorful, "lum": lum.mean((1, 2)), "contrast": lum.std((1, 2))}


class Lens:
    def __init__(self, device="cuda", dtype=torch.float16, attributes=ATTRIBUTES):
        self.device, self.dtype = device, dtype
        self.model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=dtype).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(CLIP_ID)
        self.attributes = list(attributes)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.text = self.embed_text(self.attributes)

    @torch.no_grad()
    def embed_text(self, texts):
        t = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
        pooled = self.model.text_model(**t).pooler_output
        return F.normalize(self.model.text_projection(pooled).float(), dim=-1)

    @torch.no_grad()
    def embed_images(self, img):
        x = F.interpolate(img, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        pooled = self.model.vision_model(pixel_values=x).pooler_output
        return F.normalize(self.model.visual_projection(pooled).float(), dim=-1)

    def attribute_scores(self, img_emb):
        return 100.0 * img_emb @ self.text.T  # [B, n_attr] CLIP logits
