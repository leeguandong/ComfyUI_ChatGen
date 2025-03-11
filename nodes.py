import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

sys_singlemodal = """
You are a user requirements translation expert. I have a freestyle prompt written by a non professional user for text-to-image tasks. Please convert the content of this freestyle prompt into professional prompt and professional negativePrompt, and provide the model and its parameters that are most suitable for the user's text-to-image task.
Here is the content I need you to convert:
"""

sys_multimodal = """
You are a user requirements translation expert. I have a freestyle prompt written by a non professional user for text-to-image tasks.
Additionally, a general user provide several reference images, indicating that they want the final generated image to have a style similar to those images. You should combine the reference images to convert the content of the freestyle prompt into professional prompt and professional negativePrompt, and provide the model and its parameters that are most suitable for the user's text-to-image task.
Here are the reference images and content I need you to convert:
"""


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


class ChatGenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["ChengyouJia/ChatGen-Base-2B", "ChengyouJia/ChatGen-Base-4B", "ChengyouJia/ChatGen-Base-8B",
                     "ChengyouJia/ChatGen-Evo-8B"], {"default": "ChengyouJia/ChatGen-Base-8B"}),
                "input_size": ("INT", {"default": 448, "min": 224, "max": 1024, "step": 32}),
                "max_num": ("INT", {"default": 12, "min": 1, "max": 12}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            }, "optional": {
                "local_model_path": ("STRING", {"default": "ChengyouJia/ChatGen-Base-8B"}),
            }
        }

    RETURN_TYPES = ("MODEL", "TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "chatgen"

    def load_model(self, model, input_size, max_num, load_local_model, *args, **kwargs):
        _DTYPE = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_local_model:
            model_path = kwargs.get("local_model_path", "ChengyouJia/ChatGen-Base-8B")
        else:
            model_path = "ChengyouJia/ChatGen-Base-8B"

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=_DTYPE,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        return (model, tokenizer)


class ChatGenImageProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_size": ("INT", {"default": 448, "min": 224, "max": 1024, "step": 32}),
                "max_num": ("INT", {"default": 6, "min": 1, "max": 12}),
                "use_thumbnail": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PIXEL_VALUES",)
    RETURN_NAMES = ("pixel_values",)
    FUNCTION = "process_image"
    CATEGORY = "chatgen"

    def dynamic_preprocess(self, image, min_num, max_num, image_size, use_thumbnail):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
            for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def process_image(self, image, input_size, max_num, use_thumbnail):
        image = tensor2pil(image)
        transform = build_transform(input_size=input_size)
        processed_images = self.dynamic_preprocess(
            image, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=use_thumbnail
        )
        pixel_values = [transform(img) for img in processed_images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        return (pixel_values,)


class ChatGenGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 2048}),
                "do_sample": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "pixel_values": ("PIXEL_VALUES",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "chatgen"

    def generate(self, model, tokenizer, prompt, max_new_tokens, do_sample, pixel_values=None):
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            sys_singlemodal + prompt,
            generation_config,
            history=None,
            return_history=True
        )
        return (response,)


NODE_CLASS_MAPPINGS = {
    "ChatGenModelLoader": ChatGenModelLoader,
    "ChatGenImageProcessor": ChatGenImageProcessor,
    "ChatGenGenerate": ChatGenGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGenModelLoader": "ChatGen Model Loader",
    "ChatGenImageProcessor": "ChatGen Image Processor",
    "ChatGenGenerate": "ChatGen Generate"
}
