from tempfile import tempdir
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import json
import os
import tqdm
lang_map = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    'de': "German",
    'fr': "French",
    'it': "Italian",
    'th': "Thai",
    'ru': "Russian",
    'pt': "Portuguese",
    'es': "Spanish",
    'hi': "Hindi",
    'tr': "Turkish",
    'ar': "Arabic",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def generate(text, image_file):
    pixel_values = load_image(image_file, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=512, do_sample=True, temperature=0.9, top_p=0.9, num_beams=1)
    model.system_message = sp
    response = model.chat(tokenizer, pixel_values, text, generation_config)
    return response

def ocr_mt(image_folder, ref, lang, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results = {}
    ref = json.load(open(ref, "r", encoding="utf-8"))
    src_lang, tgt_lang = lang.split("2")
    text = text_temp.format(lang=lang_map[tgt_lang])
    for img, item in tqdm.tqdm(ref.items()):
        outputs = generate(text, image_folder+img)
        results[img] = {"mt": outputs, "ref": item[tgt_lang], "src": item[src_lang]} 

    json.dump(results, open(output_path + output_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


sp = "Please strictly follow the steps below to process the text in the image:\n1. **Comprehensive Recognition**: Extract all visible text elements in the image (including words, numbers, symbols, special characters)\n2. **Translatable text**: Accurate translation into target language, Special text such as parameters, symbols can be left as they are.\n3. **Format retention**:\n   - Maintain original text alignment\n   - Original line breaks and paragraph structure are preserved\n4. **Quality check**:\n   (1) Verify that all text blocks have been processed\n   (2) Verify terminology accuracy\n**Output Standardization**:\n1. prohibit inclusion of original text\n2. Prohibit the addition of explanatory notes\n3. Only output the translated text in the target language\n" #2
text_temp = "Please translate the text in the image into {lang}."

if __name__ == '__main__':
    path = 'OpenGVLab/InternVL2_5-8B'
    root = ""
    model_name=""

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    output_name = "sft_prompt2.json"
    #MIT10M
    image_folder = root+"MIT10M-refine/data/small/"

    src_lang = ["en", "zh", "ja", "de", "es", "fr", "it", "pt"]
    tgt_lang = ["zh", "en", "ko", "ja", "de", "es", "fr", "it", "pt", "ru", "th", "hi", "tr", "ar"]
    for sl in src_lang:
        for tl in tgt_lang:
            if sl == tl:
                continue
            al = f"{sl}2{tl}"
            img_source = root+f"MIT10M-refine/test/test_{sl}.json"
            output_path = f"evaluations/{model_name}/mit10/ocr_mt/{sl}/{al}/"
            if os.path.exists(output_path + output_name):
                continue
            print(output_path)
            ocr_mt(image_folder, img_source, al, output_path)

    #ocrmt
    image_folder = root+"OCRMT30K-refine/whole_image_v2/"
    img_source = root+"OCRMT30K-refine/original_data/original_test_1000.json"
    lang = "zh2en"
    output_path = f"evaluations/{model_name}/ocrmt/ocr_mt/{lang}/"
    print(output_path)
    ocr_mt(image_folder, img_source, lang, output_path)

    #anytrans
    lang_ref = {
        "en2zh": root+"AnyTrans-refine/en2zh_231.json",
        "zh2en": root+"AnyTrans-refine/zh2en_191.json",
        "ja2zh": root+"AnyTrans-refine/ja2zh_211.json",
        "ko2zh": root+"AnyTrans-refine/ko2zh_196.json",
        "zh2ja": root+"AnyTrans-refine/zh2ja_200.json",
        "zh2ko": root+"AnyTrans-refine/zh2ko_170.json",
    }
    for lang, ref in lang_ref.items():
        image_folder = root+ f"AnyTrans-refine/images/{lang}/"
        output_path = f"evaluations/{model_name}/anytrans/{lang}/ocr_mt/"
        print(output_path)
        ocr_mt(image_folder, ref, lang, output_path)