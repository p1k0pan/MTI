import argparse
import torch
import sys

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import os
import requests
import re
import json
import tqdm
from tqdm.contrib import tzip
from pathlib import Path
import random


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
def process_query(qs, image_file, sp=None):
    if sp is not None:
        messages = [
            {"role": "system", "content": sp},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]
    else:
        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": qs},
            ],
        }
    ]
    # Preparation for inference
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return prompt, image_inputs, video_inputs

def generate(text, image_file, sp):
    qs = text
    prompt, image_inputs, video_inputs = process_query(qs, image_file, sp)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        # images=None,
        # videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    return output_text

def pp_ocr_mt(image_folder, ref, lang, ppocr_data, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results = {}
    ref = json.load(open(ref, "r", encoding="utf-8"))
    ppocr_data = json.load(open(ppocr_data, "r", encoding="utf-8"))
    src_lang, tgt_lang = lang.split("2")


    for img, item in tqdm.tqdm(ref.items()):
        p_data = ppocr_data[img]
        objs = '\n'.join(p_data["output"])

        image_path = image_folder+img
        sp = sp_temp.format(ocr_text=objs)
        text = text_temp.format(lang=lang_map[tgt_lang])
        outputs = generate(text, image_path, sp )
        results[img] = {"mt": outputs, "ref": item[tgt_lang], "src": item[src_lang], "pp_ocr": objs} 

    json.dump(results, open(output_path + output_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def pp_ocr_mt_100(image_folder, ref, lang,output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results = {}
    ref = json.load(open(ref, "r", encoding="utf-8"))
    src_lang, tgt_lang = lang.split("2")


    for img, item in tqdm.tqdm(ref.items()):
       
        objs = '\n'.join(item["ocr_result"])

        image_path = image_folder+img
        sp = sp_temp.format(ocr_text=objs)
        text = text_temp.format(lang=lang_map[tgt_lang])
        outputs = generate(text, image_path, sp )
        results[img] = {"mt": outputs, "src": item[src_lang], "pp_ocr": objs} 

    json.dump(results, open(output_path + output_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # parser.add_argument("--source_file", type=str, required=True)
    # parser.add_argument("--target_file", type=str, required=True)
    # parser.add_argument("--image_source", type=str, required=True)
    # parser.add_argument("--image_folder", type=str, required=True)
    # parser.add_argument("--prompt_temp", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )


    # default processer
    min_pixels = 1280 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_path,  min_pixels=min_pixels, max_pixels=max_pixels)

    root = ""
    output_folder=""
    args.model_path = ""
   

    """sft"""
    sp_temp = """Please strictly follow the steps below to process the text in the image:
1. **Comprehensive Recognition**: Extract all visible text elements in the image (including words, numbers, symbols, special characters)
2. **Translatable text**: Accurate translation into target language, Special text such as parameters, symbols can be left as they are.
3. **Format retention**:
   - Maintain original text alignment
   - Original line breaks and paragraph structure are preserved
4. **Quality check**:
   (1) Verify that all text blocks have been processed
   (2) Verify terminology accuracy
**Output Standardization**:
1. prohibit inclusion of original text
2. Prohibit the addition of explanatory notes
3. Only output the translated text in the target language
---
[OCR_TEXT_FOR_MODEL_REFERENCE]
{ocr_text}

(Please do not include the above original text in the final output, just the translation!)
"""#2 ocr
    text_temp = "Please translate the text in the image into {lang}."

    output_name = "sft_prompt2.json"

    # #MIT10M
    # image_folder = root+"MIT10M-refine/data/small/"

    # src_lang = ["en", "zh", "ja", "de", "es", "fr", "it", "pt"]
    # tgt_lang = ["zh", "en", "ko", "ja", "de", "es", "fr", "it", "pt", "ru", "th", "hi", "tr", "ar"]
    # for sl in src_lang:
    #     for tl in tgt_lang:
    #         if sl == tl:
    #             continue
    #         al = f"{sl}2{tl}"
    #         img_source = root+f"MIT10M-refine/test/test_{sl}.json"
    #         output_path = f"evaluations/{output_folder}/mit10/ppocr_vl_mt/{sl}/{al}/"
            
    #         if os.path.exists(output_path + output_name):
    #             continue
    #         print(output_path + output_name)
    #         ppocr_data = root+f"MIT10M-refine/ppocr/ppocr_mit10_{sl}.json"
    #         pp_ocr_mt(image_folder, img_source, al, ppocr_data, output_path)

    # #ocrmt
    # image_folder = root+"OCRMT30K-refine/whole_image_v2/"
    # img_source = root+"OCRMT30K-refine/original_data/original_test_1000.json"
    # lang = "zh2en"
    # output_path = f"evaluations/{output_folder}/ocrmt/ppocr_vl_mt/{lang}/"
    # print(output_path)
    # ppocr_data = root+"OCRMT30K-refine/ppocr_ocrmt.json"
    # pp_ocr_mt(image_folder, img_source, lang, ppocr_data, output_path)

    # #anytrans
    # lang_ref = {
    #     "en2zh": root+"AnyTrans-refine/en2zh_231.json",
    #     "zh2en": root+"AnyTrans-refine/zh2en_191.json",
    #     "ja2zh": root+"AnyTrans-refine/ja2zh_211.json",
    #     "ko2zh": root+"AnyTrans-refine/ko2zh_196.json",
    #     "zh2ja": root+"AnyTrans-refine/zh2ja_200.json",
    #     "zh2ko": root+"AnyTrans-refine/zh2ko_170.json",
    # }
    # for lang, ref in lang_ref.items():
    #     image_folder = root+ f"AnyTrans-refine/images/{lang}/"
    #     output_path = f"evaluations/{output_folder}/anytrans/{lang}/ppocr_vl_mt/"
    #     print(output_path)
    #     ppocr_data = root+f"AnyTrans-refine/ppocr_{lang}.json"
    #     pp_ocr_mt(image_folder, ref, lang, ppocr_data, output_path)

    # dataset100
    langs = ["zh2de", "zh2ar", "zh2hi", "zh2ja", "zh2ru", "zh2es"]
    image_folder = root+ "dataset100/test_images/"
    test_folder = Path(root+"dataset100/test_100")
    for lang in langs:

        for test_file in test_folder.rglob("*.json"):
            output_path = f"evaluations/{output_folder}/dataset100/ppocr_vl_mt/{lang}/{test_file.stem}/"
            if os.path.exists(output_path+output_name):
                continue
            else:
                Path(output_path).mkdir(parents=True, exist_ok=True)
            print(output_path)
            pp_ocr_mt_100(image_folder, test_file, lang, output_path)
