import json
import sys
import os
from tqdm import tqdm
from PIL import Image
import torch

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

# === Setup ===
MODEL_DIR = "../internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_merge"  # Or Path To YOUR owned models
USE_HALF = True
IS_TEST = "-test" in sys.argv
IS_TEST_ALL = "-testall" in sys.argv

# === Output Directory ===
os.makedirs("../predict_data_chat", exist_ok=True)

# === Processor & Pipeline ===
pipe = pipeline(MODEL_DIR, backend_config=TurbomindEngineConfig(session_len=8192), device="cuda")

ALL_CATEGORIES = ["E1", "E2", "E3", "E4", "M1", "M2", "M3", "H1", "H2"]


def run_inference_for_category(category, max_samples=None):
    input_path = f"../test_data_chat/{category}.jsonl"
    output_path = f"../predict_data_chat/{category}.jsonl"

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        if max_samples:
            data = data[:max_samples]

    results = []

    for item in tqdm(data, desc=f"InternVL2 Inference {category}"):
        user_content = item["conversation"][0]["content"]

        # Loading Image
        image_paths = [c["image"] for c in user_content if c["type"] == "image"]
        images = [load_image(img_path) for img_path in image_paths]

        # Additional Prompt if YOU want
        prompt = ""
        for idx in range(len(images)):
            prompt += f"Image-{idx+1}: <image>\n"
        question = next(c["text"] for c in user_content if c["type"] == "text")
        prompt += question.strip()

        # Inference
        response = pipe((prompt, images))
        answer = response.text.strip()

        # Answer Extraction and Saving
        item["conversation"][1]["content"] = [{"type": "text", "text": answer}]
        results.append(item)

    # save as jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Saved to {output_path}")

# === other cmd choices ===
if IS_TEST_ALL:
    for cat in ALL_CATEGORIES:
        run_inference_for_category(cat, max_samples=10)
elif IS_TEST:
    CATEGORY = [arg for arg in sys.argv if arg.startswith("-") and arg[1:] in ALL_CATEGORIES][0].lstrip("-")
    run_inference_for_category(CATEGORY, max_samples=10)
else:
    arg = next((arg for arg in sys.argv[1:] if arg.startswith("-")), None)
    if arg:
        categories = arg.lstrip("-").split(",")
        for cat in categories:
            run_inference_for_category(cat)
    else:
        print("❌ Error: No category specified.")
