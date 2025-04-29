import json
import sys
import os
import random
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoProcessor, AutoModelForCausalLM,AutoModelForImageTextToText

# === Setting ===
MODEL_NAME = "A_Beautiful_Model_You_Choose"
USE_HALF = True
IS_TEST = "-test" in sys.argv
IS_TEST_ALL = "-testall" in sys.argv

# === Custom Output Directory ===
os.makedirs("../predict_test_chat", exist_ok=True)

# === Model Loading ===
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if USE_HALF and torch.cuda.is_available() else torch.float32,
    device_map="cuda:0",
    low_cpu_mem_usage=True
)

ALL_CATEGORIES = ["E1", "E2", "E3", "E4", "M1", "M2", "M3", "H1", "H2"]

#this method is used to make output file evaluatable, YOU can run -testall first to take a look at YOUR model output
#Only the Model's Answer should be saved into output jsonl files
#Some model may duplicate given input and prompt
'''
def extract_llava_next_answer(text):
    if '[/INST]' in text:
        return text.split('[/INST]')[-1].strip()
    if 'assistant' in text:
        return text.split('assistant')[-1].strip()
    return text.strip()
'''

def run_inference_for_category(category, max_samples=None):
    input_path = f"../test_data_chat/{category}.jsonl"
    output_path = f"../predict_test_chat/{category}_{MODEL_NAME.split('/')[-1]}.jsonl"

    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]
        if max_samples:
            data = data[:max_samples]

    results = []
    #YOU might want to change the terminal track name in Line 53
    for example in tqdm(data, desc=f"LLaVA Next Inference {category}"):
        user_content = example["conversation"][0]["content"]
        question = next(c["text"] for c in user_content if c["type"] == "text")
        images = [Image.open(c["image"]).convert("RGB") for c in user_content if c["type"] == "image"]

        current_msg = {
            "role": "user",
            "content": [*[{"type": "image", "image": img} for img in images], {"type": "text", "text": question}]
        }

        messages = [current_msg]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(text=chat, images=images, return_tensors="pt").to(model.device, torch.float16 if USE_HALF else torch.float32)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
            decoded = processor.decode(output[0], skip_special_tokens=True)
            answer = decoded
            # If you need your answer extractor, use the codeline here
            #answer = extract_llava_answer(decoded)

        example["conversation"][1]["content"] = [{"type": "text", "text": answer}]
        results.append(example)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Inference complete for {category}. Output saved to {output_path}")

# === cmd instructions ===
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
