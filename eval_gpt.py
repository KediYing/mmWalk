import os
import json
import re
import openai
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# === Setup ===
GT_DIR = "../test_data_chat" 
PRED_DIR = "../eval_Qwen2_normal_0s" #Choose the inference result directory
ALL_CATEGORIES = ["E1", "E2", "E3", "E4", "M1", "M2", "M3", "H1", "H2"]
GPT_MODEL = "gpt-4o-mini"
NUM_WORKERS = min(16, cpu_count())  # 并发进程数量

openai.api_key = "YOUR_BEAUTIFUL_API_KEY"

# === extracters ===
def extract_answer(entry):
    return entry["conversation"][1]["content"][0]["text"].strip()

def extract_question(entry):
    user_content = entry["conversation"][0]["content"]
    for item in user_content:
        if item["type"] == "text" and "Question:" in item["text"]:
            match = re.search(r"Question:\s*(.*)", item["text"], flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return "[MISSING QUESTION]"

def extract_scenario(entry):
    user_content = entry["conversation"][0]["content"]
    for item in user_content:
        if item["type"] == "image" and "image" in item:
            path = item["image"]
            match = re.search(r"QAFrames/([^/]+)/", path)
            if match:
                folder_name = match.group(1)
                scenario = re.match(r"[A-Za-z]+", folder_name)
                if scenario:
                    return scenario.group(0)
    return "[UNKNOWN SCENARIO]"

# === GPT-4o-mini scoring ===
def score_sample(sample):
    category, question, gt_answer, pred_answer, scenario = sample
    messages = [
        {
            "role": "system",
            "content":
                "You are an intelligent evaluator designed to evaluate the correctness and similarity of generative outputs for question-answer pairs. "
                "Your task is to compare the model prediction answer with the correct answer and determine if they match in meaning. Here's the scoring criteria:\n\n"
                "### Scoring Criteria:\n"
                "5 = Perfect match or Correct in meaning\n"
                "4 = Key information correct, minor flaws\n"
                "3 = Partially correct\n"
                "2 = Mostly wrong answer for key query, but some relevance\n"
                "1 = Completely wrong or nonsense sentences\n\n"
                "Your response must ONLY be the integer score (e.g., 4). DO NOT include any text or explanation."
        },
        {
            "role": "user",
            "content":
                f"Question: {question}\n"
                f"Correct Answer: {gt_answer}\n"
                f"Predicted Answer: {pred_answer}\n\n"
                "Please provide a score from 1 to 5 based on how well the predicted answer matches the correct answer."
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0,
        )
        reply = response["choices"][0]["message"]["content"].strip()
        match = re.search(r"[1-5]", reply)
        score = int(match.group()) if match else 1
    except Exception as e:
        print(f"[!] GPT API error: {e}")
        score = 1

    return {
        "category": category,
        "scenario": scenario,
        "question": question,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "score": score
    }

# === evaluation ===
def main():
    all_results = []
    category_results = []

    output_file = open("gpt_scored_samples.jsonl", "w", encoding="utf-8")

    for cat in ALL_CATEGORIES:
        gt_path = os.path.join(GT_DIR, f"{cat}.jsonl")
        pred_file = next((f for f in os.listdir(PRED_DIR) if f.startswith(cat + "_")), None)
        if not pred_file:
            print(f"[!] Missing prediction for {cat}")
            continue
        pred_path = os.path.join(PRED_DIR, pred_file)

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = [json.loads(line) for line in f]
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = [json.loads(line) for line in f]

        samples = []
        for gt, pred in zip(gt_data, pred_data):
            question = extract_question(gt)
            gt_answer = extract_answer(gt)
            pred_answer = extract_answer(pred)
            scenario = extract_scenario(gt)
            samples.append((cat, question, gt_answer, pred_answer, scenario))

        print(f"\nScoring {cat} with multiprocessing ({NUM_WORKERS} workers)...")
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(pool.imap(score_sample, samples), total=len(samples)))

        for r in results:
            output_file.write(json.dumps(r, ensure_ascii=False) + "\n")

        cat_scores = [r["score"] for r in results]
        avg_cat_score = sum(cat_scores) / len(cat_scores)
        category_results.append({"category": cat, "avg_score": avg_cat_score})

        all_results.extend(results)

    output_file.close()

    #overall normalizing
    overall_avg = sum(r["score"] for r in all_results) / len(all_results)

    # === Category
    overall_std = (overall_avg - 1) / 4 * 100
    category_summary = []
    for item in category_results:
        std_score = (item["avg_score"] - 1) / 4 * 100
        category_summary.append({
            "category": item["category"],
            "avg_score": item["avg_score"],
            "standardized_score": std_score
        })

    print(f"\n===== Final Standardized Results (Category) =====")
    for r in category_summary:
        print(f"{r['category']}: Raw Avg {r['avg_score']:.2f} -> Standardized {r['standardized_score']:.2f}%")

    print(f"\nOverall Raw Score: {overall_avg:.2f}")
    print(f"Overall Standardized Score: {overall_std:.2f}%")

    # === Scenario
    scenario_scores = {}
    for r in all_results:
        scenario = r["scenario"]
        if scenario not in scenario_scores:
            scenario_scores[scenario] = []
        scenario_scores[scenario].append(r["score"])

    scenario_summary = []
    for scenario, scores in scenario_scores.items():
        avg_score = sum(scores) / len(scores)
        std_score = (avg_score - 1) / 4 * 100
        scenario_summary.append({
            "scenario": scenario,
            "avg_score": avg_score,
            "standardized_score": std_score
        })

    print(f"\n===== Final Standardized Results (Scenario) =====")
    for r in scenario_summary:
        print(f"{r['scenario']}: Raw Avg {r['avg_score']:.2f} -> Standardized {r['standardized_score']:.2f}%")

    # === save as json files
    with open("gpt_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "per_category": category_summary,
            "overall_score_raw": overall_avg,
            "overall_score_standardized": overall_std,
            "per_scenario": scenario_summary
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
