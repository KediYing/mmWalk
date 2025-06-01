# mmWalk: Towards Multi-modal Multi-view Walking Assistance

## Code Submission

This repository contains:

- ✅ Fully annotated QA data, classified by QA type  
- ✅ Inference code for multiple models  
- ✅ Evaluation code using GPT  
- ✅ Sample inference and evaluation results
- ✅ Finetune scripts and finetune required json files


---

## Directory Structure

Dataset can be downloaded from https://doi.org/10.7910/DVN/KKDXDK. 

> Before running any code, **make sure you have downloaded `QAFrames.zip`** and extracted it into your `Workspace/` directory.

### After extracting the dataset:

```
---Workspace
   |--QAFrames
   |   |--Busstop01
   |   |--...
   |
   |--test_data_chat
   |   |--E1.jsonl
   |   |--...
   |   |--H2.jsonl
   |
   |--inference_lmdeploy.py
   |--inference_transformer.py
   |--eval_gpt.py
```

### After running inference and evaluation:

```
---Workspace
   |--QAFrames
   |   |--Busstop01
   |   |--...
   |
   |--test_data_chat
   |   |--E1.jsonl
   |   |--...
   |   |--H2.jsonl
   |
   |--inference_lmdeploy.py
   |--inference_transformer.py
   |--eval_gpt.py
   |
   |--INFERENCE_OUTPUT_FOLDER
   |   |--E1.jsonl
   |   |--...
   |   |--H2.jsonl
   |
   |--gpt_scored_samples.json
   |--gpt_evaluation_summary.json
```

---

##  Dataset Details

### `test_data_chat/`

This folder contains 9 `.jsonl` files, each corresponding to a QA type category in the test split.

---

## Evaluation Output Format

### `eval_MODELNAME_INPUTSETTING_SHOTSETTING/`

Each folder contains:
- Model inference results
- GPT-evaluated scores and summaries for 9 QA categories

**Naming convention:**

- `MODELNAME`: model used, e.g., `Janus-Pro-7B`, `Qwen2VL-7B-Instruct`
- `INPUTSETTING`:
  - `normal`: full multiview input
  - `single`: walker view only
  - `dog+`: dog + walker views
  - `drone+`: drone + walker views
- `SHOTSETTING`:
  - `0s`: zero-shot
  - `3s`: 3-shot
  - `finetuned`: finetuned model

**Example:**

`eval_Qwen2_normal_3s` means:
> Results from Qwen2-7B-Instruct using full multiview input and 3-shot setting.

For more details, please refer to our paper.

---

##  Inference Scripts

### `inference_lmdeploy.py`

Model deployment using `lmdeploy`, including our **finetuned model**:

**Model Name:**
```
mmWalkQA_finetuned_internvl2_8b_internlm2_7b_dynamic_res_2nd_merge
```

Google Drive Download link: *(To be added)*

**Usage:**

```bash
# Run inference on E1
python inference_lmdeploy.py -E1

# Run inference on 10 samples per category
python inference_lmdeploy.py -testall
```

---

### `inference_transformer.py`

Deployment via HuggingFace `transformers`.  
Usage is identical to `inference_lmdeploy.py`.

Check code comments for details.

---

## Evaluation Script

### `eval_gpt.py`

This script runs GPT-based evaluation and produces:

- `gpt_scored_samples.json`: score for each QA pair
- `gpt_evaluation_summary.json`: average score per QA type and scenario

**Note:**
- Average score is formatted with `.2f`, which may introduce ±1 rounding errors in normalized scoring.

**Usage:**

```bash
python eval_gpt.py
```

The evaluation summary will be printed to the terminal.
---

### `finetune_related/`

This folder contains a InternVL2-8B-InternLM2.5-7B finetune script, along with the finetune required dataset metadata json and train split annotation in InternVL2 format. To run the finetune phase, YOU should follow the instruction on InternVL Official Website for finetuning InternVL2-8B by replacing the requied files and scripts in finetune_related folder.

---


## Citation

> Coming soon — please refer to our paper for citation details.

---

