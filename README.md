# mmWalk: Towards Multi-modal Multi-view Walking Assistance
Accepted by NeurIPS 2025 D&B Track

## Code Submission

This repository contains:

- ✅ Fully annotated QA data, classified by QA type
- ✅ Inference code for multiple models
- ✅ Evaluation code using GPT
- ✅ Sample inference and evaluation results
- ✅ Finetune scripts and finetune required json files
- ✅ **NEW**: Complete setup guides and automated scripts (Korean/English)

## 🚀 Quick Start

### For Korean Users (한국어 사용자)
빠른 시작을 원하시면 **[QUICKSTART_KR.md](QUICKSTART_KR.md)**를 참조하세요!

자세한 설정 가이드는 **[SETUP_GUIDE_KR.md](SETUP_GUIDE_KR.md)**를 확인하세요.

### Automated Setup
```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/mmWalk.git
cd mmWalk

# Run automated setup
bash setup_environment.sh

# Verify environment
python check_environment.py

# Start finetuning
bash finetune_mmwalk.sh
```

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

## Finetuning

### Quick Finetuning Guide

This repository now includes complete finetuning setup scripts:

1. **Automated Setup**: `bash setup_environment.sh`
2. **Environment Check**: `python check_environment.py`
3. **Start Training**: `bash finetune_mmwalk.sh`

### Configuration Files

- `finetune_mmwalk.sh`: Main finetuning script
- `zero_stage1_config.json`: DeepSpeed configuration
- `finetune_related/mmwalk.json`: Dataset metadata
- `finetune_related/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh`: Original script

### Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (2 GPUs recommended)
- **Python**: 3.8-3.10
- **CUDA**: 11.7+
- **Disk Space**: 100GB+

### Detailed Documentation

- **[QUICKSTART_KR.md](QUICKSTART_KR.md)**: Quick start guide (Korean)
- **[SETUP_GUIDE_KR.md](SETUP_GUIDE_KR.md)**: Complete setup guide (Korean)

### `finetune_related/`

This folder contains the original InternVL2-8B-InternLM2.5-7B finetune script, along with the finetune required dataset metadata json and train split annotation in InternVL2 format.

---


## Citation

```
@article{ying2025mmwalk,
  title={mmWalk: Towards Multi-modal Multi-view Walking Assistance},
  author={Ying, Kedi and Liu, Ruiping and Chen, Chongyan and Tao, Mingzhe and Shi, Hao and Yang, Kailun and Zhang, Jiaming and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2510.11520},
  year={2025}
}
```

---

