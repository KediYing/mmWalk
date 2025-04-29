# mmWalk: Towards Inclusive Multimodal Walking Embodied AI
Code Submission

This repository contents the completed QA data Annotation classified according to QA Type, the model inference code, the model evaluation code, and some of the inference results with evaluation outcomes.

test_data_chat: Folder contains 9 JSONL Annotation, for all QA Types in 9 categories in test split.
eval_MODELNAME_INPUTSETTING_SHOTSETTING: Folder contains inference results and gpt evaluated summary of each deployed model in 9 QA categories.
MODELNAME: used model method, for example Janus-Pro-7B,Qwen2VL-7B-Instruct...
INPUTSETTING: normal = full multiview, single = walker view only, dog+ =dog view + walker view, drone+ = drone view + walker view
SHOTSETTING: 0s = Zero-Shot, 3s = 3-Shot, finetuned

Inference Codes:
inference_lmdeploy.py:  Model deployment using lmdeploy package, including deployment of our finetuned  mmWalkQA_finetuned_internvl2_8b_internlm2_7b_dynamic_res_2nd_ merge, which can be accessed and downloaded at ""

python inference_lmdeploy.py -E1: deploy model for inference category E1, output folder and JSONL file format should be like any in eval_MODELNAME_INPUTSETTING_SHOTSETTING folder
-testall: run inference for 10 QA pairs in each category

inference_transformers.py: Model deployment using transformer package, usage is the same as inference_lmdeploy.py

Check Comments in Code Script for more Information Please

eval_gpt.py: Evaluater, which generate gpt_scored_samples(contains score for every single qa pair) and gpt_evaluation_summary(contains average score* across qa types and across scenarios)
*Average Score is calculated with .2f, which could lead to minor errors caused by omitting two decimal places, minor errors should not exceed ±1 in normalized score.

python eval_gpt.py can help YOU get the result directly in Terminal/CMD lines for overview.
