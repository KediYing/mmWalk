# mmWalk 파인튜닝 환경 설정 가이드

이 가이드는 WSL2 환경에서 mmWalk 데이터셋을 사용하여 InternVL2-8B 모델을 파인튜닝하는 전체 과정을 설명합니다.

## 📋 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [Git 저장소 클론](#1-git-저장소-클론)
3. [Python 가상환경 설정](#2-python-가상환경-설정)
4. [InternVL2 설치](#3-internvl2-설치)
5. [데이터셋 준비](#4-데이터셋-준비)
6. [사전학습 모델 다운로드](#5-사전학습-모델-다운로드)
7. [파인튜닝 설정](#6-파인튜닝-설정)
8. [파인튜닝 실행](#7-파인튜닝-실행)

---

## 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (최소 24GB VRAM 권장, RTX 3090/4090 또는 A100)
  - 2개의 GPU 사용 권장 (스크립트 기본 설정)
  - 1개 GPU만 사용 시 배치 사이즈 조정 필요
- **RAM**: 최소 32GB 이상 권장
- **디스크 공간**: 최소 100GB 이상 (데이터셋 + 모델 + 체크포인트)

### 소프트웨어
- **OS**: WSL2 (Ubuntu 20.04 또는 22.04)
- **Python**: 3.8 - 3.10 (3.10 권장)
- **CUDA**: 11.7 이상 (12.1 권장)
- **Git**: 최신 버전

---

## 1. Git 저장소 클론

### 1.1 작업 디렉토리 생성
```bash
# 홈 디렉토리로 이동
cd ~

# 작업 디렉토리 생성 (선택사항)
mkdir -p ~/projects
cd ~/projects
```

### 1.2 fork한 저장소 클론
```bash
# 본인의 GitHub username으로 변경
git clone https://github.com/YOUR_USERNAME/mmWalk.git
cd mmWalk
```

### 1.3 브랜치 확인
```bash
# 현재 브랜치 확인
git branch

# 원격 브랜치 확인
git branch -r
```

---

## 2. Python 가상환경 설정

### 2.1 Python 버전 확인
```bash
python3 --version
```

Python 3.8-3.10 사이 버전이 설치되어 있어야 합니다.

### 2.2 가상환경 생성
```bash
# 가상환경 생성
python3 -m venv venv_mmwalk

# 가상환경 활성화
source venv_mmwalk/bin/activate
```

가상환경이 활성화되면 터미널 프롬프트 앞에 `(venv_mmwalk)`가 표시됩니다.

### 2.3 pip 업그레이드
```bash
pip install --upgrade pip
```

---

## 3. InternVL2 설치

### 3.1 InternVL 저장소 클론
mmWalk 디렉토리와 같은 레벨에 InternVL을 클론합니다.

```bash
# 상위 디렉토리로 이동
cd ..

# InternVL 저장소 클론
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL

# InternVL2 브랜치로 전환 (필요시)
git checkout main
```

### 3.2 필수 패키지 설치
```bash
# PyTorch 설치 (CUDA 12.1 기준)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 버전이 다른 경우 https://pytorch.org/ 에서 적절한 명령어 확인

# Transformers 및 기타 의존성
pip install transformers==4.37.2
pip install sentencepiece
pip install timm==0.9.10
pip install einops
pip install shortuuid
pip install deepspeed
pip install lmdeploy
pip install torchvision
pip install Pillow

# Flash Attention 설치 (선택사항이지만 권장)
pip install flash-attn --no-build-isolation
```

### 3.3 InternVL 설치
```bash
cd InternVL/internvl_chat
pip install -e .
```

---

## 4. 데이터셋 준비

### 4.1 데이터셋 다운로드
mmWalk 데이터셋은 Harvard Dataverse에서 다운로드해야 합니다.

1. 웹 브라우저에서 다음 링크 접속:
   ```
   https://doi.org/10.7910/DVN/KKDXDK
   ```

2. `QAFrames.zip` 파일 다운로드

### 4.2 데이터셋 구조 설정
```bash
# mmWalk 디렉토리로 돌아가기
cd ~/projects/mmWalk

# pretrained 디렉토리 생성
mkdir -p pretrained/data

# 다운로드한 QAFrames.zip을 pretrained/data/로 이동
# WSL에서 Windows 다운로드 폴더는 /mnt/c/Users/YOUR_USERNAME/Downloads 에 위치
cp /mnt/c/Users/YOUR_USERNAME/Downloads/QAFrames.zip pretrained/data/

# 압축 해제
cd pretrained/data
unzip QAFrames.zip

# 압축 해제 확인
ls QAFrames/
# Busstop01, Busstop02 등의 디렉토리가 보여야 함
```

### 4.3 파인튜닝 어노테이션 데이터 준비
```bash
# mmWalk 루트로 돌아가기
cd ~/projects/mmWalk

# finetune_related 폴더의 어노테이션 압축 해제
cd finetune_related
unzip mmWalkQA_Annotation_for_Internvl2.zip

# 어노테이션 파일을 데이터셋 디렉토리로 복사
cp mmWalkQA_Annotation_for_Internvl2.jsonl ../pretrained/data/QAFrames/

# 확인
ls -lh ../pretrained/data/QAFrames/mmWalkQA_Annotation_for_Internvl2.jsonl
```

---

## 5. 사전학습 모델 다운로드

### 5.1 Hugging Face CLI 설치
```bash
pip install huggingface-hub
```

### 5.2 InternVL2-8B 모델 다운로드
```bash
# mmWalk 루트로 돌아가기
cd ~/projects/mmWalk

# pretrained 디렉토리에 모델 다운로드
huggingface-cli download \
  --resume-download \
  --local-dir pretrained/InternVL2-8B \
  --local-dir-use-symlinks False \
  OpenGVLab/InternVL2-8B
```

이 과정은 네트워크 속도에 따라 시간이 걸릴 수 있습니다 (수 GB).

### 5.3 다운로드 확인
```bash
ls -lh pretrained/InternVL2-8B/
# config.json, pytorch_model.bin 등의 파일이 있어야 함
```

---

## 6. 파인튜닝 설정

### 6.1 InternVL 파인튜닝 스크립트 통합

mmWalk의 파인튜닝을 위해서는 InternVL 저장소의 학습 스크립트와 mmWalk의 설정을 통합해야 합니다.

```bash
cd ~/projects/mmWalk

# InternVL의 학습 코드를 mmWalk에 심볼릭 링크 또는 복사
ln -s ~/projects/InternVL/internvl_chat internvl

# 또는 복사 (권장)
cp -r ~/projects/InternVL/internvl_chat/internvl .
```

### 6.2 DeepSpeed 설정 파일 생성

```bash
cd ~/projects/mmWalk
```

`zero_stage1_config.json` 파일을 생성합니다:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 1
  },
  "bf16": {
    "enabled": true
  }
}
```

### 6.3 메타데이터 경로 수정

`finetune_related/mmwalk.json` 파일이 올바른 경로를 가리키는지 확인:

```json
{
    "mmwalk": {
      "root": "./pretrained/data/QAFrames/",
      "annotation": "./pretrained/data/QAFrames/mmWalkQA_Annotation_for_Internvl2.jsonl",
      "data_augment": false,
      "repeat_time": 1,
      "length": 69390
    }
}
```

### 6.4 파인튜닝 스크립트 수정

`finetune_related/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh`를 mmWalk 루트로 복사하고 경로를 수정합니다.

---

## 7. 파인튜닝 실행

### 7.1 환경 변수 확인
```bash
# CUDA 사용 가능 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 7.2 GPU 개수에 따른 설정 조정

#### 2개 GPU 사용 (기본)
```bash
export GPUS=2
export BATCH_SIZE=16
export PER_DEVICE_BATCH_SIZE=4
```

#### 1개 GPU 사용
```bash
export GPUS=1
export BATCH_SIZE=8
export PER_DEVICE_BATCH_SIZE=4
```

### 7.3 파인튜닝 실행
```bash
cd ~/projects/mmWalk

# 파인튜닝 스크립트 실행
bash finetune_mmwalk.sh
```

### 7.4 학습 모니터링

다른 터미널 창에서:
```bash
# 학습 로그 실시간 확인
tail -f work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora/training_log.txt

# TensorBoard 실행 (선택사항)
tensorboard --logdir work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora/
```

---

## 🔍 예상 소요 시간

- **환경 설정**: 1-2시간
- **데이터셋 다운로드**: 30분 - 1시간 (네트워크 속도 의존)
- **모델 다운로드**: 30분 - 1시간 (네트워크 속도 의존)
- **파인튜닝**: 12-24시간 (GPU 성능 및 개수에 따라 다름)

---

## ⚠️ 주의사항

1. **VRAM 부족 시**: `PER_DEVICE_BATCH_SIZE`를 줄이세요 (예: 4 → 2 → 1)
2. **OOM 에러**: gradient checkpointing이 활성화되어 있는지 확인 (`--grad_checkpoint True`)
3. **경로 오류**: 모든 경로가 올바른지 확인 (특히 `pretrained/InternVL2-8B` 경로)
4. **DeepSpeed 오류**: `zero_stage1_config.json` 파일이 mmWalk 루트에 있는지 확인

---

## 📊 학습 완료 후

학습이 완료되면 다음 위치에 모델 체크포인트가 저장됩니다:
```
work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora/
```

이 모델을 사용하여 추론을 실행할 수 있습니다:
```bash
python inference_lmdeploy.py -testall
```

---

## 🐛 문제 해결

### Q: CUDA out of memory
- 배치 사이즈를 줄이세요
- GPU 개수를 늘리세요
- `max_dynamic_patch`를 줄이세요 (6 → 4)

### Q: ModuleNotFoundError
- InternVL이 올바르게 설치되었는지 확인
- `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` 실행

### Q: 데이터셋을 찾을 수 없음
- `pretrained/data/QAFrames/` 경로 확인
- 어노테이션 파일 경로 확인

---

## 📚 추가 자료

- [InternVL 공식 문서](https://github.com/OpenGVLab/InternVL)
- [DeepSpeed 문서](https://www.deepspeed.ai/)
- [mmWalk 논문](https://arxiv.org/abs/2510.11520)
