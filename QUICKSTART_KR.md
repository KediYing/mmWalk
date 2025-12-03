# mmWalk 파인튜닝 빠른 시작 가이드

처음부터 끝까지 단계별로 따라하는 간단한 가이드입니다.

## 🚀 5단계로 시작하기

### 1단계: 저장소 클론
```bash
# WSL 터미널에서 실행
cd ~
git clone https://github.com/YOUR_USERNAME/mmWalk.git
cd mmWalk
```

### 2단계: 자동 환경 설정 (권장)
```bash
# 자동 설정 스크립트 실행
bash setup_environment.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- ✅ Python 버전 확인
- ✅ 가상환경 생성
- ✅ 필수 패키지 설치
- ✅ InternVL 저장소 클론
- ✅ 디렉토리 구조 생성

**소요 시간**: 약 30-60분 (네트워크 속도에 따라 다름)

### 3단계: 데이터셋 다운로드 및 설정
```bash
# 1. 웹 브라우저에서 다운로드
# https://doi.org/10.7910/DVN/KKDXDK
# QAFrames.zip 다운로드

# 2. WSL로 파일 이동 (Windows 다운로드 폴더에서)
cp /mnt/c/Users/YOUR_USERNAME/Downloads/QAFrames.zip pretrained/data/

# 3. 압축 해제
cd pretrained/data
unzip QAFrames.zip
cd ../..

# 4. 어노테이션 파일 설정
cd finetune_related
unzip mmWalkQA_Annotation_for_Internvl2.zip
cp mmWalkQA_Annotation_for_Internvl2.jsonl ../pretrained/data/QAFrames/
cd ..
```

**소요 시간**: 약 10-30분

### 4단계: 사전학습 모델 다운로드
```bash
# 가상환경 활성화 (아직 활성화하지 않았다면)
source venv_mmwalk/bin/activate

# InternVL2-8B 모델 다운로드
huggingface-cli download \
  --resume-download \
  --local-dir pretrained/InternVL2-8B \
  --local-dir-use-symlinks False \
  OpenGVLab/InternVL2-8B
```

**소요 시간**: 약 30-60분 (네트워크 속도에 따라 다름)

### 5단계: 파인튜닝 실행! 🎉
```bash
# CUDA 사용 가능 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 파인튜닝 시작
bash finetune_mmwalk.sh
```

**소요 시간**: 약 12-24시간 (GPU 성능에 따라 다름)

---

## 📊 학습 모니터링

### 실시간 로그 확인
```bash
# 다른 터미널 창에서
tail -f work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora/training_log.txt
```

### TensorBoard 실행
```bash
# 가상환경 활성화 후
source venv_mmwalk/bin/activate

# TensorBoard 실행
tensorboard --logdir work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora/

# 브라우저에서 http://localhost:6006 접속
```

---

## ⚙️ GPU 설정 조정

### 1개 GPU만 사용하는 경우
```bash
export GPUS=1
export BATCH_SIZE=8
export PER_DEVICE_BATCH_SIZE=4

bash finetune_mmwalk.sh
```

### VRAM이 부족한 경우 (24GB 미만)
```bash
export GPUS=1
export BATCH_SIZE=4
export PER_DEVICE_BATCH_SIZE=2

bash finetune_mmwalk.sh
```

### 더 작은 배치 사이즈 (16GB VRAM)
```bash
export GPUS=1
export BATCH_SIZE=2
export PER_DEVICE_BATCH_SIZE=1

bash finetune_mmwalk.sh
```

---

## 🎯 체크리스트

파인튜닝 시작 전 다음을 확인하세요:

- [ ] Python 3.8-3.10 설치됨
- [ ] NVIDIA GPU 및 CUDA 설치됨
- [ ] 가상환경 생성 및 활성화됨
- [ ] 필수 패키지 설치됨 (PyTorch, Transformers 등)
- [ ] InternVL 저장소 클론됨
- [ ] QAFrames 데이터셋 다운로드 및 압축 해제됨
- [ ] 어노테이션 파일이 올바른 위치에 있음
- [ ] InternVL2-8B 모델 다운로드됨
- [ ] `zero_stage1_config.json` 파일 존재
- [ ] 충분한 디스크 공간 (최소 100GB)

---

## 🐛 자주 발생하는 문제

### "CUDA out of memory"
```bash
# 배치 사이즈 줄이기
export PER_DEVICE_BATCH_SIZE=2  # 또는 1
bash finetune_mmwalk.sh
```

### "No module named 'internvl'"
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 또는 InternVL 다시 복사
cp -r ../InternVL/internvl_chat/internvl .
```

### "FileNotFoundError: pretrained/InternVL2-8B"
```bash
# 모델 경로 확인
ls -l pretrained/InternVL2-8B/

# 모델 다시 다운로드
huggingface-cli download --resume-download --local-dir pretrained/InternVL2-8B --local-dir-use-symlinks False OpenGVLab/InternVL2-8B
```

### "Cannot find annotation file"
```bash
# 어노테이션 파일 위치 확인
ls -l pretrained/data/QAFrames/mmWalkQA_Annotation_for_Internvl2.jsonl

# 파일이 없다면 다시 복사
cd finetune_related
cp mmWalkQA_Annotation_for_Internvl2.jsonl ../pretrained/data/QAFrames/
cd ..
```

---

## 📈 예상 일정

| 단계 | 작업 | 소요 시간 |
|------|------|----------|
| 1 | 저장소 클론 | 1분 |
| 2 | 환경 설정 | 30-60분 |
| 3 | 데이터셋 준비 | 10-30분 |
| 4 | 모델 다운로드 | 30-60분 |
| 5 | 파인튜닝 | 12-24시간 |
| **총합** | | **약 14-26시간** |

---

## 💡 팁

1. **백그라운드 실행**: 학습을 백그라운드에서 실행하려면
   ```bash
   nohup bash finetune_mmwalk.sh > training.log 2>&1 &
   ```

2. **tmux 사용**: 세션이 끊어져도 학습이 계속되도록
   ```bash
   tmux new -s mmwalk
   bash finetune_mmwalk.sh
   # Ctrl+B, D로 detach
   # 나중에 tmux attach -t mmwalk로 재접속
   ```

3. **체크포인트 확인**: 학습 중 저장되는 체크포인트 확인
   ```bash
   ls -lh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora/
   ```

---

## 🎓 다음 단계

학습 완료 후:
1. 모델 추론 테스트
2. GPT 평가 실행
3. 결과 분석

자세한 내용은 `SETUP_GUIDE_KR.md`를 참조하세요!
