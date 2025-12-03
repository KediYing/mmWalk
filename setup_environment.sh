#!/bin/bash

# mmWalk 파인튜닝 환경 자동 설정 스크립트
# WSL2 Ubuntu 환경 기준

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "mmWalk 파인튜닝 환경 설정 시작"
echo "=========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Python 버전 확인
echo -e "\n${YELLOW}[1/7] Python 버전 확인...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "현재 Python 버전: $PYTHON_VERSION"

if python3 -c "import sys; exit(0 if (3,8) <= sys.version_info < (3,11) else 1)"; then
    echo -e "${GREEN}✓ Python 버전 적합${NC}"
else
    echo -e "${RED}✗ Python 3.8-3.10 버전이 필요합니다${NC}"
    exit 1
fi

# 2. 가상환경 생성
echo -e "\n${YELLOW}[2/7] Python 가상환경 생성...${NC}"
if [ -d "venv_mmwalk" ]; then
    echo "기존 가상환경이 존재합니다."
    read -p "삭제하고 새로 만드시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv_mmwalk
        python3 -m venv venv_mmwalk
        echo -e "${GREEN}✓ 새 가상환경 생성 완료${NC}"
    else
        echo "기존 가상환경 사용"
    fi
else
    python3 -m venv venv_mmwalk
    echo -e "${GREEN}✓ 가상환경 생성 완료${NC}"
fi

# 가상환경 활성화
source venv_mmwalk/bin/activate

# 3. pip 업그레이드
echo -e "\n${YELLOW}[3/7] pip 업그레이드...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip 업그레이드 완료${NC}"

# 4. InternVL 저장소 클론
echo -e "\n${YELLOW}[4/7] InternVL 저장소 확인...${NC}"
cd ..
if [ ! -d "InternVL" ]; then
    echo "InternVL 저장소 클론 중..."
    git clone https://github.com/OpenGVLab/InternVL.git
    echo -e "${GREEN}✓ InternVL 클론 완료${NC}"
else
    echo "InternVL 저장소가 이미 존재합니다."
fi
cd mmWalk

# 5. 필수 패키지 설치
echo -e "\n${YELLOW}[5/7] 필수 패키지 설치 중... (시간이 걸릴 수 있습니다)${NC}"

# CUDA 버전 감지
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "감지된 CUDA 버전: $CUDA_VERSION"
else
    echo -e "${YELLOW}⚠ nvidia-smi를 찾을 수 없습니다. CUDA 12.1로 가정합니다.${NC}"
    CUDA_VERSION="12.1"
fi

# PyTorch 설치
echo "PyTorch 설치 중..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 기타 패키지
echo "기타 패키지 설치 중..."
pip install transformers==4.37.2
pip install sentencepiece
pip install timm==0.9.10
pip install einops
pip install shortuuid
pip install deepspeed
pip install lmdeploy
pip install Pillow
pip install tensorboard
pip install huggingface-hub

echo -e "${GREEN}✓ 필수 패키지 설치 완료${NC}"

# 6. InternVL 설치
echo -e "\n${YELLOW}[6/7] InternVL 설치 중...${NC}"
if [ -d "../InternVL/internvl_chat" ]; then
    # InternVL을 현재 디렉토리에 복사 또는 링크
    if [ ! -d "internvl" ]; then
        cp -r ../InternVL/internvl_chat/internvl .
        echo -e "${GREEN}✓ InternVL 복사 완료${NC}"
    else
        echo "internvl 디렉토리가 이미 존재합니다."
    fi
else
    echo -e "${RED}✗ InternVL 디렉토리를 찾을 수 없습니다${NC}"
    exit 1
fi

# 7. Flash Attention 설치 (선택사항)
echo -e "\n${YELLOW}[7/7] Flash Attention 설치 (선택사항)...${NC}"
read -p "Flash Attention을 설치하시겠습니까? (권장, y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install flash-attn --no-build-isolation || echo -e "${YELLOW}⚠ Flash Attention 설치 실패 (선택사항이므로 계속 진행)${NC}"
fi

# 디렉토리 구조 생성
echo -e "\n${YELLOW}디렉토리 구조 생성...${NC}"
mkdir -p pretrained/data
mkdir -p work_dirs

echo -e "\n${GREEN}=========================================="
echo "환경 설정 완료!"
echo "==========================================${NC}"
echo ""
echo "다음 단계:"
echo "1. 데이터셋 다운로드: https://doi.org/10.7910/DVN/KKDXDK"
echo "2. QAFrames.zip을 pretrained/data/에 복사하고 압축 해제"
echo "3. InternVL2-8B 모델 다운로드:"
echo "   huggingface-cli download --resume-download --local-dir pretrained/InternVL2-8B --local-dir-use-symlinks False OpenGVLab/InternVL2-8B"
echo "4. 어노테이션 파일 압축 해제 및 복사:"
echo "   cd finetune_related && unzip mmWalkQA_Annotation_for_Internvl2.zip"
echo "   cp mmWalkQA_Annotation_for_Internvl2.jsonl ../pretrained/data/QAFrames/"
echo "5. 파인튜닝 실행: bash finetune_mmwalk.sh"
echo ""
echo "자세한 내용은 SETUP_GUIDE_KR.md를 참조하세요."
