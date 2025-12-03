#!/usr/bin/env python3
"""
mmWalk 파인튜닝 환경 검증 스크립트
실행: python check_environment.py
"""

import os
import sys
import subprocess
from pathlib import Path

# 색상 코드
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{text:^60}{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")

def print_success(text):
    print(f"{GREEN}✓{NC} {text}")

def print_error(text):
    print(f"{RED}✗{NC} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{NC} {text}")

def check_python_version():
    """Python 버전 확인"""
    print_header("Python 버전 확인")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"현재 Python 버전: {version_str}")

    if (3, 8) <= version < (3, 11):
        print_success("Python 버전 적합 (3.8 <= version < 3.11)")
        return True
    else:
        print_error("Python 3.8-3.10 버전이 필요합니다")
        return False

def check_cuda():
    """CUDA 사용 가능 여부 확인"""
    print_header("CUDA 확인")
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA 사용 가능")
            print(f"  - GPU 개수: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print_error("CUDA를 사용할 수 없습니다")
            print_warning("GPU 없이는 파인튜닝이 매우 느리거나 불가능합니다")
            return False
    except ImportError:
        print_error("PyTorch가 설치되지 않았습니다")
        return False

def check_packages():
    """필수 패키지 설치 확인"""
    print_header("필수 패키지 확인")

    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'deepspeed': 'DeepSpeed',
        'lmdeploy': 'LMDeploy',
        'timm': 'TIMM',
        'einops': 'Einops',
        'sentencepiece': 'SentencePiece',
    }

    all_installed = True
    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} 설치됨")
        except ImportError:
            print_error(f"{name} 설치되지 않음")
            all_installed = False

    # Flash Attention (선택사항)
    try:
        __import__('flash_attn')
        print_success("Flash Attention 설치됨 (선택사항)")
    except ImportError:
        print_warning("Flash Attention 설치되지 않음 (선택사항, 권장)")

    return all_installed

def check_directories():
    """필수 디렉토리 및 파일 확인"""
    print_header("디렉토리 및 파일 구조 확인")

    checks = {
        'pretrained/data/QAFrames': '데이터셋 디렉토리',
        'pretrained/data/QAFrames/mmWalkQA_Annotation_for_Internvl2.jsonl': '어노테이션 파일',
        'pretrained/InternVL2-8B': 'InternVL2-8B 모델',
        'pretrained/InternVL2-8B/config.json': '모델 설정 파일',
        'finetune_related/mmwalk.json': '메타데이터 파일',
        'zero_stage1_config.json': 'DeepSpeed 설정 파일',
        'finetune_mmwalk.sh': '파인튜닝 스크립트',
        'internvl': 'InternVL 코드',
    }

    all_exist = True
    for path, description in checks.items():
        full_path = Path(path)
        if full_path.exists():
            if full_path.is_dir():
                # 디렉토리 크기 추정
                try:
                    size = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file())
                    size_str = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.2f} MB"
                    print_success(f"{description}: {path} ({size_str})")
                except:
                    print_success(f"{description}: {path}")
            else:
                size = full_path.stat().st_size
                size_str = f"{size / 1e6:.2f} MB" if size > 1e6 else f"{size / 1e3:.2f} KB"
                print_success(f"{description}: {path} ({size_str})")
        else:
            print_error(f"{description} 없음: {path}")
            all_exist = False

    return all_exist

def check_dataset_samples():
    """데이터셋 샘플 개수 확인"""
    print_header("데이터셋 샘플 확인")

    annotation_file = Path('pretrained/data/QAFrames/mmWalkQA_Annotation_for_Internvl2.jsonl')

    if not annotation_file.exists():
        print_error("어노테이션 파일을 찾을 수 없습니다")
        return False

    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sample_count = len(lines)

        print(f"총 샘플 개수: {sample_count}")

        if sample_count == 69390:
            print_success("예상 샘플 개수와 일치 (69,390)")
            return True
        else:
            print_warning(f"예상과 다름 (예상: 69,390, 실제: {sample_count})")
            return True
    except Exception as e:
        print_error(f"어노테이션 파일 읽기 실패: {e}")
        return False

def check_disk_space():
    """디스크 공간 확인"""
    print_header("디스크 공간 확인")

    try:
        stat = os.statvfs('.')
        free_space = stat.f_bavail * stat.f_frsize / 1e9

        print(f"사용 가능한 디스크 공간: {free_space:.1f} GB")

        if free_space >= 100:
            print_success("충분한 디스크 공간")
            return True
        elif free_space >= 50:
            print_warning("디스크 공간이 부족할 수 있습니다 (최소 100GB 권장)")
            return True
        else:
            print_error("디스크 공간 부족 (최소 50GB 필요)")
            return False
    except Exception as e:
        print_warning(f"디스크 공간 확인 실패: {e}")
        return True

def main():
    """메인 함수"""
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{'mmWalk 파인튜닝 환경 검증':^60}{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")

    results = {
        'Python 버전': check_python_version(),
        'CUDA': check_cuda(),
        '필수 패키지': check_packages(),
        '파일 구조': check_directories(),
        '데이터셋': check_dataset_samples(),
        '디스크 공간': check_disk_space(),
    }

    # 최종 결과
    print_header("검증 결과 요약")

    all_passed = True
    for check, passed in results.items():
        if passed:
            print_success(f"{check}: 통과")
        else:
            print_error(f"{check}: 실패")
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{'='*60}{NC}")
        print(f"{GREEN}{'모든 검증 통과! 파인튜닝을 시작할 수 있습니다.':^60}{NC}")
        print(f"{GREEN}{'='*60}{NC}\n")
        print("다음 명령어로 파인튜닝을 시작하세요:")
        print(f"  {BLUE}bash finetune_mmwalk.sh{NC}\n")
        return 0
    else:
        print(f"{RED}{'='*60}{NC}")
        print(f"{RED}{'일부 검증 실패. 위의 오류를 해결해주세요.':^60}{NC}")
        print(f"{RED}{'='*60}{NC}\n")
        print("자세한 설정 방법은 SETUP_GUIDE_KR.md를 참조하세요.\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
