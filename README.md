# AT-DGNN

이 저장소는 [AT-DGNN](https://github.com/xmh1011/AT-DGNN.git) 코드를 기반으로 MEEG 데이터셋에 맞춰 수정한 프로젝트입니다. 구현 및 학습 내용을 포함하고 있습니다.

---

## 목차
- [프로젝트 소개](#프로젝트-소개)
- [주요 특징](#주요-특징)
- [설치 및 사용 방법](#설치-및-사용-방법)
  - [필수 조건](#1-필수-조건)
  - [설치](#2-설치)
  - [실행](#3-실행)
- [프로젝트 구조](#프로젝트-구조)

---

## 프로젝트 소개

**AT-DGNN (Attention-based Temporal Dynamic Graph Neural Network)**는 동적 그래프 학습을 위해 설계된 고급 신경망 프레임워크입니다. 이 프로젝트는 원본 AT-DGNN 코드를 기반으로, **MEEG** 데이터셋에 최적화되도록 수정했습니다. 본 프로젝트의 주요 목표는 MEEG 데이터셋의 전처리, 모델 학습 및 평가 과정을 효율적으로 수행하는 것입니다.

---

## 주요 특징

- **MEEG 데이터셋**: MEEG 데이터셋은 음악을 통한 감정 유도를 기반한 감정 유도 EEG 데이터셋.
- **AT-DGNN 모델 수정**: 신경 신호 데이터를 처리하기 위한 모델 구조 조정.
- **학습 과정 포함**: 프로젝트 수행 중 학습한 내용을 코드와 문서로 제공.

---

## 설치 및 사용 방법

### 1. 필수 조건
- Python 3.9 이상
- `einops~=0.8.0`
- `h5py~=3.9.0`
- `numpy~=1.24.3`
- `scikit_learn~=1.3.0`
- `scipy~=1.13.0`
- `torch~=2.4.0`
- `torch_geometric~=2.5.3`

다음 명령어로 필수 패키지를 설치할 수 있습니다:
```bash
pip install -r requirements.txt
```

### 2. 설치
1.	저장소를 클론합니다:
```bash
git clone https://github.com/cxeni00/AT_DGNN.git
cd AT_DGNN
```

2.	MEEG 데이터셋을 다운로드하여 data/ 디렉토리에 저장합니다.[data down load](https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR)

### 3. 실행
```bash
python main.py
```

---

## 프로젝트 구조
```plaintext
AT_DGNN/
├── config/            # 설정 파일 
├── data/              # MEEG 데이터셋
├── models/            # 수정된 AT-DGNN 모델 파일
├── train/             # 모델 학습 
├── utils/             # 유틸리티 함수 및 도구
├── main.py            # 실행 파일
├── requirements.txt   # 필요한 Python 라이브러리 목록
└── README.md          # 프로젝트 문서
```





