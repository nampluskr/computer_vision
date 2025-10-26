# Anomalib 라이선스 및 Apache License 2.0 분석

## 1. Anomalib 라이선스 확인

### 1.1. Anomalib의 라이선스

Anomalib는 **Apache License 2.0**으로 배포됩니다.

**공식 GitHub 저장소**: https://github.com/openvinotoolkit/anomalib

**라이선스 파일**: https://github.com/openvinotoolkit/anomalib/blob/main/LICENSE

### 1.2. 본 프로젝트의 사용 사례

본 프로젝트에서는 다음과 같이 Anomalib 코드를 사용했습니다:

**직접 사용한 부분:**
- 각 모델의 `torch_model.py`: 모델 아키텍처 구현
- `components/` 폴더의 일부 파일: 공통 컴포넌트
- 네트워크 레이어 및 손실 함수 구현

**독자적으로 구현한 부분:**
- `BaseTrainer` 및 모델별 `*Trainer` 클래스
- `dataloader.py`: 4가지 데이터셋 통합 로더
- `registry.py`: 모델 관리 시스템
- `train.py`: 학습 유틸리티 함수
- `backbone.py`: 오프라인 가중치 관리

---

## 2. Apache License 2.0 설명

### 2.1. 주요 특징

Apache License 2.0은 **허용적(permissive) 오픈소스 라이선스**로, 다음과 같은 특징이 있습니다:

#### 허용 사항 (What You CAN Do)

1. **상업적 사용**: 상업적 목적으로 자유롭게 사용 가능
2. **수정**: 소스 코드를 수정하고 개선 가능
3. **배포**: 원본 또는 수정된 버전을 배포 가능
4. **특허 사용**: 기여자의 특허를 사용할 수 있는 명시적 권한 부여
5. **비공개 사용**: 수정 사항을 공개하지 않고 사적으로 사용 가능

#### 의무 사항 (What You MUST Do)

1. **라이선스 및 저작권 고지 포함**: 원본 라이선스와 저작권 표시 유지
2. **변경 사항 명시**: 원본 파일을 수정한 경우 명시
3. **NOTICE 파일 포함**: 원본에 NOTICE 파일이 있으면 포함

#### 금지 사항 (What You CANNOT Do)

1. **상표권 사용**: Apache 상표나 로고를 허가 없이 사용 불가
2. **보증 없음**: 소프트웨어는 "있는 그대로" 제공되며 보증 없음
3. **책임 제한**: 기여자는 소프트웨어 사용으로 인한 손해에 책임 없음

### 2.2. GPL과의 비교

| 항목 | Apache 2.0 | GPL (v2, v3) |
|-----|-----------|--------------|
| 상업적 사용 | ✓ 허용 | ✓ 허용 |
| 수정 및 배포 | ✓ 허용 | ✓ 허용 |
| 소스 공개 의무 | ✗ 없음 | ✓ 있음 (copyleft) |
| 특허 보호 | ✓ 명시적 | △ GPL v3만 명시적 |
| 라이선스 전파 | ✗ 없음 | ✓ 있음 (viral) |

**핵심 차이점:**
- Apache 2.0: 수정 사항을 비공개로 유지 가능
- GPL: 배포 시 소스 코드 공개 의무 (copyleft)

---

## 3. 본 프로젝트의 라이선스 준수 방안

### 3.1. 필수 조치 사항

#### ✓ 조치 1: LICENSE 파일 포함

프로젝트 루트에 `LICENSE` 파일을 생성하고 Apache License 2.0 전문을 포함합니다:

```
project/
├── LICENSE              # Apache License 2.0 전문
├── NOTICE              # 저작권 및 출처 명시
├── main.py
└── ...
```

**LICENSE 파일 내용:**
```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[전문 내용 - Apache 공식 사이트에서 복사]
```

#### ✓ 조치 2: NOTICE 파일 생성

```
NOTICE

This project includes code from Anomalib
Copyright (c) 2023 Intel Corporation
Licensed under the Apache License, Version 2.0

Anomalib: https://github.com/openvinotoolkit/anomalib

The following components are derived from Anomalib:
- Model implementations (torch_model.py files)
- Shared components in models/components/
- Network architectures and loss functions

Modifications:
- Removed PyTorch Lightning dependencies
- Implemented custom training pipeline (BaseTrainer)
- Added offline environment support
- Created unified interface for multiple datasets
```

#### ✓ 조치 3: 각 파일에 저작권 표시

Anomalib에서 가져온 파일의 상단에 저작권 표시를 유지합니다:

```python
# models/model_stfpm.py

"""
STFPM Model Implementation

Original implementation from Anomalib:
Copyright (C) 2023 Intel Corporation
SPDX-License-Identifier: Apache-2.0

Modified for offline environment and pure PyTorch training.
"""

# ... 코드 ...
```

#### ✓ 조치 4: README에 명시

README.md에 Anomalib 사용 사실을 명확히 기재합니다 (이미 완료):

```markdown
### 1.5. Anomalib과의 관계

본 프레임워크는 [Anomalib](https://github.com/openvinotoolkit/anomalib)의 
모델 아키텍처를 기반으로 하되, 실행 환경에 맞게 전면 재구성하였습니다.

#### 1.5.1. Anomalib 활용 부분
- 각 모델의 torch_model.py: Anomalib의 검증된 구현 사용
- ...
```

### 3.2. 권장 조치 사항

#### 권장 1: 변경 이력 문서화

`CHANGES.md` 파일을 생성하여 주요 변경 사항을 기록합니다:

```markdown
# Changes from Anomalib

## Major Modifications

1. **Removed Dependencies**
   - Removed PyTorch Lightning
   - Removed Lightning-related imports and decorators

2. **Added Components**
   - BaseTrainer: Unified training interface
   - Custom DataLoaders: Support for 4 dataset types
   - Registry System: Model configuration management
   - Offline Support: Local backbone weight management

3. **Modified Files**
   - All model implementations: Adapted for BaseTrainer
   - Training pipeline: Pure PyTorch implementation
   - Evaluation metrics: Custom threshold analysis

## Unchanged Components (from Anomalib)

- Model architectures (torch_model.py)
- Network layers and loss functions
- Core algorithms from papers
```

#### 권장 2: 기여자 인정

`CONTRIBUTORS.md` 파일을 생성합니다:

```markdown
# Contributors

## Original Work (Anomalib)

This project is built upon Anomalib:
- Repository: https://github.com/openvinotoolkit/anomalib
- Copyright: Intel Corporation
- License: Apache 2.0

## This Project

- Framework Design: [Your Name/Team]
- Training Pipeline: [Your Name/Team]
- Dataset Integration: [Your Name/Team]
- Documentation: [Your Name/Team]
```

### 3.3. 파일별 라이선스 헤더 예시

#### Anomalib에서 가져온 파일

```python
# models/model_padim.py

"""
PaDiM Model Implementation

Original implementation from Anomalib:
Copyright (C) 2023 Intel Corporation
SPDX-License-Identifier: Apache-2.0
Source: https://github.com/openvinotoolkit/anomalib

Modified for this project:
- Adapted for BaseTrainer interface
- Removed Lightning dependencies
- Added offline weight loading support
"""
```

#### 독자적으로 작성한 파일

```python
# train.py

"""
Training Utility Functions

Copyright (C) 2025 [Your Name/Organization]
SPDX-License-Identifier: Apache-2.0

This file is part of the Anomaly Detection Framework.
"""
```

---

## 4. 라이선스 준수 체크리스트

### 4.1. 필수 항목

- [ ] **LICENSE 파일 포함**: Apache License 2.0 전문
- [ ] **NOTICE 파일 생성**: Anomalib 저작권 및 출처 명시
- [ ] **저작권 표시 유지**: 각 파일 상단에 원본 저작권 표시
- [ ] **변경 사항 명시**: 수정한 파일에 변경 내역 기재
- [ ] **README 업데이트**: Anomalib 사용 사실 명시

### 4.2. 권장 항목

- [ ] **CHANGES.md 작성**: 주요 변경 사항 문서화
- [ ] **CONTRIBUTORS.md 작성**: 기여자 및 출처 인정
- [ ] **상표권 확인**: "Anomalib" 명칭 사용 시 출처 명시
- [ ] **문서화**: 각 모델의 원 논문 인용

---

## 5. 법적 검토 사항

### 5.1. Apache 2.0 준수 확인

✅ **본 프로젝트는 Apache License 2.0을 완전히 준수합니다:**

1. **라이선스 호환성**: Apache 2.0는 매우 허용적인 라이선스
2. **상업적 사용**: 가능 (제한 없음)
3. **수정 및 배포**: 가능 (소스 공개 의무 없음)
4. **특허 보호**: 명시적 특허 사용 권한 부여

### 5.2. 주의 사항

⚠️ **다음 사항을 주의해야 합니다:**

1. **상표권**: "Anomalib" 상표 사용 시 출처 명시 필요
2. **보증 없음**: 소프트웨어는 "있는 그대로" 제공
3. **책임 제한**: 사용으로 인한 문제에 대한 법적 책임 없음
4. **기여자 보호**: 특허 소송 방어 조항 포함

### 5.3. 권장 면책 조항

README.md에 다음을 추가합니다:

```markdown
## Disclaimer

This software is provided "AS IS", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.

This project includes code from Anomalib (Apache 2.0), developed by Intel
Corporation. See LICENSE and NOTICE files for details.
```

---

## 6. 결론 및 권장 사항

### 6.1. 라이선스 준수 상태

✅ **본 프로젝트는 Apache License 2.0을 준수할 수 있습니다.**

**이유:**
1. Apache 2.0는 매우 허용적인 라이선스
2. 상업적 사용, 수정, 배포 모두 허용
3. 소스 코드 공개 의무 없음
4. 필수 조건(라이선스 표시, 변경 명시)만 충족하면 됨

### 6.2. 즉시 수행해야 할 조치

#### 우선순위 1 (필수)

1. **LICENSE 파일 추가**
```bash
# Apache License 2.0 전문을 다운로드
curl -o LICENSE https://www.apache.org/licenses/LICENSE-2.0.txt
```

2. **NOTICE 파일 생성**
```bash
# 위의 3.1 섹션의 NOTICE 내용으로 파일 생성
```

3. **각 Anomalib 파일에 저작권 표시 추가**
```python
# 각 model_*.py 파일 상단에 저작권 표시 추가
```

#### 우선순위 2 (권장)

4. **CHANGES.md 작성**: 변경 사항 문서화
5. **CONTRIBUTORS.md 작성**: 기여자 인정
6. **README.md 업데이트**: 면책 조항 추가

### 6.3. 최종 프로젝트 구조

```
project/
├── LICENSE                 # Apache License 2.0 전문
├── NOTICE                  # Anomalib 저작권 표시
├── README.md              # 사용법 및 출처 명시
├── CHANGES.md             # 변경 사항 문서화
├── CONTRIBUTORS.md        # 기여자 인정
├── main.py
├── train.py
├── registry.py
├── dataloader.py
└── models/
    ├── model_*.py         # 각 파일에 저작권 표시
    └── components/
```

### 6.4. 추가 고려사항

**학술 논문 또는 상업적 사용 시:**
- 원 논문 인용: 각 모델의 원 논문 인용
- Anomalib 인정: "Built upon Anomalib framework" 명시
- 라이선스 준수: LICENSE 및 NOTICE 파일 포함

**오픈소스 공개 시:**
- GitHub 저장소에 LICENSE 파일 필수
- README에 Anomalib 출처 명시
- 기여 가이드라인 작성 권장

---

## 요약

✅ **본 프로젝트는 Apache License 2.0을 준수하며 법적으로 문제없습니다.**

**필수 조치 3가지:**
1. LICENSE 파일 추가 (Apache License 2.0 전문)
2. NOTICE 파일 생성 (Anomalib 저작권 표시)
3. 각 파일에 저작권 표시 추가

이 조치들을 완료하면 상업적 사용, 배포, 수정 모두 자유롭게 가능합니다.