# 문서 작성 컨텍스트

## 프로젝트 개요
"Deep Learning from Scratch with NumPy" 초급 과정 교재를 작성 중입니다.

## 작성 완료 섹션
- Part 1: Chapter 1 (1.1, 1.2, 1.3), Chapter 2 (2.1, 2.2, 2.3)

## 작성 예정 섹션
- Part 1: Chapter 3 (3.1~3.5)
- Part 2: Chapter 4~8
- Part 3: Chapter 9

## 문서 형식 규칙

### 구조
- 섹션: `## X.X Section Name` (한글 설명)
- 서브섹션: `### X.X.X Subsection Name` (한글 설명)
- 챕터/섹션명은 영어, 내용은 한국어

### 수식
- 인라인: `$수식$`
- 블록: `$$수식$$`
- LaTeX 문법 사용

### 코드
- Python 코드 블록 사용
- 실행 결과는 별도 코드 블록으로 표시
- NumPy 기반 구현

### 내용 구성
1. 개념 설명 (한국어)
2. 수학적 정의 (수식)
3. NumPy 구현 (코드)
4. 실행 결과 (출력)
5. 시각화 (matplotlib, 선택적)
6. Summary 테이블 (섹션 끝)

### 예시 형식
````markdown
## 1.1 Linear Algebra Essentials

신경망의 연산은 본질적으로 선형대수 연산입니다...

### 1.1.1 Scalars, Vectors, Matrices, and Tensors

딥러닝에서 다루는 데이터는 차원에 따라...

**정의:**

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$
```python
import numpy as np
A = np.random.randn(3, 4)
...
```
```
출력 결과
```

### 1.1.7 Summary

| 연산 | 수식 | NumPy | 신경망 활용 |
|------|------|-------|-------------|
| ... | ... | ... | ... |
````

## 참조 코드

첨부된 코드들:
1. `mnist_numpy_mlp_v1_basic.py` - 기본 MNIST MLP 구현
2. `mnist_numpy_mlp_v6_trainer.py` - 모듈화된 최종 버전

## 데이터셋
- Regression: California Housing
- Binary Classification: MNIST 0 vs 1
- Multiclass Classification: MNIST 10 classes

## 요청
[작성할 섹션 번호]를 위 형식에 맞춰 작성해 주세요.