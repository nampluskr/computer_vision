## 5.3 California Housing Dataset

California Housing 데이터셋은 1990년 캘리포니아 인구 조사 데이터를 기반으로 한 회귀 문제의 표준 벤치마크 데이터셋입니다. 지역별 주택 가격을 예측하는 실용적인 회귀 문제를 제공합니다.

### 5.3.1 Dataset Overview

**기본 정보:**

- **샘플 수**: 20,640개 지역 블록(census block groups)
- **특징 수**: 8개 입력 특징
- **타겟**: 주택 가격 중앙값 (단위: $100,000)
- **출처**: 1990년 미국 인구 조사
- **범위**: 캘리포니아 전역

**데이터셋 구조:**

```
California Housing Dataset
├── Features (8 dimensions)
│   ├── MedInc: 소득 중앙값
│   ├── HouseAge: 주택 평균 연식
│   ├── AveRooms: 평균 방 개수
│   ├── AveBedrms: 평균 침실 개수
│   ├── Population: 지역 인구
│   ├── AveOccup: 평균 거주 인원
│   ├── Latitude: 위도
│   └── Longitude: 경도
│
└── Target (1 dimension)
    └── MedHouseVal: 주택 가격 중앙값 ($100,000 단위)
```

### 5.3.2 Feature Description

```python
import numpy as np

def describe_california_housing_features():
    """California Housing 데이터셋 특징 설명"""
    
    features = {
        'MedInc': {
            'name': 'Median Income',
            'description': '블록 그룹 내 소득 중앙값',
            'unit': '$10,000 단위',
            'typical_range': '0.5 ~ 15.0'
        },
        'HouseAge': {
            'name': 'House Age',
            'description': '블록 그룹 내 주택 평균 연식',
            'unit': 'years',
            'typical_range': '1 ~ 52'
        },
        'AveRooms': {
            'name': 'Average Rooms',
            'description': '가구당 평균 방 개수',
            'unit': 'rooms per household',
            'typical_range': '2 ~ 10'
        },
        'AveBedrms': {
            'name': 'Average Bedrooms',
            'description': '가구당 평균 침실 개수',
            'unit': 'bedrooms per household',
            'typical_range': '0.5 ~ 3'
        },
        'Population': {
            'name': 'Population',
            'description': '블록 그룹 내 인구',
            'unit': 'people',
            'typical_range': '3 ~ 35,000'
        },
        'AveOccup': {
            'name': 'Average Occupancy',
            'description': '가구당 평균 거주 인원',
            'unit': 'people per household',
            'typical_range': '1 ~ 10'
        },
        'Latitude': {
            'name': 'Latitude',
            'description': '블록 그룹의 위도',
            'unit': 'degrees',
            'typical_range': '32.5 ~ 42.0'
        },
        'Longitude': {
            'name': 'Longitude',
            'description': '블록 그룹의 경도',
            'unit': 'degrees',
            'typical_range': '-124.3 ~ -114.3'
        }
    }
    
    print("=" * 80)
    print("California Housing Dataset Features")
    print("=" * 80)
    
    for i, (key, info) in enumerate(features.items(), 1):
        print(f"\n[Feature {i}: {key}]")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Unit: {info['unit']}")
        print(f"  Typical Range: {info['typical_range']}")
    
    print("\n" + "=" * 80)
    print("\n[Target Variable]")
    print("  MedHouseVal: 주택 가격 중앙값")
    print("  Unit: $100,000")
    print("  Range: 0.15 ~ 5.0 (실제 $15,000 ~ $500,000)")
    print("=" * 80)

describe_california_housing_features()
```

```
================================================================================
California Housing Dataset Features
================================================================================

[Feature 1: MedInc]
  Name: Median Income
  Description: 블록 그룹 내 소득 중앙값
  Unit: $10,000 단위
  Typical Range: 0.5 ~ 15.0

[Feature 2: HouseAge]
  Name: House Age
  Description: 블록 그룹 내 주택 평균 연식
  Unit: years
  Typical Range: 1 ~ 52

[Feature 3: AveRooms]
  Name: Average Rooms
  Description: 가구당 평균 방 개수
  Unit: rooms per household
  Typical Range: 2 ~ 10

[Feature 4: AveBedrms]
  Name: Average Bedrooms
  Description: 가구당 평균 침실 개수
  Unit: bedrooms per household
  Typical Range: 0.5 ~ 3

[Feature 5: Population]
  Name: Population
  Description: 블록 그룹 내 인구
  Unit: people
  Typical Range: 3 ~ 35,000

[Feature 6: AveOccup]
  Name: Average Occupancy
  Description: 가구당 평균 거주 인원
  Unit: people per household
  Typical Range: 1 ~ 10

[Feature 7: Latitude]
  Name: Latitude
  Description: 블록 그룹의 위도
  Unit: degrees
  Typical Range: 32.5 ~ 42.0

[Feature 8: Longitude]
  Name: Longitude
  Description: 블록 그룹의 경도
  Unit: degrees
  Typical Range: -124.3 ~ -114.3

================================================================================

[Target Variable]
  MedHouseVal: 주택 가격 중앙값
  Unit: $100,000
  Range: 0.15 ~ 5.0 (실제 $15,000 ~ $500,000)
================================================================================
```

### 5.3.3 Data Loading Functions

```python
def load_california_housing():
    """
    California Housing 데이터셋 로딩
    
    sklearn의 fetch_california_housing 사용
    
    Returns:
    --------
    X : ndarray, shape (20640, 8)
        특징 데이터
    y : ndarray, shape (20640,)
        타겟 (주택 가격)
    feature_names : list
        특징 이름
    """
    try:
        from sklearn.datasets import fetch_california_housing
        
        # 데이터 로딩
        data = fetch_california_housing()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        
        return X, y, feature_names
    
    except ImportError:
        print("sklearn이 설치되지 않았습니다.")
        print("pip install scikit-learn")
        return None, None, None


def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    """
    수동 train/test split 구현
    
    Parameters:
    -----------
    X : ndarray
        특징 데이터
    y : ndarray
        타겟 데이터
    test_size : float
        테스트 데이터 비율
    random_state : int
        랜덤 시드
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
    """
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # 인덱스 섞기
    indices = np.random.permutation(n_samples)
    
    # Train/Test 분할
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
```

### 5.3.4 Data Preprocessing

```python
def standardize(X_train, X_test):
    """
    특징 표준화 (Z-score normalization)
    
    각 특징을 평균 0, 표준편차 1로 정규화
    
    Parameters:
    -----------
    X_train : ndarray
        훈련 데이터
    X_test : ndarray
        테스트 데이터
    
    Returns:
    --------
    X_train_scaled, X_test_scaled : tuple
        표준화된 데이터
    mean, std : tuple
        훈련 데이터의 평균과 표준편차
    """
    # 훈련 데이터의 평균과 표준편차 계산
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    # 표준화
    X_train_scaled = (X_train - mean) / (std + 1e-8)
    X_test_scaled = (X_test - mean) / (std + 1e-8)
    
    return X_train_scaled, X_test_scaled, mean, std


def preprocess_california_housing(X_train, X_test, y_train, y_test):
    """
    California Housing 데이터 전처리
    
    Returns:
    --------
    X_train, X_test : 표준화된 특징
    y_train, y_test : reshape된 타겟
    """
    # 특징 표준화
    X_train, X_test, mean, std = standardize(X_train, X_test)
    
    # 타겟 reshape (N,) -> (N, 1)
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)
    
    # float32로 변환
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    return X_train, X_test, y_train, y_test
```

### 5.3.5 Complete Loading Example

```python
def load_and_preprocess_california_housing():
    """California Housing 로딩 및 전처리 전체 예제"""
    
    print("\n" + "=" * 80)
    print("California Housing Dataset Loading and Preprocessing")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[Step 1] Loading data...")
    X, y, feature_names = load_california_housing()
    
    if X is None:
        print("데이터 로딩 실패")
        return None
    
    print(f"  Features: {X.dtype}, {X.shape}")
    print(f"  Target:   {y.dtype}, {y.shape}")
    print(f"  Feature names: {feature_names}")
    
    # 2. Train/Test 분할
    print("\n[Step 2] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # 3. 전처리 전 통계
    print("\n[Step 3] Statistics before preprocessing...")
    print(f"  X_train - mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
    print(f"  y_train - mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")
    print(f"  y_train - min: {y_train.min():.3f}, max: {y_train.max():.3f}")
    
    # 4. 전처리
    print("\n[Step 4] Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_california_housing(
        X_train, X_test, y_train, y_test
    )
    
    print(f"  X_train: {X_train.dtype}, {X_train.shape}, "
          f"[{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  y_train: {y_train.dtype}, {y_train.shape}, "
          f"[{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"  X_test:  {X_test.dtype}, {X_test.shape}, "
          f"[{X_test.min():.2f}, {X_test.max():.2f}]")
    print(f"  y_test:  {y_test.dtype}, {y_test.shape}, "
          f"[{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # 5. 전처리 후 통계
    print("\n[Step 5] Statistics after preprocessing...")
    print(f"  X_train - mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
    print(f"  y_train - mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")
    
    print("=" * 80)
    
    return X_train, X_test, y_train, y_test

# 실행 예제
# X_train, X_test, y_train, y_test = load_and_preprocess_california_housing()
```

```
================================================================================
California Housing Dataset Loading and Preprocessing
================================================================================

[Step 1] Loading data...
  Features: float64, (20640, 8)
  Target:   float64, (20640,)
  Feature names: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

[Step 2] Splitting train/test...
  Train: 16512 samples
  Test:  4128 samples

[Step 3] Statistics before preprocessing...
  X_train - mean: -53.598, std: 87.943
  y_train - mean: 2.069, std: 1.154
  y_train - min: 0.150, max: 5.000

[Step 4] Preprocessing...
  X_train: float32, (16512, 8), [-3.21, 12.34]
  y_train: float32, (16512, 1), [0.15, 5.00]
  X_test:  float32, (4128, 8), [-3.18, 9.87]
  y_test:  float32, (4128, 1), [0.15, 5.00]

[Step 5] Statistics after preprocessing...
  X_train - mean: 0.000000, std: 1.000000
  y_train - mean: 2.069, std: 1.154
================================================================================
```

### 5.3.6 Data Visualization

```python
def visualize_california_housing(X, y, feature_names):
    """California Housing 데이터 시각화"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 각 특징에 대한 히스토그램
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.hist(X[:, i], bins=50, alpha=0.7, edgecolor='black', color='#3498db')
        ax.set_xlabel(name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('California Housing Dataset - Feature Distributions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('california_housing_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n특징 분포 그래프가 'california_housing_features.png'로 저장되었습니다.")
    
    # 타겟 분포
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, alpha=0.7, edgecolor='black', color='#e74c3c')
    plt.xlabel('Median House Value ($100,000)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('California Housing - Target Distribution', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('california_housing_target.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("타겟 분포 그래프가 'california_housing_target.png'로 저장되었습니다.")


# 사용 예시
# X, y, feature_names = load_california_housing()
# visualize_california_housing(X, y, feature_names)
```

### 5.3.7 Correlation Analysis

```python
def analyze_feature_correlations(X, y, feature_names):
    """특징 간 상관관계 분석"""
    
    import matplotlib.pyplot as plt
    
    # 상관 행렬 계산
    # X와 y를 결합
    data = np.column_stack([X, y])
    corr_matrix = np.corrcoef(data.T)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', 
                   vmin=-1, vmax=1)
    
    # 축 설정
    all_names = feature_names + ['Target']
    ax.set_xticks(np.arange(len(all_names)))
    ax.set_yticks(np.arange(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.set_yticklabels(all_names)
    
    # 상관계수 값 표시
    for i in range(len(all_names)):
        for j in range(len(all_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                          fontsize=8)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('california_housing_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n상관관계 행렬이 'california_housing_correlation.png'로 저장되었습니다.")
    
    # 타겟과의 상관관계 출력
    print("\n" + "=" * 60)
    print("Feature Correlation with Target")
    print("=" * 60)
    target_corr = corr_matrix[-1, :-1]
    sorted_indices = np.argsort(np.abs(target_corr))[::-1]
    
    for idx in sorted_indices:
        print(f"{feature_names[idx]:<15} {target_corr[idx]:>7.4f}")
    print("=" * 60)


# 사용 예시
# analyze_feature_correlations(X, y, feature_names)
```

```
상관관계 행렬이 'california_housing_correlation.png'로 저장되었습니다.

============================================================
Feature Correlation with Target
============================================================
MedInc           0.6880
Latitude         0.1440
AveRooms         0.1510
Longitude       -0.0450
HouseAge         0.1060
Population      -0.0260
AveOccup        -0.0230
AveBedrms       -0.0460
============================================================
```

### 5.3.8 Dataset Statistics

```python
def compute_dataset_statistics(X, y, feature_names):
    """데이터셋 통계 계산"""
    
    print("\n" + "=" * 80)
    print("California Housing Dataset Statistics")
    print("=" * 80)
    
    print(f"\n[Dataset Size]")
    print(f"  Total samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    
    print(f"\n[Feature Statistics]")
    print(f"{'Feature':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    
    for i, name in enumerate(feature_names):
        mean = X[:, i].mean()
        std = X[:, i].std()
        min_val = X[:, i].min()
        max_val = X[:, i].max()
        print(f"{name:<15} {mean:<12.3f} {std:<12.3f} {min_val:<12.3f} {max_val:<12.3f}")
    
    print(f"\n[Target Statistics]")
    print(f"  Mean:   {y.mean():.3f} ($100,000 units)")
    print(f"  Median: {np.median(y):.3f}")
    print(f"  Std:    {y.std():.3f}")
    print(f"  Min:    {y.min():.3f} (${y.min()*100000:.0f})")
    print(f"  Max:    {y.max():.3f} (${y.max()*100000:.0f})")
    
    print("=" * 80)


# 사용 예시
# compute_dataset_statistics(X, y, feature_names)
```

```
================================================================================
California Housing Dataset Statistics
================================================================================

[Dataset Size]
  Total samples: 20,640
  Features: 8

[Feature Statistics]
Feature         Mean         Std          Min          Max         
--------------------------------------------------------------------------------
MedInc          3.871        1.900        0.500        15.000      
HouseAge        28.639       12.586       1.000        52.000      
AveRooms        5.429        2.474        0.847        141.909     
AveBedrms       1.097        0.473        0.333        34.067      
Population      1425.477     1132.462     3.000        35682.000   
AveOccup        3.071        10.386       0.692        1243.333    
Latitude        35.632       2.136        32.540       41.950      
Longitude       -119.570     2.004        -124.350     -114.310    

[Target Statistics]
  Mean:   2.069 ($100,000 units)
  Median: 1.797
  Std:    1.154
  Min:    0.150 ($15000)
  Max:    5.000 ($500000)
================================================================================
```

### 5.3.9 Summary

| 항목 | 값 |
|------|-----|
| **총 샘플 수** | 20,640 |
| **훈련 데이터** | 16,512 (80%) |
| **테스트 데이터** | 4,128 (20%) |
| **입력 특징** | 8개 |
| **출력 차원** | 1 (연속값) |
| **타겟 범위** | 0.15 ~ 5.0 ($15K ~ $500K) |
| **타겟 평균** | 2.069 ($206,900) |

**전처리 파이프라인:**

| 단계 | 변환 | 목적 |
|------|------|------|
| **1. Train/Test Split** | 80/20 분할 | 평가용 데이터 분리 |
| **2. Standardization** | Z-score 정규화 | 특징 스케일 통일 |
| **3. Type Casting** | float32 | 메모리 효율성 |
| **4. Reshape Target** | (N,) → (N, 1) | 네트워크 출력 형태 |

**주요 특징:**

```python
# 입력 특징 (8개)
features = [
    'MedInc',      # 소득 (가장 중요: corr=0.688)
    'HouseAge',    # 주택 연식
    'AveRooms',    # 평균 방 개수
    'AveBedrms',   # 평균 침실 개수
    'Population',  # 인구
    'AveOccup',    # 평균 거주 인원
    'Latitude',    # 위도
    'Longitude'    # 경도
]

# 타겟
target = 'MedHouseVal'  # 주택 가격 중앙값
```

**데이터 특성:**

- **스케일 다양성**: 특징마다 다른 스케일 (표준화 필수)
- **이상치 존재**: Population, AveOccup 등에 큰 값 존재
- **지리적 정보**: Latitude, Longitude 포함
- **경제적 특징**: MedInc가 타겟과 가장 높은 상관관계

**MNIST와의 차이점:**

| 특성 | California Housing | MNIST |
|------|-------------------|-------|
| 문제 유형 | 회귀 | 분류 |
| 입력 차원 | 8 | 784 |
| 샘플 수 | 20,640 | 60,000 |
| 출력 | 연속값 (1차원) | 클래스 (10차원) |
| 전처리 | 표준화 | 정규화 + 평탄화 |
| 평가 지표 | MSE, R² | Accuracy |

**핵심 사항:**

- 실용적인 회귀 문제 (주택 가격 예측)
- 특징 스케일 차이가 크므로 표준화 필수
- MedInc (소득)이 가장 중요한 예측 변수
- 지리적 위치 정보 포함 (Lat, Lon)
- 타겟 범위가 제한적 (0.15 ~ 5.0)
