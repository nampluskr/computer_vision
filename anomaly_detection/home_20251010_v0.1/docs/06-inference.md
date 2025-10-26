# Inference - 배포 및 추론

## 목차

1. [개요](#1-개요)
2. [모델 로드](#2-모델-로드)
3. [단일 이미지 추론](#3-단일-이미지-추론)
4. [배치 추론](#4-배치-추론)
5. [레이블 없는 데이터 추론](#5-레이블-없는-데이터-추론)
6. [임계값 설정](#6-임계값-설정)
7. [이상 맵 생성 및 저장](#7-이상-맵-생성-및-저장)
8. [배포 최적화](#8-배포-최적화)
9. [실시간 추론](#9-실시간-추론)
10. [모델 변환 및 내보내기](#10-모델-변환-및-내보내기)
11. [프로덕션 배포](#11-프로덕션-배포)
12. [문제 해결](#12-문제-해결)

---

## 1. 개요

추론(Inference)은 학습된 모델을 사용하여 새로운 데이터에 대해 예측을 수행하는 과정입니다. 본 문서에서는 레이블이 없는 데이터에 대한 이상 감지 및 배포 방법을 다룹니다.

### 1.1. 추론 워크플로우

```
모델 로드 → 데이터 준비 → 전처리 → 추론 → 후처리 → 결과 출력
    ↓           ↓          ↓        ↓        ↓          ↓
 Weights   Images    Transform  Forward  Threshold  Decision
```

### 1.2. 출력 형식

추론 결과는 다음을 포함합니다:

- **pred_score**: 이미지 수준 이상 점수 (scalar)
- **anomaly_map**: 픽셀 수준 이상 맵 (H x W)
- **prediction**: 이진 분류 결과 (normal/anomaly)
- **confidence**: 예측 신뢰도

---

## 2. 모델 로드

### 2.1. 학습된 모델 로드

```python
import torch
from registry import get_trainer

# Initialize trainer
trainer = get_trainer(model_type="stfpm", img_size=256)

# Load trained weights
weight_path = "/path/to/outputs/mvtec/bottle/stfpm/model_mvtec_bottle_stfpm_epochs-50.pth"
trainer.load_model(weight_path)

# Set to evaluation mode
trainer.model.eval()
```

### 2.2. 모델 상태 확인

```python
# Check if model is loaded correctly
checkpoint = torch.load(weight_path, map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())
# Expected: ['model', 'optimizer', 'scheduler']

print(f"Model loaded from: {weight_path}")
print(f"Device: {trainer.device}")
print(f"Model mode: {'eval' if not trainer.model.training else 'train'}")
```

### 2.3. GPU/CPU 설정

```python
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize trainer with specific device
trainer = get_trainer(...)
trainer.model = trainer.model.to(device)
trainer.device = device

print(f"Using device: {device}")
```

---

## 3. 단일 이미지 추론

### 3.1. 기본 추론

```python
import torch
from PIL import Image
from torchvision import transforms
from registry import get_trainer

# Load model
trainer = get_trainer("stfpm", 256)
trainer.load_model("/path/to/model.pth")
trainer.model.eval()

# Prepare image
img_path = "/path/to/test_image.png"
img = Image.open(img_path).convert("RGB")

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Transform image
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
img_tensor = img_tensor.to(trainer.device)

# Inference
with torch.no_grad():
    output = trainer.model(img_tensor)

# Get results
pred_score = output['pred_score'].item()
anomaly_map = output['anomaly_map'].squeeze().cpu().numpy()

print(f"Anomaly score: {pred_score:.4f}")
print(f"Anomaly map shape: {anomaly_map.shape}")
```

### 3.2. 결과 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_result(image_path, anomaly_map, pred_score, threshold=0.5):
    """Visualize inference result"""
    # Load original image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Normalize anomaly map
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Anomaly map
    im = axes[1].imshow(anomaly_map_norm, cmap='jet')
    axes[1].set_title(f"Anomaly Map\nScore: {pred_score:.4f}")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img_array)
    axes[2].imshow(anomaly_map_norm, cmap='jet', alpha=0.5)
    prediction = "Anomaly" if pred_score >= threshold else "Normal"
    axes[2].set_title(f"Overlay\nPrediction: {prediction}")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("inference_result.png", dpi=150, bbox_inches='tight')
    plt.show()

# Usage
visualize_result(img_path, anomaly_map, pred_score, threshold=0.5)
```

---

## 4. 배치 추론

### 4.1. 다중 이미지 추론

```python
import os
from pathlib import Path
import pandas as pd

def batch_inference(trainer, image_dir, output_csv, threshold=0.5):
    """Perform inference on multiple images"""
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Get all images
    image_paths = list(Path(image_dir).glob("*.png")) + \
                  list(Path(image_dir).glob("*.jpg"))
    
    results = []
    
    trainer.model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            # Load and transform image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(trainer.device)
            
            # Inference
            output = trainer.model(img_tensor)
            pred_score = output['pred_score'].item()
            
            # Prediction
            prediction = "anomaly" if pred_score >= threshold else "normal"
            
            results.append({
                "filename": img_path.name,
                "score": pred_score,
                "prediction": prediction,
                "threshold": threshold
            })
            
            print(f"{img_path.name}: {pred_score:.4f} ({prediction})")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Summary
    n_anomaly = (df['prediction'] == 'anomaly').sum()
    n_normal = (df['prediction'] == 'normal').sum()
    print(f"\nSummary:")
    print(f"  Total: {len(df)}")
    print(f"  Normal: {n_normal} ({n_normal/len(df)*100:.1f}%)")
    print(f"  Anomaly: {n_anomaly} ({n_anomaly/len(df)*100:.1f}%)")
    
    return df

# Usage
results = batch_inference(
    trainer=trainer,
    image_dir="/path/to/unlabeled_images",
    output_csv="/path/to/results.csv",
    threshold=0.5
)
```

### 4.2. DataLoader를 사용한 배치 추론

```python
from torch.utils.data import Dataset, DataLoader

class InferenceDataset(Dataset):
    """Dataset for inference on unlabeled images"""
    
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.png")) + \
                          list(Path(image_dir).glob("*.jpg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return {
            "image": img,
            "filename": img_path.name,
            "path": str(img_path)
        }

def batch_inference_dataloader(trainer, image_dir, batch_size=8):
    """Batch inference using DataLoader"""
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = InferenceDataset(image_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    results = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(trainer.device)
            filenames = batch['filename']
            
            # Inference
            outputs = trainer.model(images)
            scores = outputs['pred_score'].cpu().numpy()
            
            for filename, score in zip(filenames, scores):
                results.append({
                    "filename": filename,
                    "score": float(score)
                })
    
    return pd.DataFrame(results)

# Usage
results = batch_inference_dataloader(
    trainer=trainer,
    image_dir="/path/to/unlabeled_images",
    batch_size=16
)
```

---

## 5. 레이블 없는 데이터 추론

### 5.1. 추론 전용 데이터셋 생성

레이블이 없는 데이터를 위한 디렉토리 구조:

```
inference_data/
└── images/
    ├── image_001.png
    ├── image_002.png
    ├── image_003.png
    └── ...
```

### 5.2. 추론 스크립트

```python
# inference.py
import argparse
import torch
from pathlib import Path
from registry import get_trainer

def main(args):
    # Load model
    print(f"Loading model: {args.model_type}")
    trainer = get_trainer(model_type=args.model_type, img_size=args.img_size)
    trainer.load_model(args.weight_path)
    trainer.model.eval()
    
    print(f"Model loaded from: {args.weight_path}")
    print(f"Device: {trainer.device}")
    
    # Perform inference
    print(f"\nPerforming inference on: {args.image_dir}")
    results = batch_inference(
        trainer=trainer,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        threshold=args.threshold
    )
    
    print(f"\nInference complete!")
    print(f"Results saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on unlabeled images")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, required=True,
                       help="Model type (e.g., stfpm, padim)")
    parser.add_argument("--weight_path", type=str, required=True,
                       help="Path to model weights")
    
    # Data arguments
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing unlabeled images")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Output CSV file path")
    
    # Inference arguments
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Anomaly threshold")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Input image size")
    
    # Path arguments
    parser.add_argument("--backbone_dir", type=str, default="/path/to/backbones")
    parser.add_argument("--dataset_dir", type=str, default="/path/to/datasets")
    
    args = parser.parse_args()
    main(args)
```

### 5.3. 실행 예시

```bash
# Run inference
python inference.py \
    --model_type stfpm \
    --weight_path /path/to/model.pth \
    --image_dir /path/to/unlabeled_images \
    --output_csv /path/to/results.csv \
    --threshold 0.5 \
    --img_size 256
```

---

## 6. 임계값 설정

### 6.1. 임계값 결정 방법

학습 후 생성된 `results_*_thresholds.txt` 파일에서 임계값을 확인할 수 있습니다.

```python
def load_threshold_from_results(result_file, method='f1'):
    """Load threshold from training results"""
    with open(result_file, 'r') as f:
        content = f.read()
    
    # Parse threshold
    method_map = {
        'f1': 'F1 (Percentile)',
        'f1_uniform': 'F1 (Uniform)',
        'roc': 'ROC (Youden J)',
        'percentile': 'Percentile (95%)'
    }
    
    search_str = method_map[method]
    for line in content.split('\n'):
        if search_str in line:
            threshold = float(line.split(':')[1].strip())
            return threshold
    
    return 0.5  # Default

# Usage
result_file = "/path/to/results_mvtec_bottle_stfpm_thresholds.txt"
threshold = load_threshold_from_results(result_file, method='f1')
print(f"Using threshold: {threshold:.4f}")
```

### 6.2. 임계값 방법 비교

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| F1 (Percentile) | F1 score 최대화 (백분위 샘플링) | 균형잡힌 성능 | 데이터 분포 의존 |
| F1 (Uniform) | F1 score 최대화 (균등 샘플링) | 전체 범위 고려 | 계산 비용 높음 |
| ROC (Youden J) | TPR - FPR 최대화 | 이론적 근거 명확 | 불균형 데이터에 민감 |
| Percentile (95%) | 정상 데이터 95% 분위 | 해석 용이 | False Positive 높을 수 있음 |

### 6.3. 커스텀 임계값 설정

```python
def find_optimal_threshold(scores, labels, metric='f1'):
    """Find optimal threshold for given metric"""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Generate candidate thresholds
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thr
    
    return best_threshold, best_score

# Usage with validation set
val_scores = []  # Collect from validation
val_labels = []  # Ground truth labels

threshold, f1 = find_optimal_threshold(
    np.array(val_scores),
    np.array(val_labels),
    metric='f1'
)

print(f"Optimal threshold: {threshold:.4f} (F1: {f1:.4f})")
```

---

## 7. 이상 맵 생성 및 저장

### 7.1. 이상 맵 저장

```python
import cv2

def save_anomaly_maps(trainer, image_dir, output_dir, threshold=0.5):
    """Generate and save anomaly maps for all images"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    image_paths = list(Path(image_dir).glob("*.png")) + \
                  list(Path(image_dir).glob("*.jpg"))
    
    trainer.model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            # Load image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(trainer.device)
            
            # Inference
            output = trainer.model(img_tensor)
            pred_score = output['pred_score'].item()
            anomaly_map = output['anomaly_map'].squeeze().cpu().numpy()
            
            # Normalize anomaly map
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / \
                              (anomaly_map.max() - anomaly_map.min() + 1e-8)
            
            # Convert to heatmap
            anomaly_map_colored = cv2.applyColorMap(
                (anomaly_map_norm * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Save
            prediction = "anomaly" if pred_score >= threshold else "normal"
            output_filename = f"{img_path.stem}_{prediction}_{pred_score:.3f}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, anomaly_map_colored)
            print(f"Saved: {output_filename}")

# Usage
save_anomaly_maps(
    trainer=trainer,
    image_dir="/path/to/images",
    output_dir="/path/to/anomaly_maps",
    threshold=0.5
)
```

### 7.2. 오버레이 시각화

```python
def create_overlay_visualization(image_path, anomaly_map, output_path, alpha=0.5):
    """Create overlay of original image and anomaly map"""
    
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize anomaly map to match image size
    h, w = img.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
    
    # Normalize
    anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / \
                       (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(
        (anomaly_map_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    
    # Save
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay_bgr)

# Usage
create_overlay_visualization(
    image_path="/path/to/image.png",
    anomaly_map=anomaly_map,
    output_path="/path/to/overlay.png",
    alpha=0.5
)
```

### 7.3. 세그멘테이션 마스크 생성

```python
def generate_segmentation_mask(anomaly_map, threshold=0.5):
    """Generate binary segmentation mask from anomaly map"""
    
    # Normalize
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / \
                       (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Threshold
    mask = (anomaly_map_norm >= threshold).astype(np.uint8) * 255
    
    return mask

def save_segmentation_masks(trainer, image_dir, output_dir, threshold=0.5):
    """Generate and save segmentation masks"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ... (similar to save_anomaly_maps)
    
    for img_path in image_paths:
        # ... (inference code)
        
        # Generate mask
        mask = generate_segmentation_mask(anomaly_map, threshold)
        
        # Save
        output_path = os.path.join(output_dir, f"{img_path.stem}_mask.png")
        cv2.imwrite(output_path, mask)

# Usage
save_segmentation_masks(
    trainer=trainer,
    image_dir="/path/to/images",
    output_dir="/path/to/masks",
    threshold=0.5
)
```

---

## 8. 배포 최적화

### 8.1. 모델 경량화

#### TorchScript 변환

```python
def convert_to_torchscript(trainer, output_path, img_size=256):
    """Convert model to TorchScript for deployment"""
    
    trainer.model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size).to(trainer.device)
    
    # Trace model
    traced_model = torch.jit.trace(trainer.model, example_input)
    
    # Save
    traced_model.save(output_path)
    print(f"TorchScript model saved to: {output_path}")
    
    return traced_model

# Usage
traced_model = convert_to_torchscript(
    trainer=trainer,
    output_path="/path/to/model_traced.pt",
    img_size=256
)

# Load and use
loaded_model = torch.jit.load("/path/to/model_traced.pt")
loaded_model.eval()

# Inference
with torch.no_grad():
    output = loaded_model(img_tensor)
```

#### ONNX 변환

```python
def convert_to_onnx(trainer, output_path, img_size=256):
    """Convert model to ONNX format"""
    
    trainer.model.eval()
    
    # Create example input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(trainer.device)
    
    # Export to ONNX
    torch.onnx.export(
        trainer.model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pred_score', 'anomaly_map'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'pred_score': {0: 'batch_size'},
            'anomaly_map': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to: {output_path}")

# Usage
convert_to_onnx(
    trainer=trainer,
    output_path="/path/to/model.onnx",
    img_size=256
)

# Load and use with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("/path/to/model.onnx")
input_name = session.get_inputs()[0].name

# Inference
outputs = session.run(None, {input_name: img_numpy})
```

### 8.2. 추론 속도 최적화

#### Mixed Precision

```python
from torch.cuda.amp import autocast

def inference_with_amp(trainer, img_tensor):
    """Inference with automatic mixed precision"""
    
    with torch.no_grad():
        with autocast():
            output = trainer.model(img_tensor)
    
    return output

# Usage
output = inference_with_amp(trainer, img_tensor)
```

#### Batch Processing

```python
def optimized_batch_inference(trainer, images, batch_size=32):
    """Optimized batch inference"""
    
    all_scores = []
    all_maps = []
    
    trainer.model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(trainer.device)
            
            outputs = trainer.model(batch_tensor)
            
            all_scores.extend(outputs['pred_score'].cpu().numpy())
            all_maps.extend(outputs['anomaly_map'].cpu().numpy())
    
    return all_scores, all_maps
```

---

## 9. 실시간 추론

### 9.1. 웹캠 추론

```python
import cv2

def realtime_inference_webcam(trainer, threshold=0.5):
    """Real-time inference from webcam"""
    
    cap = cv2.VideoCapture(0)
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    trainer.model.eval()
    
    print("Press 'q' to quit")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform
            img_tensor = transform(frame_rgb).unsqueeze(0).to(trainer.device)
            
            # Inference
            output = trainer.model(img_tensor)
            pred_score = output['pred_score'].item()
            
            # Display result
            prediction = "Anomaly" if pred_score >= threshold else "Normal"
            color = (0, 0, 255) if pred_score >= threshold else (0, 255, 0)
            
            cv2.putText(frame, f"{prediction}: {pred_score:.3f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 2)
            
            cv2.imshow('Anomaly Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
realtime_inference_webcam(trainer, threshold=0.5)
```

### 9.2. 비디오 파일 추론

```python
def process_video(trainer, video_path, output_path, threshold=0.5):
    """Process video file for anomaly detection"""
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    trainer.model.eval()
    frame_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert and transform
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(trainer.device)
            
            # Inference
            output = trainer.model(img_tensor)
            pred_score = output['pred_score'].item()
            
            # Annotate frame
            prediction = "Anomaly" if pred_score >= threshold else "Normal"
            color = (0, 0, 255) if pred_score >= threshold else (0, 255, 0)
            
            cv2.putText(frame, f"{prediction}: {pred_score:.3f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 2)
            cv2.putText(frame, f"Frame: {frame_count}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"\nVideo processing complete!")
    print(f"Output saved to: {output_path}")

# Usage
process_video(
    trainer=trainer,
    video_path="/path/to/input_video.mp4",
    output_path="/path/to/output_video.mp4",
    threshold=0.5
)
```

---

## 10. 모델 변환 및 내보내기

### 10.1. 모델 체크포인트 정리

```python
def export_clean_checkpoint(weight_path, output_path):
    """Export clean checkpoint (model weights only)"""
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Extract only model weights
    clean_checkpoint = {
        'model': checkpoint['model']
    }
    
    # Save
    torch.save(clean_checkpoint, output_path)
    
    print(f"Clean checkpoint saved to: {output_path}")
    
    # Compare sizes
    import os
    original_size = os.path.getsize(weight_path) / (1024 * 1024)
    clean_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Clean size: {clean_size:.2f} MB")
    print(f"Reduction: {(1 - clean_size/original_size)*100:.1f}%")

# Usage
export_clean_checkpoint(
    weight_path="/path/to/model_full.pth",
    output_path="/path/to/model_clean.pth"
)
```

### 10.2. 모델 정보 추출

```python
def extract_model_info(weight_path):
    """Extract and display model information"""
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    info = {
        "checkpoint_keys": list(checkpoint.keys()),
        "model_keys": len(checkpoint['model'].keys()) if 'model' in checkpoint else 0,
    }
    
    # Count parameters
    if 'model' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['model'].values())
        info['total_parameters'] = f"{total_params:,}"
        info['size_mb'] = f"{total_params * 4 / (1024 * 1024):.2f}"  # Assuming float32
    
    # Print info
    print("="*70)
    print("Model Information")
    print("="*70)
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    print("="*70)
    
    return info

# Usage
info = extract_model_info("/path/to/model.pth")
```

---

## 11. 프로덕션 배포

### 11.1. Flask API 서버

```python
# app.py
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

# Global model
trainer = None

def load_model():
    """Load model at startup"""
    global trainer
    trainer = get_trainer("stfpm", 256)
    trainer.load_model("/path/to/model.pth")
    trainer.model.eval()
    print("Model loaded successfully")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(trainer.device)
        
        # Inference
        with torch.no_grad():
            output = trainer.model(img_tensor)
        
        pred_score = float(output['pred_score'].item())
        threshold = 0.5
        prediction = "anomaly" if pred_score >= threshold else "normal"
        
        return jsonify({
            'score': pred_score,
            'prediction': prediction,
            'threshold': threshold
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
```

실행:

```bash
python app.py
```

테스트:

```bash
curl -X POST -F "image=@test_image.png" http://localhost:5000/predict
```

### 11.2. FastAPI 서버 (고성능)

```python
# fastapi_app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

app = FastAPI()

# Global model
trainer = None

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    global trainer
    trainer = get_trainer("stfpm", 256)
    trainer.load_model("/path/to/model.pth")
    trainer.model.eval()
    print("Model loaded successfully")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict endpoint"""
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(trainer.device)
        
        # Inference
        with torch.no_grad():
            output = trainer.model(img_tensor)
        
        pred_score = float(output['pred_score'].item())
        threshold = 0.5
        prediction = "anomaly" if pred_score >= threshold else "normal"
        
        return JSONResponse({
            'score': pred_score,
            'prediction': prediction,
            'threshold': threshold
        })
    
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({'status': 'healthy'})
```

실행:

```bash
pip install fastapi uvicorn python-multipart
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 11.3. Docker 배포

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

빌드 및 실행:

```bash
# Build image
docker build -t anomaly-detection-api .

# Run container
docker run -d -p 8000:8000 \
    -v /path/to/backbones:/app/backbones \
    -v /path/to/model:/app/model \
    --gpus all \
    anomaly-detection-api
```

---

## 12. 문제 해결

### 12.1. 추론 결과가 학습과 다름

**증상:** 학습 시에는 높은 AUROC를 보였지만 추론에서 성능이 낮음

**해결:**

1. 정규화 확인:
```python
# Check if normalization matches training
# normalize=True for most models
# normalize=False for Autoencoder, EfficientAD
```

2. 이미지 전처리 확인:
```python
# Ensure same preprocessing as training
transform = transforms.Compose([
    transforms.Resize(256),        # Same as training
    transforms.CenterCrop(256),    # Same as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

3. 모델 모드 확인:
```python
# Ensure evaluation mode
trainer.model.eval()

# Disable gradient computation
with torch.no_grad():
    output = trainer.model(img_tensor)
```

### 12.2. 메모리 오류

**증상:** Out of memory during inference

**해결:**

1. 배치 크기 감소:
```python
# Process images one by one
for img_path in image_paths:
    # Process single image
    pass
```

2. 명시적 메모리 정리:
```python
import gc

# After inference
del output
torch.cuda.empty_cache()
gc.collect()
```

### 12.3. 느린 추론 속도

**증상:** Inference takes too long

**해결:**

1. GPU 사용 확인:
```python
print(f"Using device: {trainer.device}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

2. 배치 처리:
```python
# Process multiple images at once
batch_size = 16  # Adjust based on memory
```

3. TorchScript 사용:
```python
# Convert to TorchScript for faster inference
traced_model = torch.jit.trace(trainer.model, example_input)
```
