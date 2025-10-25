# BaseTrainer 설계 문서

## 📋 목차
1. [개요](#개요)
2. [핵심 특징](#핵심-특징)
3. [구조 및 아키텍처](#구조-및-아키텍처)
4. [설계 원칙](#설계-원칙)
5. [주요 컴포넌트](#주요-컴포넌트)
6. [Hook 시스템](#hook-시스템)
7. [확장 방법](#확장-방법)
8. [향후 개선 사항](#향후-개선-사항)

---

## 개요

`BaseTrainer`는 PyTorch 기반의 추상 학습 프레임워크로, Template Method 패턴을 활용하여 다양한 딥러닝 모델의 학습 과정을 표준화하고 확장 가능하게 설계되었습니다.

### 주요 목표
- **재사용성**: 공통 학습 로직을 한 곳에 집중
- **확장성**: 다양한 모델/태스크에 쉽게 적용
- **유지보수성**: 명확한 구조와 일관된 인터페이스
- **연속 학습**: Checkpoint를 통한 학습 재개 지원

---

## 핵심 특징

### 1. **Template Method 패턴**
```python
class BaseTrainer(ABC):
    @abstractmethod
    def training_step(self, batch, batch_idx):
        """서브클래스에서 구현 필수"""
        raise NotImplementedError
    
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """서브클래스에서 구현 필수"""
        raise NotImplementedError
```

### 2. **Hook 시스템**
학습 과정의 각 단계에 커스텀 로직 삽입 가능
```python
def on_train_epoch_start(self): pass
def on_train_batch_end(self, outputs, batch, batch_idx): pass
def on_validation_epoch_end(self, outputs): pass
# ... 총 8개의 hook 메서드
```

### 3. **연속 학습 지원**
```python
self.global_epoch = 0    # 누적 총 epoch 수
self.global_step = 0     # 누적 총 step 수
self.history = {}        # 학습 이력 누적
```

### 4. **유연한 구성**
```python
def configure_optimizers(self):
    return {
        'optimizer': optimizer,
        'scheduler': scheduler  # Optional
    }

def configure_early_stoppers(self):
    return {
        'train': EarlyStopper(...),  # Optional
        'valid': EarlyStopper(...)   # Optional
    }
```

---

## 구조 및 아키텍처

### 클래스 다이어그램
```
┌─────────────────────────────────────┐
│         BaseTrainer (ABC)           │
├─────────────────────────────────────┤
│ # 상태 변수                          │
│ - model                             │
│ - optimizer, scheduler              │
│ - global_epoch, global_step         │
│ - history                           │
│ - best_model_state                  │
├─────────────────────────────────────┤
│ # 추상 메서드 (구현 필수)            │
│ + training_step()                   │
│ + validation_step()                 │
├─────────────────────────────────────┤
│ # 구성 메서드 (Optional)             │
│ + configure_optimizers()            │
│ + configure_early_stoppers()        │
├─────────────────────────────────────┤
│ # Hook 메서드 (Override 가능)        │
│ + on_train_epoch_start()            │
│ + on_train_batch_end()              │
│ + on_validation_epoch_end()         │
│ + ...                               │
├─────────────────────────────────────┤
│ # 핵심 메서드                        │
│ + fit()                             │
│ - _train_epoch()                    │
│ - _validate_epoch()                 │
│ - _update_scheduler()               │
├─────────────────────────────────────┤
│ # 유틸리티                           │
│ + save_checkpoint()                 │
│ + load_checkpoint()                 │
│ + save_model()                      │
│ + load_model()                      │
└─────────────────────────────────────┘
           ▲
           │ 상속
           │
┌──────────┴──────────┐
│                     │
ClassificationTrainer  AnomalyTrainer
```

### 학습 플로우
```
fit()
  │
  ├─> on_fit_start()
  │     ├─> _setup_optimizers()
  │     └─> _setup_early_stoppers()
  │
  ├─> for epoch in range(1, num_epochs + 1):
  │     │
  │     ├─> on_train_epoch_start()
  │     │     └─> model.train()
  │     │
  │     ├─> _train_epoch(train_loader)
  │     │     ├─> for batch in train_loader:
  │     │     │     ├─> on_train_batch_start()
  │     │     │     ├─> training_step()  ← 구현 필수
  │     │     │     ├─> loss.backward()
  │     │     │     ├─> optimizer.step()
  │     │     │     ├─> global_step += 1
  │     │     │     └─> on_train_batch_end()
  │     │     └─> return averaged outputs
  │     │
  │     ├─> on_train_epoch_end()
  │     │     └─> _update_history()
  │     │
  │     ├─> _check_train_stopping()
  │     │
  │     ├─> if has_valid_loader:
  │     │     ├─> on_validation_epoch_start()
  │     │     │     └─> model.eval()
  │     │     │
  │     │     ├─> _validate_epoch(valid_loader)
  │     │     │     ├─> for batch in valid_loader:
  │     │     │     │     ├─> on_validation_batch_start()
  │     │     │     │     ├─> validation_step()  ← 구현 필수
  │     │     │     │     └─> on_validation_batch_end()
  │     │     │     └─> return averaged outputs
  │     │     │
  │     │     ├─> on_validation_epoch_end()
  │     │     │     └─> _update_history()
  │     │     │
  │     │     └─> _check_valid_stopping()
  │     │           └─> save best_model_state
  │     │
  │     └─> _update_scheduler()
  │           └─> log LR changes
  │
  └─> on_fit_end()
        └─> save best model
```

---

## 설계 원칙

### 1. **관심사의 분리 (Separation of Concerns)**
```python
# ✅ 좋은 예: 각 메서드가 명확한 책임
def _train_epoch(self, train_loader):
    """한 epoch의 학습만 담당"""
    # ...

def _validate_epoch(self, valid_loader):
    """한 epoch의 검증만 담당"""
    # ...

def _update_scheduler(self):
    """Scheduler 업데이트만 담당"""
    # ...
```

### 2. **개방-폐쇄 원칙 (Open-Closed Principle)**
```python
# ✅ 확장에는 열려있고, 수정에는 닫혀있음
class ClassificationTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        """새로운 기능 추가 (BaseTrainer 수정 불필요)"""
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        
        return dict(loss=loss, acc=acc)
```

### 3. **의존성 역전 원칙 (Dependency Inversion)**
```python
# ✅ 추상화에 의존
def __init__(self, model, loss_fn=None, device=None, logger=None):
    self.model = model              # 구체적인 모델이 아닌 nn.Module
    self.loss_fn = loss_fn          # 구체적인 loss가 아닌 callable
    self.logger = logger            # 구체적인 logger가 아닌 interface
```

### 4. **단일 책임 원칙 (Single Responsibility)**
```python
# ✅ 각 메서드가 하나의 명확한 책임
def _setup_optimizers(self):
    """Optimizer 설정만 담당"""

def _setup_early_stoppers(self):
    """Early Stopper 설정만 담당"""

def _format_time(self, seconds):
    """시간 포맷팅만 담당"""

def _update_scheduler(self):
    """Scheduler 업데이트만 담당"""
```

### 5. **Don't Repeat Yourself (DRY)**
```python
# ✅ 공통 로직은 한 곳에
def _update_history(self, outputs):
    """Train/Valid 모두에서 재사용"""
    history_key = 'train' if self.training else 'valid'
    for key, value in outputs.items():
        self.history[history_key].setdefault(key, [])
        self.history[history_key][key].append(value)
```

### 6. **Fail Fast**
```python
# ✅ 문제를 조기에 발견
def __init__(self, model, ...):
    if model is None:
        raise ValueError("Model must be provided")
    
def _check_valid_stopping(self, valid_outputs):
    monitor_key = self.valid_early_stopper.monitor
    if monitor_key not in valid_outputs:
        self.logger.error(f"Monitor key '{monitor_key}' not found")
        return False
```

---

## 주요 컴포넌트

### 1. **상태 변수**

#### **학습 진행 상태**
```python
self.epoch           # 현재 fit의 epoch (1~num_epochs)
self.global_epoch    # 누적 총 epoch 수 (0~∞)
self.global_step     # 누적 총 step 수 (0~∞)
self.training        # True: train mode, False: eval mode
```

#### **모델 관련**
```python
self.model              # 학습할 모델
self.device             # 학습 디바이스 (cuda/cpu)
self.loss_fn            # 손실 함수
self.best_model_state   # 최고 성능 모델의 state_dict
```

#### **최적화 관련**
```python
self.optimizer          # Optimizer (Optional)
self.scheduler          # LR Scheduler (Optional)
self.train_early_stopper  # Train Early Stopper (Optional)
self.valid_early_stopper  # Valid Early Stopper (Optional)
```

#### **기록 관련**
```python
self.history            # {'train': {...}, 'valid': {...}}
self.logger             # Logging 객체
self.output_dir         # 출력 디렉토리
self.run_name           # 실행 이름
```

### 2. **Early Stopper**
```python
class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3, mode='max', 
                 target_value=None, monitor='loss'):
        self.patience = patience      # 개선 없이 기다릴 epoch 수
        self.min_delta = min_delta    # 개선으로 인정할 최소 변화량
        self.mode = mode              # 'max' or 'min'
        self.target_value = target_value  # 목표값 (도달하면 조기 종료)
        self.monitor = monitor        # 모니터링할 metric 이름
```

**특징:**
- Patience 기반 조기 종료
- Target value 도달 시 종료
- Train/Valid 별도 설정 가능

### 3. **Logger 시스템**
```python
def set_logger(output_dir, run_name):
    """콘솔 + 파일 동시 로깅"""
    logger = logging.getLogger(run_name)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"{run_name}_training.log")
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    return logger
```

### 4. **History 시스템**
```python
self.history = {
    'train': {
        'loss': [0.5, 0.3, 0.2, ...],
        'acc': [0.8, 0.9, 0.95, ...]
    },
    'valid': {
        'loss': [0.6, 0.4, 0.3, ...],
        'acc': [0.75, 0.85, 0.92, ...]
    }
}
```

**특징:**
- Metric별 epoch 단위 기록
- Checkpoint에 저장되어 연속 학습 시 누적
- 학습 후 시각화/분석 용이

---

## Hook 시스템

### Hook 메서드 목록

| Hook | 호출 시점 | 용도 |
|------|----------|------|
| `on_fit_start()` | fit 시작 시 | 초기화, 로깅 |
| `on_fit_end()` | fit 종료 시 | 최종 저장, 요약 |
| `on_train_epoch_start()` | Train epoch 시작 | Model mode 설정 |
| `on_train_epoch_end(outputs)` | Train epoch 종료 | History 업데이트, 로깅 |
| `on_train_batch_start(batch, idx)` | Train batch 시작 | Batch 전처리 |
| `on_train_batch_end(outputs, batch, idx)` | Train batch 종료 | Gradient clipping, 로깅 |
| `on_validation_epoch_start()` | Valid epoch 시작 | Model mode 설정 |
| `on_validation_epoch_end(outputs)` | Valid epoch 종료 | History 업데이트, 로깅 |
| `on_validation_batch_start(batch, idx)` | Valid batch 시작 | Batch 전처리 |
| `on_validation_batch_end(outputs, batch, idx)` | Valid batch 종료 | 중간 결과 수집 |

### Hook 활용 예시

#### **1. Gradient Clipping**
```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    if self.optimizer is not None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

#### **2. 주기적인 Checkpoint 저장**
```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    if self.global_step % 1000 == 0 and self.global_step > 0:
        checkpoint_path = f"{self.output_dir}/checkpoint_step_{self.global_step}.pth"
        self.save_checkpoint(checkpoint_path)
```

#### **3. TensorBoard 로깅**
```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    self.writer.add_scalar('train/loss', outputs['loss'], self.global_step)
    self.writer.add_scalar('train/lr', 
                          self.optimizer.param_groups[0]['lr'], 
                          self.global_step)
```

#### **4. Learning Rate Warmup**
```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    warmup_steps = 1000
    if self.global_step < warmup_steps:
        lr_scale = self.global_step / warmup_steps
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_scale
```

#### **5. 커스텀 메트릭 수집**
```python
def on_validation_batch_end(self, outputs, batch, batch_idx):
    # 배치별 예측 결과 수집
    if not hasattr(self, 'all_predictions'):
        self.all_predictions = []
        self.all_targets = []
    
    self.all_predictions.append(outputs['predictions'])
    self.all_targets.append(batch['label'])

def on_validation_epoch_end(self, outputs):
    # Epoch 끝에 전체 결과로 메트릭 계산
    from sklearn.metrics import confusion_matrix
    
    predictions = torch.cat(self.all_predictions).cpu().numpy()
    targets = torch.cat(self.all_targets).cpu().numpy()
    
    cm = confusion_matrix(targets, predictions)
    self.logger.info(f"Confusion Matrix:\n{cm}")
    
    # 초기화
    self.all_predictions = []
    self.all_targets = []
```

---

## 확장 방법

### 1. **Classification Trainer 예시**
```python
class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, num_classes, loss_fn=None, logger=None):
        super().__init__(model, loss_fn, logger=logger)
        self.num_classes = num_classes
    
    def training_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        
        return dict(loss=loss, acc=acc)
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        
        return dict(loss=loss, acc=acc)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return dict(optimizer=optimizer, scheduler=scheduler)
    
    def configure_early_stoppers(self):
        return dict(
            valid=EarlyStopper(patience=10, mode='max', monitor='acc', target_value=0.99)
        )
```

### 2. **Anomaly Detection Trainer 예시**
```python
class AnomalyTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, logger=None):
        super().__init__(model, loss_fn, logger=logger)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        
        # Autoencoder
        reconstructed = self.model(images)
        loss = self.loss_fn(reconstructed, images)
        
        return dict(loss=loss)
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        
        reconstructed = self.model(images)
        loss = self.loss_fn(reconstructed, images)
        
        return dict(loss=loss)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
    
    def configure_early_stoppers(self):
        return dict(
            valid=EarlyStopper(patience=15, mode='min', monitor='loss')
        )
```

---

## 향후 개선 사항

### 1. ⭐ **Metric 기반 Scheduler 지원 (ReduceLROnPlateau)**

#### **현재 문제**
```python
# ❌ ReduceLROnPlateau는 metric 파라미터 필요
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
scheduler.step(valid_loss)  # metric 필요

# 현재는 단순히 scheduler.step()만 호출
def _update_scheduler(self):
    self.scheduler.step()  # ❌ metric 전달 안 됨
```

#### **개선 방안**
```python
def _update_scheduler(self, valid_outputs=None):
    if self.optimizer is None or self.scheduler is None:
        return
    
    old_lrs = [group['lr'] for group in self.optimizer.param_groups]
    
    # ReduceLROnPlateau 감지
    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if valid_outputs is not None and hasattr(self, 'valid_early_stopper'):
            monitor_key = self.valid_early_stopper.monitor
            if monitor_key in valid_outputs:
                metric = valid_outputs[monitor_key]
                self.scheduler.step(metric)
            else:
                self.logger.warning(
                    f"ReduceLROnPlateau: monitor key '{monitor_key}' not found"
                )
        else:
            self.logger.warning("ReduceLROnPlateau requires validation outputs")
    else:
        # 일반 epoch 기반 scheduler
        self.scheduler.step()
    
    new_lrs = [group['lr'] for group in self.optimizer.param_groups]
    
    # LR 변경 로깅
    lr_threshold = 1e-8
    if len(new_lrs) == 1:
        if abs(old_lrs[0] - new_lrs[0]) > lr_threshold:
            self.logger.info(f"Learning rate updated: {old_lrs[0]:.6f} → {new_lrs[0]:.6f}")
    else:
        for idx, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
            if abs(old_lr - new_lr) > lr_threshold:
                self.logger.info(f"Learning rate [group {idx}] updated: {old_lr:.6f} → {new_lr:.6f}")

# fit 메서드에서 호출 수정
def fit(self, ...):
    # ...
    if self.has_valid_loader:
        self.on_validation_epoch_start()
        valid_outputs = self._validate_epoch(valid_loader)
        self.on_validation_epoch_end(valid_outputs)
        if self._check_valid_stopping(valid_outputs):
            break
    
    self._update_scheduler(valid_outputs if self.has_valid_loader else None)
```

#### **사용 예시**
```python
def configure_optimizers(self):
    optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    return dict(optimizer=optimizer, scheduler=scheduler)

def configure_early_stoppers(self):
    return dict(
        valid=EarlyStopper(patience=10, mode='min', monitor='loss')
    )

# 출력:
# [ 15/50] loss:0.023 | (val) loss:0.019 (2m 15s)
# Learning rate updated: 0.001000 → 0.000500
```

---

### 2. ⭐ **Gradient Clipping**

#### **구현 방안 1: Hook 활용 (권장)**
```python
class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, num_classes, grad_clip_norm=None, ...):
        super().__init__(model, ...)
        self.grad_clip_norm = grad_clip_norm
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.grad_clip_norm is not None and self.optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.grad_clip_norm
            )
```

#### **구현 방안 2: BaseTrainer에 통합**
```python
class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None, 
                 grad_clip_norm=None, grad_clip_value=None):
        # ...
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
    
    def _train_epoch(self, train_loader):
        # ...
        for batch_idx, batch in enumerate(progress_bar):
            # ...
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                outputs = self.training_step(batch, batch_idx)
                loss = outputs.get('loss')
                
                if loss is not None:
                    loss.backward()
                    
                    # ✅ Gradient Clipping
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=self.grad_clip_norm
                        )
                    if self.grad_clip_value is not None:
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(), 
                            clip_value=self.grad_clip_value
                        )
                    
                    self.optimizer.step()
                self.global_step += 1
```

#### **사용 예시**
```python
# 방법 1: 생성자에서 설정
trainer = ClassificationTrainer(
    model, 
    num_classes=10, 
    grad_clip_norm=1.0,  # ✅ Max gradient norm
    logger=logger
)

# 방법 2: Hook 오버라이드
class MyTrainer(BaseTrainer):
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Gradient norm 계산 및 로깅
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 10.0:
            self.logger.warning(f"Large gradient norm: {total_norm:.2f}")
        
        # Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

---

### 3. ⭐ **Mixed Precision Training (AMP)**

#### **구현 방안**
```python
class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 use_amp=False):
        # ...
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    def _train_epoch(self, train_loader):
        accumulated_outputs = {}
        total_images = 0
        
        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f"{self.epoch_info} Training")
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                total_images += batch_size
                
                self.on_train_batch_start(batch, batch_idx)
                
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    
                    # ✅ Mixed Precision Context
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        outputs = self.training_step(batch, batch_idx)
                        loss = outputs.get('loss')
                    
                    if loss is not None:
                        if self.use_amp:
                            # ✅ Scaled backward
                            self.scaler.scale(loss).backward()
                            
                            # ✅ Gradient clipping (optional)
                            if self.grad_clip_norm is not None:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    max_norm=self.grad_clip_norm
                                )
                            
                            # ✅ Scaled optimizer step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # 일반 backward
                            loss.backward()
                            
                            if self.grad_clip_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    max_norm=self.grad_clip_norm
                                )
                            
                            self.optimizer.step()
                    
                    self.global_step += 1
                else:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        outputs = self.training_step(batch, batch_idx)
                
                # outputs의 tensor를 item()으로 변환
                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    accumulated_outputs.setdefault(name, 0.0)
                    accumulated_outputs[name] += value * batch_size
                
                progress_bar.set_postfix({
                    name: f"{total_value / total_images:.3f}"
                    for name, total_value in accumulated_outputs.items()
                })
                self.on_train_batch_end(outputs, batch, batch_idx)
        
        return {name: value / total_images for name, value in accumulated_outputs.items()}
    
    @torch.no_grad()
    def _validate_epoch(self, valid_loader):
        accumulated_outputs = {}
        total_images = 0
        
        with tqdm(valid_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f"{self.epoch_info} Validation")
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                total_images += batch_size
                
                self.on_validation_batch_start(batch, batch_idx)
                
                # ✅ Validation도 AMP 사용
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.validation_step(batch, batch_idx)
                
                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    accumulated_outputs.setdefault(name, 0.0)
                    accumulated_outputs[name] += value * batch_size
                
                progress_bar.set_postfix({
                    name: f"{total_value / total_images:.3f}"
                    for name, total_value in accumulated_outputs.items()
                })
                self.on_validation_batch_end(outputs, batch, batch_idx)
        
        return {name: value / total_images for name, value in accumulated_outputs.items()}
    
    def save_checkpoint(self, filepath):
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "history": self.history,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()  # ✅ Scaler 저장
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath, strict=True):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.global_step = checkpoint.get("global_step", 0)
        self.global_epoch = checkpoint.get("global_epoch", 0)
        self.history = checkpoint.get("history", {"train": {}, "valid": {}})
        
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])  # ✅ Scaler 로드
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        self.logger.info(f"Resumed from global_epoch: {self.global_epoch}, global_step: {self.global_step}")
```

#### **사용 예시**
```python
# ✅ Mixed Precision 활성화
trainer = ClassificationTrainer(
    model,
    num_classes=10,
    loss_fn=loss_fn,
    logger=logger,
    use_amp=True  # ✅ AMP 활성화
)

trainer.fit(train_loader, num_epochs=50, valid_loader=valid_loader)

# 장점:
# 1. 메모리 사용량 ~50% 감소
# 2. 학습 속도 ~2x 향상 (GPU dependent)
# 3. 정확도는 거의 동일
```

---

### 4. ⭐ **Gradient Accumulation**

#### **구현 방안**
```python
class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 accumulation_steps=1):
        # ...
        self.accumulation_steps = accumulation_steps
    
    def _train_epoch(self, train_loader):
        accumulated_outputs = {}
        total_images = 0
        
        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f"{self.epoch_info} Training")
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                total_images += batch_size
                
                self.on_train_batch_start(batch, batch_idx)
                
                if self.optimizer is not None:
                    # ✅ accumulation_steps 번째 batch마다만 zero_grad
                    if batch_idx % self.accumulation_steps == 0:
                        self.optimizer.zero_grad()
                    
                    outputs = self.training_step(batch, batch_idx)
                    loss = outputs.get('loss')
                    
                    if loss is not None:
                        # ✅ Loss scaling
                        scaled_loss = loss / self.accumulation_steps
                        scaled_loss.backward()
                        
                        # ✅ accumulation_steps 번째 batch마다만 optimizer step
                        if (batch_idx + 1) % self.accumulation_steps == 0:
                            if self.grad_clip_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    max_norm=self.grad_clip_norm
                                )
                            self.optimizer.step()
                            self.global_step += 1
                else:
                    outputs = self.training_step(batch, batch_idx)
                
                # ...
```

#### **사용 예시**
```python
# ✅ Effective batch size = 32 * 4 = 128
trainer = ClassificationTrainer(
    model,
    num_classes=10,
    loss_fn=loss_fn,
    logger=logger,
    accumulation_steps=4  # ✅ 4 batch마다 update
)

train_loader = DataLoader(dataset, batch_size=32)  # 실제 batch size는 32
trainer.fit(train_loader, num_epochs=50)

# 효과:
# - 메모리는 batch_size=32만 사용
# - 학습 효과는 batch_size=128과 동일
```

---

### 5. ⭐ **Distributed Training (DDP)**

#### **구현 방안**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 distributed=False, local_rank=0):
        self.distributed = distributed
        self.local_rank = local_rank
        
        if distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            model = model.to(self.device)
            self.model = DDP(model, device_ids=[local_rank])
        else:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
        
        # ...
    
    def _train_epoch(self, train_loader):
        # DDP에서는 sampler의 epoch 설정 필요
        if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.global_epoch)
        
        # ...
    
    def save_checkpoint(self, filepath):
        # DDP에서는 rank 0만 저장
        if self.distributed and self.local_rank != 0:
            return
        
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # DDP의 경우 model.module.state_dict() 사용
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "history": self.history,
        }
        # ...
```

#### **사용 예시**
```python
# launch script: torchrun --nproc_per_node=4 train.py

import torch.distributed as dist

def main():
    # DDP 초기화
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # DistributedSampler 사용
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,  # ✅ Shuffle 대신 sampler
        num_workers=4,
        pin_memory=True
    )
    
    # Trainer 생성
    trainer = ClassificationTrainer(
        model,
        num_classes=10,
        loss_fn=loss_fn,
        logger=logger,
        distributed=True,
        local_rank=local_rank
    )
    
    trainer.fit(train_loader, num_epochs=50, valid_loader=valid_loader)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

---

### 6. ⭐ **EMA (Exponential Moving Average)**

#### **구현 방안**
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 use_ema=False, ema_decay=0.999):
        # ...
        self.use_ema = use_ema
        self.ema = EMA(self.model, decay=ema_decay) if use_ema else None
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update()
    
    def _validate_epoch(self, valid_loader):
        # Validation 시 EMA 모델 사용
        if self.ema is not None:
            self.ema.apply_shadow()
        
        accumulated_outputs = {}
        # ... validation logic
        
        if self.ema is not None:
            self.ema.restore()
        
        return accumulated_outputs
```

---

### 7. ⭐ **Progress Callback & Metrics Tracking**

#### **구현 방안**
```python
class Callback:
    def on_fit_start(self, trainer): pass
    def on_fit_end(self, trainer): pass
    def on_epoch_start(self, trainer): pass
    def on_epoch_end(self, trainer, outputs): pass
    def on_batch_end(self, trainer, outputs): pass

class TensorBoardCallback(Callback):
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def on_batch_end(self, trainer, outputs):
        if trainer.training:
            for key, value in outputs.items():
                self.writer.add_scalar(f'train/{key}', value, trainer.global_step)
            self.writer.add_scalar('train/lr', 
                                 trainer.optimizer.param_groups[0]['lr'], 
                                 trainer.global_step)
    
    def on_epoch_end(self, trainer, outputs):
        for key, value in outputs.items():
            phase = 'train' if trainer.training else 'valid'
            self.writer.add_scalar(f'{phase}/{key}', value, trainer.global_epoch)

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 callbacks=None):
        # ...
        self.callbacks = callbacks or []
    
    def on_fit_start(self):
        # ...
        for callback in self.callbacks:
            callback.on_fit_start(self)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        for callback in self.callbacks:
            callback.on_batch_end(self, outputs)

# 사용 예시
trainer = ClassificationTrainer(
    model,
    callbacks=[
        TensorBoardCallback('./runs/experiment1'),
        WandBCallback(project='my-project'),
        CheckpointCallback(save_every=5)
    ]
)
```

---

### 8. ⭐ **Model Profiling & Debugging**

#### **구현 방안**
```python
class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 profile=False):
        # ...
        self.profile = profile
    
    def _train_epoch(self, train_loader):
        if self.profile:
            from torch.profiler import profile, ProfilerActivity
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            ) as prof:
                # 첫 몇 batch만 프로파일링
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx >= 10:
                        break
                    # ... training logic
            
            # 프로파일링 결과 저장
            prof.export_chrome_trace(f"{self.output_dir}/profile_trace.json")
            self.logger.info(f"Profile saved to {self.output_dir}/profile_trace.json")
        
        # 일반 학습 진행
        accumulated_outputs = {}
        # ...
```

---

### 9. ⭐ **Auto Resume from Last Checkpoint**

#### **구현 방안**
```python
class BaseTrainer(ABC):
    def fit(self, train_loader, num_epochs, valid_loader=None, 
            output_dir=None, run_name=None, auto_resume=True):
        self.has_valid_loader = valid_loader is not None
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.run_name = run_name or "best_model"
        
        # ✅ Auto resume
        if auto_resume and output_dir is not None:
            last_checkpoint = self._find_last_checkpoint()
            if last_checkpoint is not None:
                self.logger.info(f"Found checkpoint: {last_checkpoint}")
                self.load_checkpoint(last_checkpoint)
        
        self.on_fit_start()
        # ...
    
    def _find_last_checkpoint(self):
        """마지막 checkpoint 찾기"""
        if not os.path.exists(self.output_dir):
            return None
        
        checkpoints = [
            f for f in os.listdir(self.output_dir)
            if f.startswith('checkpoint_') and f.endswith('.pth')
        ]
        
        if not checkpoints:
            return None
        
        # Epoch 번호로 정렬
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(self.output_dir, checkpoints[-1])
```

---

### 10. ⭐ **Configurable Batch Dictionary Keys**

#### **현재 문제**
```python
# ❌ batch["image"]와 batch["label"]이 하드코딩됨
batch_size = batch["image"].shape[0]
```

#### **개선 방안**
```python
class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None,
                 batch_image_key='image', batch_size_dim=0):
        # ...
        self.batch_image_key = batch_image_key
        self.batch_size_dim = batch_size_dim
    
    def _train_epoch(self, train_loader):
        accumulated_outputs = {}
        total_images = 0
        
        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                # ✅ Configurable key
                batch_size = batch[self.batch_image_key].shape[self.batch_size_dim]
                total_images += batch_size
                # ...

# 사용 예시
trainer = ClassificationTrainer(
    model,
    batch_image_key='input',  # ✅ 'image' 대신 'input' 사용
    batch_size_dim=0
)
```

---

## 📊 개선 우선순위

| 순위 | 항목 | 중요도 | 난이도 | 영향 범위 |
|------|------|--------|--------|----------|
| 1 | Gradient Clipping | ⭐⭐⭐ | 낮음 | 학습 안정성 |
| 2 | Mixed Precision (AMP) | ⭐⭐⭐ | 중간 | 메모리, 속도 |
| 3 | Metric 기반 Scheduler | ⭐⭐⭐ | 낮음 | LR 조정 |
| 4 | Gradient Accumulation | ⭐⭐ | 중간 | 메모리 |
| 5 | Callback System | ⭐⭐ | 중간 | 확장성 |
| 6 | Auto Resume | ⭐⭐ | 낮음 | 편의성 |
| 7 | EMA | ⭐⭐ | 중간 | 성능 |
| 8 | Distributed Training | ⭐ | 높음 | 속도 |
| 9 | Model Profiling | ⭐ | 낮음 | 디버깅 |
| 10 | Configurable Keys | ⭐ | 낮음 | 유연성 |

---

## ✅ 요약

### **현재 BaseTrainer의 강점**
1. ✅ 명확한 추상화와 확장 인터페이스
2. ✅ Hook 시스템으로 커스터마이징 용이
3. ✅ 연속 학습 지원 (checkpoint, history)
4. ✅ Early stopping 지원
5. ✅ Scheduler 통합 및 LR 변경 로깅
6. ✅ 깔끔한 로깅 시스템

### **핵심 개선 필요 사항**
1. 🔧 Metric 기반 Scheduler (ReduceLROnPlateau)
2. 🔧 Gradient Clipping
3. 🔧 Mixed Precision Training
4. 🔧 Gradient Accumulation
5. 🔧 Callback System

이러한 개선을 통해 BaseTrainer는 더욱 강력하고 실무에서 바로 사용 가능한 프레임워크가 될 것입니다! 🎉