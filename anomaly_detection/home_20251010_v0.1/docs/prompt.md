#### OLED 디스플레이 화질 검사에 현재 작성된 이상감지 프레임워크를 적용할려고 합니다.
- 기존 작성된 데이터로더와 연관하여, root_dir / dataset_type / category 네이밍을 그대로 사용
- root_dir: OLED 제품별 화질 검사시 측정된 이미지가 저장되는 상위 폴더이고, data 폴더에 이미지들이 저장됨
- dataset_type: 해상도가 다른 서로 다른 OLED 제품 이름 (ex. module1, module2, ...)
- category: 화질 검사에 사용하는 패턴 이미지 종류 (ex. pattern1, pattern2, ...)
- 측정 데이터 형식: module1\data_rgb\good or anomaly\pattern1 {frequency} {dimming}.png (ex. parttern1 120 200.png) - 빈칸으로 구분
- 파일경로 및 이름으로부터 get_info(image_path)
- info["freq"], info["dimming"], info["pattern"],  info["module"], info["label"] = None or 1 o r 0

데이터셋 및 데이터로더 정의부터 시작해야 합니다.

#### 학습/추론 데이터 구조
- data_rgb 폴더에 모든 이미지가 저장되어 있음 (실시간 저장 아님)
- data_rgb에 사용자가 good 과 anomaly 폴더로 이미지를 분리해서 저장함
- 학습데이터: data_rgb\good\*.png 의 일부 (예 80%)
- 검증 데이터: data_rgb\good\*.png 의 일부 (예 20%) + data_rgb\anomaly\*.png (100%)
- 메타데이터 파일을 생성해야 함 data_rgb\{data_type}.csv
- 레이블 정보는 사용자가 확인하여 작성된 csv 항목에 추가함

#### Category(Pattern) 처리
- 기존 프레임워크와 동일하게 category(= pattern)별로 모델 학습
- pattern별로 별도 모델 생성 / 학습 / 평가

```python
train(dataset_type="module1", category="pattern1", ...)
train(dataset_type="module1", category="pattern2", ...)
```

#### 이미지 네이밍 규칙
- 파일명 형식이 항상 {pattern} {freq} {dimming}.png​ 으로 정해져 있음
- 빈칸으로 pattern / freq / dimming 을 구분하고, 파일 확장자는 png 로 고정됨
- pattern 이름에는 빈칸이 포함되어 있지 않음

#### VAD (Vision Anomaly Detection) Framework
```
vad/
└── models/
    └── components/
    │   ├── pattern2 60 100.png
    │   ├── 
    │   │   pattern1 120 200.png
        ├── defect1/
        │   ├── pattern1 120 180.png
        │   └── pattern2 60 80.png
        ├── defect2/
        │   ├── pattern1 120 100.png
        │   └── pattern2 60 20.png
        └── data_info.csv
```

#### 데이터 구조
```
root_dir/
└── module1/                          # dataset_type
    └── data_rgb/                     # 고정 폴더 (이름 변하지 않음)
        ├── normal/                     # 정상 이미지 (defect_type = normal)
        │   ├── pattern1 120 200.png
        │   └── pattern2 60 100.png
        ├── defect1/                  # 이상 이미지 (defect_type = defect1)
        │   ├── pattern1 120 180.png
        │   └── pattern2 60 80.png
        ├── defect2/                  # 이상 이미지 (defect_type = defect2)
        │   ├── pattern1 120 100.png
        │   └── pattern2 60 20.png
        └── data_info.csv             # 전체 메타데이터 (data_rgb 폴더내 *.png 이미지 정보를 읽어서 생성)
```

#### CSV 메타데이터 형식: 파일경로/파일명 root_dir/module1/data_rgb/data_info.csv
filename, pattern, freq, dimming, label, defect_type, image_path
pattern1 120 200.png, pattern1, 120, 200, 0, normal, root_dir/module1/data_rgb/normal/pattern1 120 200.png
pattern2 60 100.png, pattern2, 60, 100, 0, normal, root_dir/module1/data_rgb/normal/pattern2 60 100.png
pattern1 120 180.png, pattern1, 120, 180, 1, defect1, root_dir/module1/data_rgb/defect1/pattern1 120 180.png
pattern2 60 80.png, pattern2, 60, 80, 1, defect1, root_dir/module1/data_rgb/defect1/pattern2 60 80.png
pattern1 120 100.png, pattern1, 120, 100, 1, defect2, root_dir/module1/data_rgb/defect2/pattern1 120 100.png
pattern2 60 20.png, pattern2, 60, 20, 1, defect2, root_dir/module1/data_rgb/defect2/pattern2 60 20.png

#### CustomDataset:
```
dataset_dir/
├── module1/
│   └── data_rgb/
│       ├── good/
│       │   ├── grid 120 200_000.png
│       │   └── pattern1 120 200_000.png
│       └── defect1/
├── module2/
│   └── data_rgb/
│       ├── good/
│       └── defect1/
└── module3/
    └── data_rgb/
```
- module1, module2, module3는 물리적으로만 분리
- 동일 category(pattern)는 여러 module에 걸쳐 존재 가능
- category별로 학습하되, 모든 module의 데이터를 합쳐서 사용