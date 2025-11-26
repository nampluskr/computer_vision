class CelebA(Dataset):
    ATTRIBUTES = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
        "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
        "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
        "Wearing_Necktie", "Young"
    ]
    
    def __init__(self, root_dir, split, attributes=None, transform=None):
        self.split = split
        self.attributes = attributes or []
        self.transform = transform or T.Compose([T.ToTensor()])
        
        image_dir = os.path.join(root_dir, "img_align_celeba", "img_align_celeba")
        split_map = {"train": 0, "valid": 1, "test": 2}
        
        split_df = pd.read_csv(os.path.join(root_dir, "list_eval_partition.csv"))
        attr_df = pd.read_csv(os.path.join(root_dir, "list_attr_celeba.csv"))
        attr_df[self.ATTRIBUTES] = (attr_df[self.ATTRIBUTES] == 1).astype(int)
        df = split_df.merge(attr_df[["image_id"] + self.ATTRIBUTES], on="image_id", how="inner")
        
        if split == "all":
            selected_df = df
        elif split in split_map:
            selected_df = df[df["partition"] == split_map[split]]
        else:
            raise ValueError("split must be 'train', 'valid', 'test', or 'all'")
        
        # 조건 필터링 (예: Male=1 and Smiling=1)
        for attr in self.attributes:
            # 예: self.<attr>_value = 1 또는 0 (필터링 조건 저장용, 예제에선 생략)
            # 실제선 외부에서 조건을 인자로 받는 방식으로 확장 가능
            pass  # 조건 필터링은 외부에서 필터링된 df 전달하거나 별도 인자로 처리 가능
        
        self.image_paths = [os.path.join(image_dir, name) for name in selected_df["image_id"].tolist()]
        self.labels = selected_df[self.ATTRIBUTES].values.astype(int).tolist()  # 다중 레이블
        self.selected_attributes = self.ATTRIBUTES
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        
        label = torch.tensor(self.labels[idx]).long()
        class_names = [self.selected_attributes[i] for i, val in enumerate(label) if val == 1]
        if not class_names:
            class_names = ["None"]
        
        return dict(image=image, label=label, class_names=class_names)

    def filter_by_attributes(self, **conditions):
        indices = []
        for idx in range(len(self)):
            label_dict = {attr: self.labels[idx][i] for i, attr in enumerate(self.selected_attributes)}
            if all(label_dict.get(k, -1) == v for k, v in conditions.items()):
                indices.append(idx)
        return Subset(self, indices)


class LegoBricks(Dataset):
    def __init__(self, root_dir, split, transform=None, test_ratio=0.2, random_state=42):
        self.transform = transform or T.Compose([T.ToTensor()])

        image_paths = glob(os.path.join(root_dir, "dataset", "*.png"))
        shuffler = random.Random(random_state)
        shuffler.shuffle(image_paths)
        num_test = int(len(image_paths) * test_ratio)

        if split == "train":
            self.image_paths = image_paths[num_test:]
        elif split in "test":
            self.image_paths = image_paths[:num_test]
        elif split == "all":
            self.image_paths = image_paths[:]
        else:
            raise ValueError("split must be 'train', 'test', or 'all'")

        classnames = [self._get_classname(p) for p in self.image_paths]
        self.CLASS_NAMES = sorted(set(classnames))

        name_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        self.labels = [name_to_idx[name] for name in classnames]

        self.split = split
        self.test_ratio = test_ratio
        self.random_state = random_state

    def _get_classname(self, image_path):
        stem = os.path.basename(image_path)
        parts = os.path.splitext(stem)[0].split()
        return ' '.join(parts[1:-1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        class_name = self.CLASS_NAMES[label.item()]
        return dict(image=image, label=label, class_name=class_name)
