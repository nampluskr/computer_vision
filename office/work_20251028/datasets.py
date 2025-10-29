import os
import numpy as np
import gzip
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.data._utils.collate import default_collate


def get_train_loader(dataset, batch_size, collate_fn=None):
    return DataLoader(dataset, batch_size, shuffle=True, # drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=False,
        collate_fn=collate_fn if collate_fn is not None else default_collate)


def get_test_loader(dataset, batch_size, collate_fn=None):
    return DataLoader(dataset, batch_size, shuffle=False, # drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=False,
        collate_fn=collate_fn if collate_fn is not None else default_collate)


class MNIST(Dataset):
    CLASS_NAMES = [str(i) for i in range(10)]

    def __init__(self, root_dir, split, transform=None):
        self.num_classes = len(self.CLASS_NAMES)
        self.transform = transform or T.Compose([T.ToPILImage(), T.ToTensor()])

        if split == "train":
            images_path = os.path.join(root_dir, 'train-images-idx3-ubyte.gz')
            labels_path = os.path.join(root_dir, 'train-labels-idx1-ubyte.gz')
        elif split == "test":
            images_path = os.path.join(root_dir, 't10k-images-idx3-ubyte.gz')
            labels_path = os.path.join(root_dir, 't10k-labels-idx1-ubyte.gz')
        else:
            raise ValueError("split must be 'train' or 'test'")

        with gzip.open(images_path, 'rb') as file:
            self.images = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
            self.images = self.images.reshape(-1, 28, 28)

        with gzip.open(labels_path, 'rb') as file:
            self.labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()
        class_name = self.CLASS_NAMES[label.item()]
        return dict(image=image, label=label, class_name=class_name)


class FashionMNIST(MNIST):
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]


class CIFAR10(Dataset):
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root_dir, split, transform=None):
        self.num_classes = len(self.CLASS_NAMES)
        # self.transform = transform or T.Compose([T.ToPILImage(), T.ToTensor()])
        self.images = []
        self.labels = []

        data_dir = os.path.join(root_dir, "cifar-10-batches-py")
        if split == "train":
            data_batches = [f"data_batch_{i}" for i in range(1, 6)]
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif split == "test":
            data_batches = ["test_batch"]
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise ValueError("split must be 'train' or 'test'")

        for batch_name in data_batches:
            batch_path = os.path.join(data_dir, batch_name)
            with open(batch_path, "rb") as file:
                batch = pickle.load(file, encoding='bytes')
                self.images.append(batch[b'data'])
                self.labels.extend(batch[b'labels'])

        self.images = np.vstack(self.images)
        self.images = self.images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx]).long()
        class_name = self.CLASS_NAMES[label.item()]
        return dict(image=image, label=label, class_name=class_name)


class OxfordPets(Dataset):
    CLASS_NAMES = [
        'Abyssinian', 'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound',
        'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_Shorthair',
        'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter',
        'German_Shepherd', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin',
        'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher',
        'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll',
        'Russian_Blue', 'Saint_Bernard', 'Samoyed', 'Scottish_Terrier',
        'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
        'Wheaten_Terrier', 'Yorkshire_Terrier'
    ]
    BINARY_CLASS_NAMES = ['cat', 'dog']

    def __init__(self, root_dir, split, transform=None, **kwargs):
        self.num_classes = len(self.CLASS_NAMES)
        self.transform = transform or T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_paths = []
        self.labels = []
        self.binary_labels = []

        if split == "train":
            split_file = os.path.join(root_dir, "annotations", "trainval.txt")
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(0.3),
                # T.RandomVerticalFlip(0.3),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif split == "test":
            split_file = os.path.join(root_dir, "annotations", "test.txt")
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError("split must be 'train' or 'test'")

        # https://github.com/tensorflow/models/issues/3134
        images_png = [
            "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
            "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
            "Abyssinian_5", "Abyssinian_34",
        ]
        images_corrupt = ["chihuahua_121", "beagle_116"]

        with open(split_file, 'r') as file:
            for line in file:
                filename, label_str, *_ = line.strip().split()
                if filename in images_png + images_corrupt : continue

                image_path = os.path.join(root_dir, "images", f"{filename}.jpg")
                label = int(label_str) - 1  # 1-based -> 0-based
                binary_label = 0 if filename[0].isupper() else 1

                self.image_paths.append(image_path)
                self.labels.append(label)
                self.binary_labels.append(binary_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        class_name = self.CLASS_NAMES[label.item()]
        binary_label = torch.tensor(self.binary_labels[idx]).long()
        binary_class_name = self.BINARY_CLASS_NAMES[binary_label.item()]
        return dict(image=image, label=label, class_name=class_name,
            binary_label=binary_label, binary_class_name=binary_class_name)


if __name__ == "__main__":

    if 1:
        print("\n*** MNIST Dataset ***")
        root_dir = "/home/namu/myspace/NAMU/datasets/mnist"
        train_loader = get_train_loader(dataset=MNIST(root_dir, "train"), batch_size=64)
        test_loader = get_test_loader(dataset=MNIST(root_dir, "test"), batch_size=32)

        batch = next(iter(train_loader))
        images, labels = batch["image"], batch["label"]
        print(images.shape, images.dtype, images.min(), images.max())
        print(labels.shape, labels.dtype, labels.min(), labels.max())

        batch = next(iter(test_loader))
        images, labels = batch["image"], batch["label"]
        print(images.shape, images.dtype, images.min(), images.max())
        print(labels.shape, labels.dtype, labels.min(), labels.max())

    if 1:
        print("\n*** CIFAR10 Dataset ***")
        root_dir = "/home/namu/myspace/NAMU/datasets/cifar10"
        train_loader = get_train_loader(dataset=CIFAR10(root_dir, "train"), batch_size=64)
        test_loader = get_test_loader(dataset=CIFAR10(root_dir, "test"), batch_size=32)

        batch = next(iter(train_loader))
        images, labels = batch["image"], batch["label"]
        print(images.shape, images.dtype, images.min(), images.max())
        print(labels.shape, labels.dtype, labels.min(), labels.max())

        batch = next(iter(test_loader))
        images, labels = batch["image"], batch["label"]
        print(images.shape, images.dtype, images.min(), images.max())
        print(labels.shape, labels.dtype, labels.min(), labels.max())

    if 1:
        print("\n*** Oxford Pets Dataset ***")
        root_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
        train_loader = get_train_loader(dataset=OxfordPets(root_dir, "train"), batch_size=32)
        test_loader = get_test_loader(dataset=OxfordPets(root_dir, "test"), batch_size=16)

        batch = next(iter(train_loader))
        images, labels, binary_labels = batch["image"], batch["label"], batch["binary_label"]
        print(images.shape, images.dtype, images.min(), images.max())
        print(labels.shape, labels.dtype, labels.min(), labels.max())
        print(binary_labels.shape, binary_labels.dtype, binary_labels.min(), binary_labels.max())

        batch = next(iter(test_loader))
        images, labels, binary_labels = batch["image"], batch["label"], batch["binary_label"]
        print(images.shape, images.dtype, images.min(), images.max())
        print(labels.shape, labels.dtype, labels.min(), labels.max())
        print(binary_labels.shape, binary_labels.dtype, binary_labels.min(), binary_labels.max())

