import os
import numpy as np
import gzip

#################################################################
## Data Loading Functions
#################################################################

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def get_mnist(data_dir, split="train"):
    if split == "train":
        images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    elif split == "test":
        images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
        labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    else:
        raise ValueError(">> split must be train or test!")
    return images, labels


class DataLoader:
    def __init__(self, images, labels, batch_size, shuffle=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_images = len(images)
        self.num_batches = int(np.ceil(len(images) / batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = np.arange(self.num_images)
        if self.shuffle:
            indices = np.random.permutation(indices)

        for i in range(self.num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            images = self.images[indices[start:end]]
            labels = self.labels[indices[start:end]]
            yield images, labels


#################################################################
## Math Functions
#################################################################

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    if targets.ndim == 1:
        batch_size = preds.shape[0]
        probs = preds[np.arange(batch_size), targets]
    else:   # one-hot labels
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def accuracy(preds, targets):
    preds = preds.argmax(axis=1)
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)
    return (preds == targets).mean()


if __name__ == "__main__":

    #################################################################
    ## Data Loading / Preprocessing
    #################################################################
    
    data_dir = "/mnt/d/datasets/mnist"
    x_train, y_train = get_mnist(data_dir, split="train")
    x_test, y_test = get_mnist(data_dir, split="test")

    print("\n>> Data before preprocessing:")
    print(f"train images: {x_train.dtype}, {x_train.shape}, [{x_train.min()}, {x_train.max()}]")
    print(f"train labels: {y_train.dtype}, {y_train.shape}, [{y_train.min()}, {y_train.max()}]")

    ## Preprocessing
    x_train = x_train.astype(np.float32).reshape(-1, 28*28) / 255
    x_test = x_test.astype(np.float32).reshape(-1, 28*28) / 255

    y_train = one_hot(y_train, num_classes=10).astype(np.int64)
    y_test = one_hot(y_test, num_classes=10).astype(np.int64)

    print("\n>> Data after preprocessing:")
    print(f"train images: {x_train.dtype}, {x_train.shape}, [{x_train.min()}, {x_train.max()}]")
    print(f"train labels: {y_train.dtype}, {y_train.shape}, [{y_train.min()}, {y_train.max()}]")
    
    train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)

    #################################################################
    ## Modeling: 3-layer MLP (input layer - hidden layer - output layer)
    #################################################################
    
    np.random.seed(42)
    input_size, hidden_size, output_size = 28*28, 100, 10

    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, hidden_size)
    b2 = np.zeros(hidden_size)
    w3 = np.random.randn(hidden_size, output_size)
    b3 = np.zeros(output_size)

    #################################################################
    ## Training: Propagate Forward / Bacward - Update weights / baises
    #################################################################
    
    num_epochs = 10
    learning_rate = 0.01

    print("\n>> Training start ...")
    for epoch in range(1, num_epochs + 1):
        batch_loss = 0
        batch_acc = 0
        total_size = 0

        for x, y in train_loader:
            x_size = x.shape[0]
            total_size += x_size

            # Forward propagation
            z1 = np.dot(x, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(a2, w3) + b3
            out = softmax(z3)

            loss = cross_entropy(out, y)
            acc = accuracy(out, y)

            # Backward propagation
            grad_z3 = (out - y) / y.shape[0]
            grad_w3 = np.dot(a2.T, grad_z3)
            grad_b3 = np.sum(grad_z3, axis=0)

            grad_a2 = np.dot(grad_z3, w3.T)
            grad_z2 = a2 * (1 - a2) * grad_a2
            grad_w2 = np.dot(a1.T, grad_z2)
            grad_b2 = np.sum(grad_z2, axis=0)

            grad_a1 = np.dot(grad_z2, w2.T)
            grad_z1 = a1 * (1 - a1) * grad_a1
            grad_w1 = np.dot(x.T, grad_z1)
            grad_b1 = np.sum(grad_z1, axis=0)

            # Update weights and biases
            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1
            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2
            w3 -= learning_rate * grad_w3
            b3 -= learning_rate * grad_b3

            batch_loss += loss * x_size
            batch_acc += acc * x_size

        print(f"[{epoch:3d}/{num_epochs}] "
              f"loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")

    #################################################################
    ## Evaluation using test data
    #################################################################

    batch_loss = 0
    batch_acc = 0
    total_size = 0

    for x, y in test_loader:
        x_size = x.shape[0]
        total_size += x_size

        # Forward propagation
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        out = softmax(z3)

        loss = cross_entropy(out, y)
        acc = accuracy(out, y)

        batch_loss += loss * x_size
        batch_acc += acc * x_size

    print(f"\n>> Evaluation: loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")
