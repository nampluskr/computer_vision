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
            end = start + self.batch_size
            yield self.images[indices[start:end]], self.labels[indices[start:end]]


#################################################################
## Math Functions
#################################################################

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def relu(x):
    return np.where(x >= 0, x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    batch_size = preds.shape[0]
    return -np.sum(targets*np.log(preds + 1.0E-8)) / batch_size


def accuracy(preds, targets):
    return (preds.argmax(axis=1) == targets.argmax(axis=1)).mean()


#################################################################
## Modules
#################################################################

class Module:
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)


class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class ReLU(Module):
    def forward(self, x):
        self.mask = x <= 0
        self.out = relu(x)
        return self.out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = [
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
        ]
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class CrossEntropyWithLogits:
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        self.preds = softmax(logits)
        self.targets = targets
        return cross_entropy(self.preds, self.targets)

    def backward(self):
        return (self.preds - self.targets) / self.targets.shape[0]


#################################################################
## Optimizer
#################################################################

class Optimizer:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr


class SGD(Optimizer):
    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, model, lr, beta1=0.0, beta2=0.999):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.ms = [np.zeros_like(param) for param in self.params]
        self.vs = [np.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m *= self.beta1
            m += (1 - self.beta1) * grad
            m_hat = m / (1.0 - self.beta1 ** self.iter)

            v *= self.beta2
            v += (1 - self.beta2) * (grad ** 2)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


if __name__ == "__main__":

    #################################################################
    ## Data Loading / Preprocessing
    #################################################################

    data_dir = r"E:\datasets\mnist"
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
    model = MLP(input_size, hidden_size, output_size)
    loss_fn = CrossEntropyWithLogits()
    optimizer = Adam(model, lr=0.001)

    #################################################################
    ## Training: Propagate Forward / Bacward - Update weights / baises
    #################################################################

    num_epochs = 10

    print("\n>> Training start ...")
    for epoch in range(1, num_epochs + 1):
        batch_loss = 0
        batch_acc = 0
        total_size = 0

        for x, y in train_loader:
            x_size = x.shape[0]
            total_size += x_size

            # Forward propagation
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy(softmax(logits), y)

            # Backward propagation
            dout = loss_fn.backward()
            model.backward(dout)

            # Update weights and biases
            optimizer.step()

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

        logits = model(x)
        loss = loss_fn(logits, y)
        acc = accuracy(softmax(logits), y)

        batch_loss += loss * x_size
        batch_acc += acc * x_size

    print(f"\n>> Evaluation: loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")
