import os
import numpy as np
import gzip

#################################################################
## Data Loading
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


class MNIST:
    def __init__(self, data_dir, split="train", transform=None):
        if split == "train":
            self.images = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
            self.labels = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
        elif split == "test":
            self.images = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
            self.labels = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
        else:
            raise ValueError(">> split must be train or test!")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image, label = self.transform(image, label)
        return image, label


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_images = len(dataset)
        if drop_last:
            self.num_batches = self.num_images // batch_size
        else:
            self.num_batches = int(np.ceil(self.num_images / batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = np.arange(self.num_images)
        if self.shuffle:
            indices = np.random.permutation(indices)
        if self.drop_last:
            indices = indices[:self.num_batches * self.batch_size]

        for i in range(self.num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            images, labels = self.dataset[indices[start:end]]
            yield images, labels


#################################################################
## Math Functions
#################################################################

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    batch_size = preds.shape[0]
    if targets.ndim == 1:
        probs = preds[np.arange(batch_size), targets]
    else:   # one-hot labels
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def log_softmax(x):
    if x.ndim == 1:
        return x - np.max(x) - np.log(np.sum(np.exp(x - np.max(x))))
    max_x = np.max(x, axis=1, keepdims=True)
    return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))


def cross_entropy_with_logits(logits, targets):
    log_probs = log_softmax(logits)
    if targets.ndim == 1:
        batch_size = logits.shape[0]
        return -np.mean(log_probs[np.arange(batch_size), targets])
    return -np.mean(np.sum(targets * log_probs, axis=1))


def accuracy(preds, targets):
    pred_classes = preds.argmax(axis=1)
    if targets.ndim == 2:  # one-hot
        target_classes = targets.argmax(axis=1)
    else:  # integer labels
        target_classes = targets
    return (pred_classes == target_classes).mean()


def binary_accuracy(preds, targets, threshold=0.5):
    pred_classes = (preds >= threshold).astype(int)
    if preds.ndim > 1:
        pred_classes = pred_classes.flatten()
    if targets.ndim > 1:
        targets = targets.flatten()
    return (pred_classes == targets).mean()


def bce_with_logits(logits, targets):
    # log(sigmoid(x)) = x - softplus(x) = -max(-x, 0) - log(1 + exp(-|x|))
    # log(1 - sigmoid(x)) = -softplus(x) = -max(x, 0) - log(1 + exp(-|x|))
    max_val = np.maximum(-logits, 0)
    loss = max_val + np.log(np.exp(-max_val) + np.exp(-logits - max_val))
    return np.mean(targets * loss + (1 - targets) * (logits + loss))


#################################################################
## Modules
#################################################################

class Module:
    def __init__(self):
        self.params = []
        self.grads = []
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        self.training = True
        for layer in getattr(self, 'layers', []):
            layer.train()

    def eval(self):
        self.training = False
        for layer in getattr(self, 'layers', []):
            layer.eval()

    def zero_grad(self):
        for grad in self.grads:
            grad.fill(0)
        for layer in getattr(self, 'layers', []):
            layer.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features, init='he'):
        super().__init__()
        if init == 'xavier':    # sigmoid, tanh
            scale = np.sqrt(2.0 / (in_features + out_features))
        elif init == 'he':      # relu, leakyrelu
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = 1.0

        self.w = np.random.randn(in_features, out_features) * scale
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


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        return dout * np.where(self.x > 0, 1, self.alpha)


class Tanh(Module):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out ** 2)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        return dout * self.mask


class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.layers = [
            Linear(input_size, hidden_size, init="he"),
            LeakyReLU(0.2),
            Dropout(dropout),
            Linear(hidden_size, hidden_size, init="he"),
            LeakyReLU(0.2),
            Dropout(dropout),
            Linear(hidden_size, output_size, init="xavier"),
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


#################################################################
## Loss / Metrics
#################################################################

class Metric:
    def __call__(self, *args):
        return self.forward(*args)


class CrossEntropyWithLogits(Metric):
    def forward(self, logits, targets):
        self.preds = softmax(logits)
        self.targets = targets
        return cross_entropy_with_logits(logits, targets)

    def backward(self):
        batch_size = self.preds.shape[0]
        if self.targets.ndim == 1:
            grad = self.preds.copy()
            grad[np.arange(batch_size), self.targets] -= 1
            return grad / batch_size
        else:  # one-hot labels
            return (self.preds - self.targets) / batch_size


class BCEWithLogits(Metric):
    def forward(self, logits, targets):
        self.preds = sigmoid(logits)
        self.targets = targets
        return bce_with_logits(logits, targets)

    def backward(self):
        batch_size = self.preds.shape[0]
        return (self.preds - self.targets) / batch_size


class Accuracy(Metric):
    def forward(self, preds, targets):
        return accuracy(preds, targets)


class BinaryAccuracy(Metric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def forward(self, preds, targets):
        return binary_accuracy(preds, targets, self.threshold)


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
            param[...] -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, model, lr, beta1=0.9, beta2=0.999):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.ms = [np.zeros_like(param) for param in self.params]
        self.vs = [np.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.iter)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param[...] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class AdamW(Optimizer):
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, weight_decay=0.01):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0
        self.ms = [np.zeros_like(param) for param in self.params]
        self.vs = [np.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            param[...] -= self.lr * self.weight_decay * param

            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.iter)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param[...] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


#################################################################
## Trainer
#################################################################

import sys
from tqdm import tqdm


class Classifier:
    def __init__(self, model, optimizer=None, loss_fn=None):
        self.model = model
        self.optimizer = optimizer or AdamW(model, lr=0.001, weight_decay=0.01)
        self.loss_fn = loss_fn or CrossEntropyWithLogits()
        self.acc_metric = Accuracy()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_step(self, images, labels):
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(softmax(logits), labels)

        self.model.zero_grad()
        dout = self.loss_fn.backward()
        self.model.backward(dout)
        self.optimizer.step()
        return dict(loss=loss, acc=acc)

    def eval_step(self, images, labels):
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(softmax(logits), labels)
        return dict(loss=loss, acc=acc)


def train(model, dataloader):
    model.train()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for images, labels in progress_bar:
            batch_size = images.shape[0]
            total += batch_size

            outputs = model.train_step(images, labels)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def evaluate(model, dataloader):
    model.eval()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for images, labels in progress_bar:
            batch_size = images.shape[0]
            total += batch_size

            outputs = model.eval_step(images, labels)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += float(value) * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}"
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def fit(model, train_loader, num_epochs, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        epoch_info = f"[{epoch:3d}/{num_epochs}]"
        train_results = train(model, train_loader)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history


if __name__ == "__main__":

    #################################################################
    ## Data Loading / Preprocessing
    #################################################################

    def preprocess(images, labels):
        images = images.astype(np.float32).reshape(-1, 28*28) / 255
        return images, labels

    data_dir = "/mnt/d/datasets/mnist"
    train_dataset = MNIST(data_dir, split="train", transform=preprocess)
    test_dataset = MNIST(data_dir, split="test", transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    #################################################################
    ## Modeling: 3-layer MLP (input layer - hidden layer - output layer)
    #################################################################

    np.random.seed(42)
    input_size, hidden_size, output_size = 28*28, 100, 10
    model = MLP(input_size, hidden_size, output_size, dropout=0.3)
    optimizer = AdamW(model, lr=0.001, weight_decay=0.01)
    clf = Classifier(model, optimizer=optimizer)

    #################################################################
    ## Training: Propagate Forward / Bacward - Update weights / baises
    #################################################################

    print("\n>> Training start ...")
    history = fit(clf, train_loader, num_epochs=10, valid_loader=test_loader)

    #################################################################
    ## Evaluation using test data
    #################################################################

    results = evaluate(clf, test_loader)
    print(f"\n>> Evaluation: loss:{results['loss']:.3f} acc:{results['acc']:.3f}")
