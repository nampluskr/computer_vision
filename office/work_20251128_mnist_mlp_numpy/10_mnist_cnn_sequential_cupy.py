from backend import backend, xp
print(f"Using: {'CuPy (GPU)' if backend.use_gpu else 'NumPy (CPU)'}")

import os
import numpy as np
import gzip

#################################################################
## Data Loading
#################################################################

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), xp.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    data = xp.asarray(data)
    data = xp.pad(data, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    return data

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), xp.uint8, offset=8)
    data = xp.asarray(data)
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
            self.num_batches = int(xp.ceil(self.num_images / batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = xp.arange(self.num_images)
        if self.shuffle:
            indices = xp.random.permutation(indices)
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
    return xp.eye(num_classes)[x]


def sigmoid(x):
    return xp.where(x >= 0, 1 / (1 + xp.exp(-x)), xp.exp(x) / (1 + xp.exp(x)))


def relu(x):
    return xp.maximum(0, x)


def softmax(x):
    if x.ndim == 1:
        e_x = xp.exp(x - xp.max(x))
        return e_x / xp.sum(e_x)
    e_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    return e_x / xp.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    batch_size = preds.shape[0]
    if targets.ndim == 1:
        probs = preds[xp.arange(batch_size), targets]
    else:   # one-hot labels
        probs = xp.sum(preds * targets, axis=1)
    return -xp.mean(xp.log(probs + 1e-8))


def log_softmax(x):
    if x.ndim == 1:
        return x - xp.max(x) - xp.log(xp.sum(xp.exp(x - xp.max(x))))
    max_x = xp.max(x, axis=1, keepdims=True)
    return x - max_x - xp.log(xp.sum(xp.exp(x - max_x), axis=1, keepdims=True))


def cross_entropy_with_logits(logits, targets):
    log_probs = log_softmax(logits)
    if targets.ndim == 1:
        batch_size = logits.shape[0]
        return -xp.mean(log_probs[xp.arange(batch_size), targets])
    return -xp.mean(xp.sum(targets * log_probs, axis=1))


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
    max_val = xp.maximum(-logits, 0)
    loss = max_val + xp.log(xp.exp(-max_val) + xp.exp(-logits - max_val))
    return xp.mean(targets * loss + (1 - targets) * (logits + loss))


def im2col(images, kernel_size, stride, padding):
    if padding > 0:
        images = xp.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    B, C, H, W = images.shape
    K = kernel_size
    out_h = (H - K) // stride + 1
    out_w = (W - K) // stride + 1
    cols = xp.zeros((B, C, K, K, out_h, out_w))

    for y in range(K):
        y_max = y + stride * out_h
        for x in range(K):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = images[:, :, y:y_max:stride, x:x_max:stride]

    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, -1), out_h, out_w


def col2im(cols, images_shape, kernel_size, stride, padding):
    B, C, H, W = images_shape
    if padding > 0:
        H_pad, W_pad = H + 2 * padding, W + 2 * padding
        images = xp.zeros((B, C, H_pad, W_pad))
    else:
        images = xp.zeros((B, C, H, W))
        H_pad, W_pad = H, W

    K = kernel_size
    out_h = (H_pad - K) // stride + 1
    out_w = (W_pad - K) // stride + 1
    cols_reshaped = cols.reshape(B, out_h, out_w, C, K, K).transpose(0, 3, 4, 5, 1, 2)

    for y in range(K):
        for x in range(K):
            images[:, :, y:y + stride * out_h:stride, x:x + stride * out_w:stride] += cols_reshaped[:, :, y, x, :, :]

    if padding > 0:
        images = images[:, :, padding:-padding, padding:-padding]
    return images


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
            scale = xp.sqrt(2.0 / (in_features + out_features))
        elif init == 'he':      # relu, leakyrelu
            scale = xp.sqrt(2.0 / in_features)
        else:
            scale = 1.0

        self.w = xp.random.randn(in_features, out_features) * scale
        self.b = xp.zeros(out_features)
        self.grad_w = xp.zeros_like(self.w)
        self.grad_b = xp.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])
        self.x = None

    def forward(self, x):
        self.x = x
        return xp.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = xp.dot(self.x.T, dout)
        self.grad_b[...] = xp.sum(dout, axis=0)
        return xp.dot(dout, self.w.T)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init='he'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if init == 'xavier':
            scale = xp.sqrt(1.0 / (in_channels * kernel_size * kernel_size))
        elif init == 'he':
            scale = xp.sqrt(2.0 / in_channels)
        else:
            scale = 1.0

        self.w = xp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = xp.zeros(out_channels)

        self.grad_w = xp.zeros_like(self.w)
        self.grad_b = xp.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]

        self.x = None
        self.col_cache = None  # (col_x, out_h, out_w)
        self.col_w = None

    def forward(self, x):
        B, C, H, W = x.shape
        self.x = x

        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding)
        self.col_cache = (col_x, out_h, out_w)
        self.col_w = self.w.reshape(self.out_channels, -1)  # (out_c, in_c * K * K)

        out = xp.dot(col_x, self.col_w.T) + self.b          # (B*out_h*out_w, out_c)
        out = out.reshape(B, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        B, out_c, out_h, out_w = dout.shape
        col_x, out_h_cache, out_w_cache = self.col_cache

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grad_b[...] = xp.sum(dout_flat, axis=0)

        grad_w_flat = xp.dot(dout_flat.T, col_x)                    # (out_c, in_c*K*K)
        self.grad_w[...] = grad_w_flat.reshape(self.grad_w.shape)   # (out_c, in_c, K, K)

        dcol_x = xp.dot(dout_flat, self.col_w)                      # (B*out_h*out_w, in_c*K*K)
        dx = col2im(dcol_x, self.x.shape, self.kernel_size, self.stride, self.padding)
        return dx


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
        return xp.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        return dout * xp.where(self.x > 0, 1, self.alpha)


class Tanh(Module):
    def forward(self, x):
        self.out = xp.tanh(x)
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
            self.mask = (xp.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        return dout * self.mask


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.w = xp.ones(num_features)          # gamma
        self.b = xp.zeros(num_features)         # beta
        self.grad_w = xp.zeros_like(self.w)
        self.grad_b = xp.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads  = [self.grad_w, self.grad_b]

        self.running_mean = xp.zeros(num_features, dtype=xp.float32)
        self.running_var  = xp.ones(num_features, dtype=xp.float32)
        self.training = True

        self.x = None
        self.x_norm = None
        self.mean = None
        self.var = None
        self.inv_std = None

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        assert C == self.num_features, \
            f"Channel mismatch: got {C}, expected {self.num_features}"

        if self.training:
            mean = x.mean(axis=(0, 2, 3))          # (C,)
            var  = x.var(axis=(0, 2, 3))           # (C,)

            self.mean = mean
            self.var  = var
            self.inv_std = 1.0 / xp.sqrt(var + self.eps)

            x_centered = x - mean.reshape(1, C, 1, 1)
            x_norm = x_centered * self.inv_std.reshape(1, C, 1, 1)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean_reshaped = self.running_mean.reshape(1, C, 1, 1)
            var_reshaped  = self.running_var.reshape(1, C, 1, 1)
            x_norm = (x - mean_reshaped) / xp.sqrt(var_reshaped + self.eps)

        self.x_norm = x_norm
        out = self.w.reshape(1, C, 1, 1) * x_norm + self.b.reshape(1, C, 1, 1)
        return out

    def backward(self, dout):
        B, C, H, W = dout.shape
        dbeta = dout.sum(axis=(0, 2, 3))                    # (C,)
        dgamma = (dout * self.x_norm).sum(axis=(0, 2, 3))   # (C,)

        self.grad_b[...] = dbeta
        self.grad_w[...] = dgamma

        dx_norm = dout * self.w.reshape(1, C, 1, 1)
        dvar = -0.5 * xp.sum(dx_norm * (self.x - self.mean.reshape(1, C, 1, 1)),
                             axis=(0, 2, 3)) * self.inv_std**3
        dmean = -xp.sum(dx_norm, axis=(0, 2, 3)) * self.inv_std - \
                dvar * 2 * xp.mean(self.x - self.mean.reshape(1, C, 1, 1), axis=(0, 2, 3))

        dx = dx_norm * self.inv_std.reshape(1, C, 1, 1)
        dx += dvar.reshape(1, C, 1, 1) * (self.x - self.mean.reshape(1, C, 1, 1)) / (B * H * W)
        dx += dmean.reshape(1, C, 1, 1) / (B * H * W)

        return dx

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        self.grad_w.fill(0)
        self.grad_b.fill(0)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

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

    def train(self):
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)


# class MLP(Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
#         super().__init__()
#         self.layers = [
#             Linear(input_size, hidden_size, init="he"),
#             LeakyReLU(0.2),
#             Dropout(dropout),
#             Linear(hidden_size, hidden_size, init="he"),
#             LeakyReLU(0.2),
#             Dropout(dropout),
#             Linear(hidden_size, output_size, init="xavier"),
#         ]
#         for layer in self.layers:
#             self.params.extend(layer.params)
#             self.grads.extend(layer.grads)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def backward(self, dout):
#         for layer in reversed(self.layers):
#             dout = layer.backward(dout)
#         return dout


class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.model = Sequential(
            Linear(input_size, hidden_size, init="he"),
            LeakyReLU(0.2),
            Dropout(dropout),
            Linear(hidden_size, hidden_size, init="he"),
            LeakyReLU(0.2),
            Dropout(dropout),
            Linear(hidden_size, output_size, init="xavier")
        )
        self.params = self.model.params
        self.grads = self.model.grads

    def forward(self, x):
        return self.model(x)

    def backward(self, dout):
        return self.model.backward(dout)

    def train(self):
        self.training = True
        self.model.train()

    def eval(self):
        self.training = False
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad()


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        self.x = None
        self.col = None
        self.max_idx = None
        self.out_shape = None

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        # 1. Padding
        if P > 0:
            x = xp.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')

        H_p, W_p = x.shape[2], x.shape[3]

        # Output size
        out_h = (H_p - K) // S + 1
        out_w = (W_p - K) // S + 1
        self.out_shape = (B, C, out_h, out_w)

        # 2. im2col 방식으로 window 추출 (벡터화)
        col = xp.zeros((B, C, K, K, out_h, out_w))
        for i in range(K):
            i_end = i + S * out_h
            for j in range(K):
                j_end = j + S * out_w
                col[:, :, i, j, :, :] = x[:, :, i:i_end:S, j:j_end:S]

        # 3. Flatten kernel dims
        col_flat = col.reshape(B, C, K*K, out_h, out_w)

        # 4. Max pooling
        max_idx = xp.argmax(col_flat, axis=2)      # (B, C, oh, ow)
        out = xp.max(col_flat, axis=2)

        # Save for backward
        self.col = col_flat
        self.max_idx = max_idx

        return out

    def backward(self, dout):
        B, C, out_h, out_w = self.out_shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        # 1. dx 초기화 (padding 포함)
        H_p = self.x.shape[2] + 2 * P if P > 0 else self.x.shape[2]
        W_p = self.x.shape[3] + 2 * P if P > 0 else self.x.shape[3]
        dx = xp.zeros((B, C, H_p, W_p), dtype=dout.dtype)

        # 2. max_idx (B, C, oh, ow) -> (kh, kw) 분해
        max_idx = self.max_idx  # shape (B, C, oh, ow)
        kh = max_idx // K
        kw = max_idx % K

        # 3. output grid 만들기
        oh_idx = xp.arange(out_h).reshape(1, 1, out_h, 1)
        ow_idx = xp.arange(out_w).reshape(1, 1, 1, out_w)

        # 4. 입력 위치 계산
        h_pos = kh + oh_idx * S   # shape (B,C,oh,ow)
        w_pos = kw + ow_idx * S

        # 5. batch/channels grid
        b_idx = xp.arange(B).reshape(B, 1, 1, 1)
        c_idx = xp.arange(C).reshape(1, C, 1, 1)

        # 6. scatter add (벡터화)
        dx[b_idx, c_idx, h_pos, w_pos] += dout

        # 7. padding 제거
        if P > 0:
            dx = dx[:, :, P:-P, P:-P]

        return dx

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass



class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass


# class CNN(Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.layers = [
#             Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(32),
#             ReLU(),
#             MaxPool2d(kernel_size=2, stride=2),

#             Conv2d(32, 64, kernel_size=3, padding=1),
#             BatchNorm2d(64),
#             ReLU(),
#             MaxPool2d(kernel_size=2, stride=2),

#             Flatten(),
#             Linear(64 * 8 * 8, 512),
#             ReLU(),
#             Linear(512, num_classes)
#         ]
#         for layer in self.layers:
#             self.params.extend(layer.params)
#             self.grads.extend(layer.grads)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def backward(self, dout):
#         for layer in reversed(self.layers):
#             dout = layer.backward(dout)
#         return dout

class CNN(Module):
    def __init__(self, in_channels=3, num_classes=10, hidden_channels=(32, 64)):
        super().__init__()
        c1, c2 = hidden_channels

        self.features = Sequential(
            Conv2d(in_channels, c1, kernel_size=3, padding=1),
            BatchNorm2d(c1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # H/2

            Conv2d(c1, c2, kernel_size=3, padding=1),
            BatchNorm2d(c2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # H/2
        )

        self.classifier = Sequential(
            Flatten(),
            Linear(c2 * 8 * 8, 512),
            ReLU(),
            Linear(512, num_classes)
        )
        self.params = self.features.params + self.classifier.params
        self.grads = self.features.grads + self.classifier.grads

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def backward(self, dout):
        dout = self.classifier.backward(dout)
        dout = self.features.backward(dout)
        return dout

    def train(self):
        self.training = True
        self.features.train()
        self.classifier.train()

    def eval(self):
        self.training = False
        self.features.eval()
        self.classifier.eval()

    def zero_grad(self):
        self.features.zero_grad()
        self.classifier.zero_grad()


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
            grad[xp.arange(batch_size), self.targets] -= 1
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


class Momentum(Optimizer):
    def __init__(self, model, lr, momentum=0.9):
        super().__init__(model, lr)
        self.momentum = momentum
        self.velocities = [xp.zeros_like(param) for param in self.params]

    def step(self):
        for param, grad, velocity in zip(self.params, self.grads, self.velocities):
            velocity[...] = self.momentum * velocity - self.lr * grad
            param[...] += velocity


class RMSprop(Optimizer):
    def __init__(self, model, lr, alpha=0.99, eps=1e-8):
        super().__init__(model, lr)
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [xp.zeros_like(param) for param in self.params]

    def step(self):
        for param, grad, square_avg in zip(self.params, self.grads, self.square_avg):
            square_avg[...] = self.alpha * square_avg + (1 - self.alpha) * (grad ** 2)
            param[...] -= self.lr * grad / (xp.sqrt(square_avg) + self.eps)


class Adam(Optimizer):
    def __init__(self, model, lr, beta1=0.9, beta2=0.999):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.ms = [xp.zeros_like(param) for param in self.params]
        self.vs = [xp.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.iter)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param[...] -= self.lr * m_hat / (xp.sqrt(v_hat) + 1e-8)


class AdamW(Optimizer):
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, weight_decay=0.01):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0
        self.ms = [xp.zeros_like(param) for param in self.params]
        self.vs = [xp.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            param[...] -= self.lr * self.weight_decay * param

            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.iter)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param[...] -= self.lr * m_hat / (xp.sqrt(v_hat) + 1e-8)


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

    if 1:
        xp.random.seed(42)

        #################################################################
        ## Data Loading / Preprocessing
        #################################################################

        def preprocess(images, labels):
            images = images.astype(xp.float32).reshape(-1, 32*32) / 255.0
            return images, labels

        data_dir = "/mnt/d/datasets/mnist"
        train_dataset = MNIST(data_dir, split="train", transform=preprocess)
        test_dataset = MNIST(data_dir, split="test", transform=preprocess)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        #################################################################
        ## Modeling: 3-layer MLP (input layer - hidden layer - output layer)
        #################################################################

        input_size, hidden_size, output_size = 32*32, 100, 10
        model = MLP(input_size=32*32, hidden_size=100, output_size=10, dropout=0.3)
        optimizer = Momentum(model, lr=0.01, momentum=0.9)
        # optimizer = AdamW(model, lr=0.001, weight_decay=0.01)
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

    if 1:
        xp.random.seed(42)

        #################################################################
        ## Data Loading / Preprocessing
        #################################################################

        def preprocess(images, labels):
            images = images.astype(xp.float32) / 255.0
            images = xp.expand_dims(images, axis=1)
            # images = xp.tile(images, (1, 3, 1, 1))
            return images, labels

        data_dir = "/mnt/d/datasets/mnist"
        train_dataset = MNIST(data_dir, split="train", transform=preprocess)
        test_dataset = MNIST(data_dir, split="test", transform=preprocess)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        #################################################################
        ## Modeling: CNN
        #################################################################

        model = CNN(in_channels=1, num_classes=10, hidden_channels=(16, 32))
        # optimizer = Momentum(model, lr=0.01, momentum=0.9)
        optimizer = Adam(model, lr=0.001)
        # optimizer = AdamW(model, lr=0.001, weight_decay=0.01)
        # optimizer = RMSprop(model, lr=0.001, alpha=0.99, eps=1e-8)
        
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
