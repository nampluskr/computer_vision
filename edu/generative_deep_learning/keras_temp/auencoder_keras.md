```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # INFO/DEBUG 로그 숨기기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # oneDNN 관련 이슈 방지 (옵션)

import numpy as np
import matplotlib.pyplot as plt
import gzip

import tensorflow as tf
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K

import os
import gzip
import numpy as np

def load_fashion_mnist(root_dir):
    data_dir = root_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Fashion MNIST Not Found!: {data_dir}")

    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    with gzip.open(train_images_path, 'rb') as f:
        x_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        x_train = x_train.reshape(-1, 28, 28)

    with gzip.open(train_labels_path, 'rb') as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(test_images_path, 'rb') as f:
        x_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        x_test = x_test.reshape(-1, 28, 28)

    with gzip.open(test_labels_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return (x_train, y_train), (x_test, y_test)

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

root_dir = '/home/namu/myspace/NAMU/datasets/fashion_mnist'
(x_train,y_train), (x_test,y_test) = load_fashion_mnist(root_dir)

x_train = preprocess(x_train)
x_test = preprocess(x_test)

print(f"train images: {type(x_train)} {x_train.shape}, dtype={x_train.dtype}, min={x_train.min()}, max={x_train.max()}")
print(f"train labels: {type(y_train)} {y_train.shape}, dtype={y_train.dtype}, min={y_train.min()}, max={y_train.max()}")

# Encoder
encoder_input = layers.Input(shape=(32, 32, 1), name = "encoder_input")
x = layers.Conv2D(32, (3, 3), strides = 2, activation = 'relu', padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
encoder_output = layers.Dense(2, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)

# Decoder
decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation = 'relu', padding="same")(x)
decoder_output = layers.Conv2D(1, (3, 3), strides = 1, activation="sigmoid",
    padding="same", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)

# Autoencoder
autoencoder = models.Model(encoder_input, decoder(encoder_output))

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=100, shuffle=True, 
    validation_data=(x_test, x_test))
```

```python
n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]
embeddings = encoder.predict(example_images)

figsize = 8
plt.figure(figsize=(figsize, figsize))
plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    cmap="rainbow",
    c=example_labels,
    alpha=0.8,
    s=3,
)
plt.colorbar()
plt.show()

# Get the range of the existing embeddings
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)

# Sample some points in the latent space
grid_width, grid_height = (6, 3)
sample = np.random.uniform(mins, maxs, size=(grid_width * grid_height, 2))

reconstructions = decoder.predict(sample)
figsize = 8
plt.figure(figsize=(figsize, figsize))

# ... the original embeddings ...
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)

# ... and the newly generated points in the latent space
plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show()

# Add underneath a grid of the decoded images
fig = plt.figure(figsize=(figsize, grid_height * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        str(np.round(sample[i, :], 1)),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(reconstructions[i, :, :], cmap="Greys")

# Colour the embeddings by their label (clothing type - see table)
figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(embeddings[:, 0], embeddings[:, 1],
    cmap="rainbow", c=example_labels, alpha=0.8, s=20)
plt.colorbar()

x = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), grid_size)
y = np.linspace(max(embeddings[:, 1]), min(embeddings[:, 1]), grid_size)
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

reconstructions = decoder.predict(grid)
plt.scatter(grid[:, 0], grid[:, 1], c="black", alpha=1, s=10)
plt.show()

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :], cmap="Greys")
plt.show()
```
