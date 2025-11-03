```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # INFO/DEBUG 로그 숨기기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # oneDNN 관련 이슈 방지 (옵션)

import numpy as np
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

def load_cifar10(root_dir):
    data_dir = os.path.join(root_dir, "cifar-10-batches-py")
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"CIFAR-10 Not Found!: {data_dir}")
    
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            x_train.append(batch[b'data'])
            y_train.extend(batch[b'labels'])
    
    x_train = np.vstack(x_train)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)

    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x_test = batch[b'data']
        y_test = batch[b'labels']
    
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

root_dir = '/home/namu/myspace/NAMU/datasets/cifar10'
(x_train, y_train), (x_test, y_test) = load_cifar10(root_dir)

NUM_CLASSES = 10
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)
```

### MLP

```python
input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation = 'relu')(x)
x = layers.Dense(units=150, activation = 'relu')(x)
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(input_layer, output_layer)

optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

model.evaluate(x_test, y_test)

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)
fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, 
        ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, 
        ha='center', transform=ax.transAxes)
    ax.imshow(img)
plt.show()
```

### CNN

```python
input_layer = layers.Input((32, 32, 3))
x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(rate = 0.5)(x)
output_layer = layers.Dense(10, activation='softmax')(x)

model = models.Model(input_layer, output_layer)

optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test, batch_size=1000)

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)
fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, 
        ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, 
        ha='center', transform=ax.transAxes)
    ax.imshow(img)
plt.show()
```
