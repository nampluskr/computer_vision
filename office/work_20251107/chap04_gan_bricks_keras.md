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
from tensorflow.keras import optimizers, losses, metrics
import tensorflow.keras.backend as K

from scipy.stats import norm

IMAGE_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 300
LOAD_MODEL = False
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002
NOISE_PARAM = 0.1

### utils

def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()

def preprocess(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img

train_data = utils.image_dataset_from_directory(
    "/home/namu/myspace/NAMU/datasets/lego_bricks/dataset",
    labels=None,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

train = train_data.map(lambda x: preprocess(x))

train_sample = sample_batch(train)
display(train_sample)
```

```python
discriminator_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(
    discriminator_input
)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    512, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    1,
    kernel_size=4,
    strides=1,
    padding="valid",
    use_bias=False,
    activation="sigmoid",
)(x)
discriminator_output = layers.Flatten()(x)

discriminator = models.Model(discriminator_input, discriminator_output)
discriminator.summary()


generator_input = layers.Input(shape=(Z_DIM,))
x = layers.Reshape((1, 1, Z_DIM))(generator_input)
x = layers.Conv2DTranspose(
    512, kernel_size=4, strides=1, padding="valid", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    64, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(
    CHANNELS,
    kernel_size=4,
    strides=2,
    padding="same",
    use_bias=False,
    activation="tanh",
)(x)
generator = models.Model(generator_input, generator_output)
generator.summary()

class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.d_real_acc_metric = metrics.BinaryAccuracy(name="d_real_acc")
        self.d_fake_acc_metric = metrics.BinaryAccuracy(name="d_fake_acc")
        self.d_acc_metric = metrics.BinaryAccuracy(name="d_acc")
        self.g_loss_metric = metrics.Mean(name="g_loss")
        self.g_acc_metric = metrics.BinaryAccuracy(name="g_acc")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_real_acc_metric,
            self.d_fake_acc_metric,
            self.d_acc_metric,
            self.g_loss_metric,
            self.g_acc_metric,
        ]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        # Train the discriminator on fake images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(
                random_latent_vectors, training=True
            )
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(
                generated_images, training=True
            )

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + NOISE_PARAM * tf.random.uniform(
                tf.shape(real_predictions)
            )
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - NOISE_PARAM * tf.random.uniform(
                tf.shape(fake_predictions)
            )

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(real_labels, fake_predictions)

        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.d_real_acc_metric.update_state(real_labels, real_predictions)
        self.d_fake_acc_metric.update_state(fake_labels, fake_predictions)
        self.d_acc_metric.update_state(
            [real_labels, fake_labels], [real_predictions, fake_predictions]
        )
        self.g_loss_metric.update_state(g_loss)
        self.g_acc_metric.update_state(real_labels, fake_predictions)

        return {m.name: m.result() for m in self.metrics}

# Create a DCGAN
dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=Z_DIM)
dcgan.compile(
    d_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
    g_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
)
dcgan.fit(train, epochs=EPOCHS)
```

```
Epoch 1/300
2025-11-07 10:23:33.012624: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_1/dropout_4/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
313/313 [==============================] - 309s 956ms/step - d_loss: 0.2032 - d_real_acc: 0.8576 - d_fake_acc: 0.8723 - d_acc: 0.8649 - g_loss: 4.8330 - g_acc: 0.1277
Epoch 2/300
313/313 [==============================] - 17s 53ms/step - d_loss: 0.0306 - d_real_acc: 0.8978 - d_fake_acc: 0.9055 - d_acc: 0.9016 - g_loss: 5.1667 - g_acc: 0.0945
Epoch 3/300
313/313 [==============================] - 17s 53ms/step - d_loss: 0.0565 - d_real_acc: 0.8895 - d_fake_acc: 0.8970 - d_acc: 0.8932 - g_loss: 5.0534 - g_acc: 0.1030
Epoch 4/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.1555 - d_real_acc: 0.8741 - d_fake_acc: 0.8776 - d_acc: 0.8758 - g_loss: 4.3441 - g_acc: 0.1224
Epoch 5/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.1368 - d_real_acc: 0.8768 - d_fake_acc: 0.8798 - d_acc: 0.8783 - g_loss: 4.1207 - g_acc: 0.1203
Epoch 6/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.1096 - d_real_acc: 0.8880 - d_fake_acc: 0.8899 - d_acc: 0.8890 - g_loss: 4.1853 - g_acc: 0.1101
Epoch 7/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.1632 - d_real_acc: 0.8787 - d_fake_acc: 0.8779 - d_acc: 0.8783 - g_loss: 4.1236 - g_acc: 0.1221
Epoch 8/300
313/313 [==============================] - 18s 55ms/step - d_loss: 0.0884 - d_real_acc: 0.8943 - d_fake_acc: 0.8933 - d_acc: 0.8938 - g_loss: 4.2847 - g_acc: 0.1067
Epoch 9/300
313/313 [==============================] - 17s 53ms/step - d_loss: 0.0577 - d_real_acc: 0.9061 - d_fake_acc: 0.9075 - d_acc: 0.9068 - g_loss: 4.7533 - g_acc: 0.0925
Epoch 10/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.0638 - d_real_acc: 0.8991 - d_fake_acc: 0.9016 - d_acc: 0.9003 - g_loss: 4.7371 - g_acc: 0.0984
Epoch 11/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.0382 - d_real_acc: 0.9084 - d_fake_acc: 0.9099 - d_acc: 0.9092 - g_loss: 4.9644 - g_acc: 0.0901
Epoch 12/300
313/313 [==============================] - 18s 55ms/step - d_loss: 0.0293 - d_real_acc: 0.9099 - d_fake_acc: 0.9083 - d_acc: 0.9091 - g_loss: 5.2496 - g_acc: 0.0917
Epoch 13/300
313/313 [==============================] - 17s 54ms/step - d_loss: -0.0053 - d_real_acc: 0.9227 - d_fake_acc: 0.9233 - d_acc: 0.9230 - g_loss: 5.3289 - g_acc: 0.0767
Epoch 14/300
313/313 [==============================] - 17s 54ms/step - d_loss: 0.0212 - d_real_acc: 0.9117 - d_fake_acc: 0.9101 - d_acc: 0.9109 - g_loss: 5.2839 - g_acc: 0.0899
Epoch 15/300
313/313 [==============================] - 17s 54ms/step - d_loss: -0.0038 - d_real_acc: 0.9152 - d_fake_acc: 0.9180 - d_acc: 0.9166 - g_loss: 5.6491 - g_acc: 0.0820
Epoch 16/300
313/313 [==============================] - 17s 53ms/step - d_loss: -0.0492 - d_real_acc: 0.9289 - d_fake_acc: 0.9293 - d_acc: 0.9291 - g_loss: 5.7090 - g_acc: 0.0707
Epoch 17/300
313/313 [==============================] - 17s 55ms/step - d_loss: 0.0622 - d_real_acc: 0.9065 - d_fake_acc: 0.9112 - d_acc: 0.9089 - g_loss: 5.9576 - g_acc: 0.0888
Epoch 18/300
135/313 [===========>..................] - ETA: 12s - d_loss: -0.0501 - d_real_acc: 0.9256 - d_fake_acc: 0.9286 - d_acc: 0.9271 - g_loss: 5.9783 - g_acc: 0.0714
```
