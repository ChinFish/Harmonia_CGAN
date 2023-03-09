import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy, Mean

from sklearn.metrics import accuracy_score


class ConditionalGAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.g_acc_tracker = BinaryAccuracy(name="g_acc")
        self.d_real_acc_tracker = BinaryAccuracy(name="d_real_acc")
        self.d_fake_acc_tracker = BinaryAccuracy(name="d_fake_acc")
        self.g_loss_tracker = Mean(name="g_loss")
        self.d_real_loss_tracker = Mean(name="d_real_loss")
        self.d_fake_loss_tracker = Mean(name="d_fake_loss")

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        # Unpack data
        x_mb, y_mb = data

        # Parameters
        batch_size = tf.shape(x_mb)[0]

        label_real = tf.ones((batch_size, 1))
        label_fake = tf.zeros((batch_size, 1))

        y_mb_rand = tf.random.shuffle(y_mb, seed=None, name=None)
        g_pred = self.generator(y_mb_rand, training=False)

        # Train Discriminator (Real)
        with tf.GradientTape() as tape:
            d_pred = self.discriminator([x_mb, y_mb], training=True)
            d_real_loss = self.d_loss_fn(label_real, d_pred)
        grads = tape.gradient(d_real_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        self.d_real_acc_tracker.update_state(label_real, d_pred)
        self.d_real_loss_tracker.update_state(d_real_loss)

        # Train Discriminator (Fake)
        with tf.GradientTape() as tape:
            d_pred = self.discriminator([g_pred, y_mb], training=True)
            d_fake_loss = self.d_loss_fn(label_fake, d_pred)
        grads = tape.gradient(d_fake_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        self.d_fake_acc_tracker.update_state(label_fake, d_pred)
        self.d_fake_loss_tracker.update_state(d_fake_loss)

        # Train Generator
        with tf.GradientTape() as tape:
            g_pred = self.generator(y_mb, training=True)
            d_pred = self.discriminator([g_pred, y_mb], training=False)
            g_loss = self.g_loss_fn(label_real, d_pred)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.g_acc_tracker.update_state(label_real, d_pred)
        self.g_loss_tracker.update_state(g_loss)

        # Return Metric
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.g_acc_tracker, self.d_real_acc_tracker, self.d_fake_acc_tracker,
                self.g_loss_tracker, self.d_real_loss_tracker, self.d_fake_loss_tracker]

def Generator(label_dim, target_dim):

    init = initializers.RandomNormal(stddev=0.02)

    label = layers.Input(shape=(label_dim,))

    x = layers.Dense(target_dim//8, kernel_initializer=init)(label)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(target_dim//4, kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(target_dim//2, kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(target_dim, kernel_initializer=init, activation='sigmoid')(x)

    return Model(inputs=label, outputs=x, name="Generator")


def Discriminator(target_dim, label_dim):

    init = initializers.RandomNormal(stddev=0.02)

    data = layers.Input(shape=(target_dim,))
    label = layers.Input(shape=(label_dim,))

    label_embed = layers.Dense(target_dim)(label)
    x = layers.Concatenate(axis=1)([data, label_embed])
    # x = layers.multiply([data, label_embed])

    x = layers.Dense(target_dim//2, kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(target_dim//4, kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(target_dim//8, kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1, kernel_initializer=init, activation='sigmoid')(x)

    return Model(inputs=[data, label], outputs=x, name="Discriminator")