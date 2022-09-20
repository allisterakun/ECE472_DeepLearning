#!/bin/env python3.8

"""
Allister Liu -- modified from Professor Curro's example code
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))


@dataclass
class SineModel:  # modified for sine model
    weights: np.ndarray
    bias: float
    # add mus and sigmas
    means: np.ndarray
    SDs: np.ndarray


@dataclass
class Data:
    model: SineModel
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(size=(self.num_samples, 1))
        clean_y = (
            np.sin(2 * np.pi * self.x) + self.model.bias
        )  # y_noiseless = sin(2*pi*x)
        self.y = clean_y + 0.1 * np.random.normal()  # noisy y

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices].flatten(), self.y[choices].flatten()


def compare_linear_models(a: SineModel, b: SineModel):  # modified for sine model
    for w_a, w_b in zip(a.weights, b.weights):
        print(f"{w_a:0.2f}, {w_b:0.2f}")
    for mu_a, mu_b in zip(a.means, b.means):
        print(f"{mu_a:0.2f}, {mu_b:0.2f}")
    for sig_a, sig_b in zip(a.SDs, b.SDs):
        print(f"{sig_a:0.2f}, {sig_b:0.2f}")

    print(f"{a.bias:0.2f}, {b.bias:0.2f}")


font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.1, "Standard deviation of noise random variable")
flags.DEFINE_bool("debug", False, "Set logging level to debug")


class Model(tf.Module):
    def __init__(self, rng, num_features):

        self.num_features = num_features
        self.w = tf.Variable(rng.uniform(shape=[self.num_features, 1]))
        self.b = tf.Variable(rng.uniform(shape=[1, 1]))
        self.mus = tf.Variable(rng.uniform(shape=[self.num_features, 1]))
        self.sigmas = tf.Variable(rng.uniform(shape=[self.num_features, 1]))

    def __call__(self, x):
        # the formula for y_hat
        return tf.squeeze(
            tf.transpose(self.w)
            @ tf.exp(-tf.square(x - self.mus) / (tf.square(self.sigmas)))
            + self.b
        )

    @property
    def model(self):
        return SineModel(
            self.w.numpy().reshape([self.num_features]),
            self.b.numpy().squeeze(),
            self.mus.numpy().reshape([self.num_features]),
            self.sigmas.numpy().reshape([self.num_features]),
        )


def main(a):
    logging.basicConfig()

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    data_generating_model = SineModel(
        weights=np_rng.integers(low=0, high=5, size=FLAGS.num_features),
        bias=0,
        means=np_rng.uniform(low=0, high=1, size=FLAGS.num_features),
        SDs=np_rng.uniform(low=0, high=1, size=FLAGS.num_features),
    )
    logging.debug(data_generating_model)

    data = Data(
        data_generating_model,
        np_rng,
        FLAGS.num_features,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    model = Model(tf_rng, FLAGS.num_features)
    logging.debug(model.model)

    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    bar = trange(FLAGS.num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    logging.debug(model.model)

    # print out true values versus estimates
    print("w,    w_hat")
    compare_linear_models(data.model, model.model)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

    ax[0].set_title("Sine Regression")
    ax[1].set_title("Bases")
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[0].set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 100)

    # plot y_hat
    ax[0].plot(xs, model(xs), "--")
    # truth
    ax[0].scatter(data.x, data.y, color="orange")
    ax[0].plot(xs, np.sin(2 * np.pi * xs))

    for mu, sig in zip(model.mus.numpy(), model.sigmas.numpy()):
        ax[1].plot(
            xs, np.exp(-np.square(xs - mu) / (np.square(sig)))
        )  # formula for phi

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{script_path}/fit.pdf")


if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    app.run(main)
