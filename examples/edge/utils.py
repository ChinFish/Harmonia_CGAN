"""Utility functions for GANs.
    Class:
        plt_result: for GAN training plots.
    
    Function:

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score


class plt_result:
    def __init__(self, no, batch_size):
        self.no = no
        self.batch_size = batch_size
        self.step = int(no / batch_size)
        self.xlim_left = 0
        self.xlim_right = 0
        self.ylim_left = 0
        self.ylim_right = 0

    def plt_clear(self):
        plt.clf()
        plt.cla()
        plt.close("all")

    def forward(self, x):
        return x // self.step

    def inverse(self, x):
        return x * self.step

    def clip_data(self, data):
        lower_bound = np.mean(data) - 3 * np.std(data)
        upper_bound = np.mean(data) + 3 * np.std(data)
        return np.clip(data, lower_bound, upper_bound)

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def plt_loss(self, data_lists, legends):
        self.plt_clear()
        plt.figure(figsize=(6.4, 4.8))
        fig, ax = plt.subplots()
        for data in data_lists:
            # data = self.clip_data(data)
            data = self.moving_average(data, 10)
            ax.plot(data)
        ax.legend(legends)
        ax.set_title('Model loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Step')
        secax = ax.secondary_xaxis('top', functions=(self.forward, self.inverse))
        secax.set_xlabel('Epoch')
        return fig

    def plt_acc(self, data_lists, legends, have_steps=True, smooth=10, title='Model Acc.'):
        self.plt_clear()
        plt.figure(figsize=(6.4, 4.8))
        fig, ax = plt.subplots()
        for data in data_lists:
            if smooth > 0:
                # data = self.clip_data(data)
                data = self.moving_average(data, smooth)
            ax.plot(data)
        ax.legend(legends)
        ax.set_title(title)
        ax.set_ylabel('Acc.')
        if have_steps == True:
            ax.set_xlabel('Step')
            secax = ax.secondary_xaxis('top', functions=(self.forward, self.inverse))
            secax.set_xlabel('Epoch')
        else:
            ax.set_xlabel('Epoch')
        return fig

    def plt_f1(self, data_lists, legends, have_steps=True, smooth=10):
        self.plt_clear()
        plt.figure(figsize=(6.4, 4.8))
        fig, ax = plt.subplots()
        for data in data_lists:
            if smooth > 0:
                # data = self.clip_data(data)
                data = self.moving_average(data, smooth)
            ax.plot(data)
        ax.legend(legends)
        ax.set_title('F1 Score')
        ax.set_ylabel('F1 Score')
        if have_steps == True:
            ax.set_xlabel('Step')
            secax = ax.secondary_xaxis('top', functions=(self.forward, self.inverse))
            secax.set_xlabel('Epoch')
        else:
            ax.set_xlabel('Epoch')
        return fig

    def plt_err(self, data_lists, legends, have_steps=True, smooth=10):
        self.plt_clear()
        plt.figure(figsize=(6.4, 4.8))
        fig, ax = plt.subplots()
        for data in data_lists:
            if smooth > 10:
                # data = self.clip_data(data)
                data = self.moving_average(data, smooth)
            ax.plot(data)
        ax.legend(legends)
        ax.set_title('Error rate')
        ax.set_ylabel('Error rate')
        if have_steps == True:
            ax.set_xlabel('Step')
            secax = ax.secondary_xaxis('top', functions=(self.forward, self.inverse))
            secax.set_xlabel('Epoch')
        else:
            ax.set_xlabel('Epoch')
        return fig

    def plt_minor_pct(self, minor_pct_list):
        self.plt_clear()
        plt.figure(figsize=(6.4, 4.8))
        fig, ax = plt.subplots()
        ax.plot(minor_pct_list)
        ax.set_title('Minor Allele Percent')
        ax.set_ylabel('Percent')
        ax.set_xlabel('Step')
        secax = ax.secondary_xaxis('top', functions=(self.forward, self.inverse))
        secax.set_xlabel('Epoch')
        return fig

    def calc_pca_lim(self, true_gt_pca):
        xlim_buffer = (max(true_gt_pca[:, 0]) - min(true_gt_pca[:, 0])) / 8
        self.xlim_left = min(true_gt_pca[:, 0]) - xlim_buffer
        self.xlim_right = max(true_gt_pca[:, 0]) + xlim_buffer

        ylim_buffer = (max(true_gt_pca[:, 1]) - min(true_gt_pca[:, 1])) / 8
        self.ylim_left = min(true_gt_pca[:, 1]) - ylim_buffer
        self.ylim_right = max(true_gt_pca[:, 1]) + ylim_buffer

    def plt_mnist(self, epoch, x_fake):
        num_row = 2
        num_col = 5
        x_fake = np.rint(x_fake)
        self.plt_clear()
        fig, axes = plt.subplots(2, 5, figsize=(2 * num_col, 2 * num_row + 0.5))
        for i in range(10):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(x_fake[i].reshape(28, 28).astype(int), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.suptitle("Epoch:{}".format(epoch))
        return fig

    def plt_pca(self, epoch, pca_data_lists, legends):
        self.plt_clear()
        plt.figure(figsize=(5.0, 5.0))
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        ax.set_title("Epoch:{}".format(epoch))
        ax.set_xlim(self.xlim_left, self.xlim_right)
        ax.set_ylim(self.ylim_left, self.ylim_right)

        # Plotting Loop
        i = 0
        for pca_data in pca_data_lists:
            ax.scatter(pca_data[:, 0], pca_data[:, 1], label=legends[i])
            i += 1

        ax.legend(legends)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        return fig


def tf2np(data):
    try:
        data = data.numpy()
    except:
        pass
    return np.rint(data)


def sk_f1_score(y_true, y_pred):

    y_true = tf2np(y_true).flatten()
    y_pred = tf2np(y_pred).flatten()
    f1_val = f1_score(y_true, y_pred, average="macro")

    return f1_val


def sk_acc_score(y_true, y_pred):

    y_true = tf2np(y_true).flatten()
    y_pred = tf2np(y_pred).flatten()
    acc_val = accuracy_score(y_true, y_pred)

    return acc_val


def calc_pct(x_pred):

    x_pred = tf2np(x_pred)
    value, count = np.unique(x_pred, return_counts=True)

    if value.shape[0] == 2:
        return count[1] / sum(count)
    elif value[0] == 0:
        return 0
    elif value[0] == 1:
        return 1
    else:
        raise ValueError("Error")


def plt_clear():
    plt.clf()
    plt.cla()
    plt.close()


def add_noise(data):
    noise = abs(np.random.normal(0, .1, data.shape))
    noise = np.where(data == 0, 0.5 * noise, 0 - 3 * noise)

    data = data + noise
    return data
