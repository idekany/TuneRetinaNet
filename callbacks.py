import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from keras.utils import io_utils


class PlotLearning(tf.keras.callbacks.Callback):
    """
    Custom Callback class for plotting the training progress on the fly.
    """

    def __init__(self, eval_metrics: list = None, n_zoom: int = 200, n_update: int = 20, figname: str = "liveplot"):
        """
        :param eval_metrics: list
            A list of performance evaluation metrics to be extracted from the training logs for plotting.
        :param n_zoom: int
            The loss and the metrics plots will be zoomed after n_zoom epochs have passed in order to make smaller
            improvements also visible after the initial rapid progress of the training.
        :param n_update: int
            The plot will be updated after every n_update traning epochs.
        :param figname: str
            The name of the output figure file (without extension.)
        """

        super(PlotLearning, self).__init__()
        self.n_zoom = n_zoom
        self.n_update = n_update
        self.eval_metrics = eval_metrics
        self.n_metrics = len(eval_metrics)
        self.figname = figname
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.metr = [[] for _ in range(self.n_metrics)]
        self.val_metr = [[] for _ in range(self.n_metrics)]
        self.fig = None
        self.axes = None
        self.logs = []

    def on_train_begin(self, logs=None):
        self.fig, self.axes = \
            plt.subplots(2 + self.n_metrics, 1, sharex=False, figsize=(6, 4 + 2 * self.n_metrics))
        self.fig.subplots_adjust(bottom=0.06, top=0.98, hspace=0.15, left=0.07, right=0.8, wspace=0)

    def on_train_end(self, logs=None):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.metr = [[] for _ in range(self.n_metrics)]
        self.val_metr = [[] for _ in range(self.n_metrics)]
        self.fig.clf()
        plt.close(self.fig)
        plt.close('all')
        del self.fig

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        for ii in range(self.n_metrics):
            assert logs.__contains__(self.eval_metrics[ii]), "invalid metric: {}".format(self.eval_metrics[ii])
            self.metr[ii].append(logs.get(self.eval_metrics[ii]))
            self.val_metr[ii].append(logs.get('val_' + self.eval_metrics[ii]))
        self.i += 1

        if np.mod(self.i + 1, self.n_update) == 0:

            epochs = np.array(self.x) + 1

            self.axes[0].grid(True)
            self.axes[0].tick_params(axis='both', direction='in', labelleft=False, labelright=True)
            # clear_output(wait=True)
            self.axes[0].yaxis.tick_right()
            self.axes[0].plot(epochs, np.log10(self.losses), 'r-', label="TR", alpha=0.6)
            if self.val_losses[0] is not None:
                self.axes[0].plot(epochs, np.log10(self.val_losses), 'g-', label="CV", alpha=0.6)
                # print(np.min(self.val_losses))
            self.axes[0].set_ylabel('log(loss)')
            self.axes[0].legend(loc='upper left')

            self.axes[1].grid(True)
            self.axes[1].tick_params(axis='both', direction='in', labelleft=False, labelright=True)
            self.axes[1].yaxis.tick_right()
            log_tr_losses = np.log10(self.losses)
            self.axes[1].plot(epochs[-self.n_zoom:], log_tr_losses[-self.n_zoom:], 'r-', label="TR", alpha=0.6)
            if self.val_losses[0] is not None:
                log_val_losses = np.log10(self.val_losses)
                self.axes[1].plot(epochs[-self.n_zoom:], log_val_losses[-self.n_zoom:], 'g-', label="CV", alpha=0.6)
            if self.i > self.n_zoom:
                if self.val_losses[0] is not None:
                    minval = np.min([log_tr_losses[-self.n_zoom:].min(), log_val_losses[-self.n_zoom:].min()])
                    maxval = np.max([log_tr_losses[-self.n_zoom:].max(), log_val_losses[-self.n_zoom:].max()])
                else:
                    minval = log_tr_losses[-self.n_zoom:].min()
                    maxval = log_tr_losses[-self.n_zoom:].max()
                span = maxval - minval
                self.axes[1].set_ylim((minval - span / 10., maxval + span / 10.))
            self.axes[1].set_ylabel('log(loss)')

            for jj in range(self.n_metrics):

                self.axes[jj + 2].grid(True)
                self.axes[jj + 2].tick_params(axis='both', direction='in', labelleft=False, labelright=True)
                self.axes[jj + 2].yaxis.tick_right()
                self.axes[jj + 2].plot(epochs[-self.n_zoom:], self.metr[jj][-self.n_zoom:], 'r-', label="TR")
                if self.val_metr is not None:
                    self.axes[jj + 2].plot(epochs[-self.n_zoom:], self.val_metr[jj][-self.n_zoom:], 'g-', label="CV")
                if self.i > self.n_zoom:

                    if self.val_metr[jj][0] is not None:  # check if there is validation data for this metric
                        minval = np.min([np.array(self.metr[jj][-self.n_zoom:]).min(),
                                         np.array(self.val_metr[jj][-self.n_zoom:]).min()])
                        maxval = np.max([np.array(self.metr[jj][-self.n_zoom:]).max(),
                                         np.array(self.val_metr[jj][-self.n_zoom:]).max()])
                        span = maxval - minval
                        self.axes[jj + 2].set_ylim((minval - span / 10., maxval + span / 10.))
                    else:  # if there is no val. data for this metric, then compute extrema from training data only
                        minval = np.array(self.metr[jj][-self.n_zoom:]).min()
                        maxval = np.array(self.metr[jj][-self.n_zoom:]).max()
                        span = maxval - minval
                        self.axes[jj + 2].set_ylim((minval - span / 10., maxval + span / 10.))
                if jj == self.n_metrics - 1:
                    self.axes[jj + 2].set_xlabel('epoch')
                self.axes[jj + 2].set_ylabel(self.eval_metrics[jj])

            plt.savefig(self.figname + '.png', format='png')
            for ax in self.axes:
                ax.cla()


class EarlyStopping(tf.keras.callbacks.Callback):
    """
    This is a customized version of the keras EarlyStopping class.

    Stop training when a monitored metric has stopped improving.
    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.
    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.
    Args:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
        displays messages when the callback takes an action.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 checkpoint_best_weights=False,
                 checkpoint_dir='.',
                 optimizer=None):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.checkpoint_best_weights = checkpoint_best_weights
        self.best_weights = None
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = optimizer

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or
                    self.monitor.endswith('auc') or self.monitor.endswith('iou') or
                    self.monitor.endswith('map')):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
        self.model.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        'Restoring model weights from the end of the best epoch: '
                        f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:

            if self.checkpoint_best_weights:

                if self.optimizer is None:
                    self.ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
                else:
                    self.ckpt = tf.compat.v2.train.Checkpoint(model=self.model, optimizer=self.optimizer)

                self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir,
                                                          checkpoint_name='early_stopping_ckpt', max_to_keep=None)
                self.manager.save()
            if self.verbose > 0:
                io_utils.print_msg(f'Epoch {self.stopped_epoch + 1}: early stopping')

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
