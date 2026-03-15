import tensorflow as tf
from src.visualization.plots import save_metric_plots


class LivePlotCallback(tf.keras.callbacks.Callback):
    """Keras callback that overwrites training plot PNGs after each epoch.

    Open the output PNGs in a viewer (e.g. VS Code) and they will refresh
    as training progresses — one file per metric, updated in place.
    """

    def __init__(self, output_dir="outputs", filenames=None):
        super().__init__()
        self.output_dir = output_dir
        self.filenames = filenames
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        save_metric_plots(self.history, self.output_dir, self.filenames)
