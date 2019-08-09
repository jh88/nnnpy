import numpy as np
from tqdm import tqdm

from . import losses
from . import metrics as metrics_module
from . import optimizers


class Model():
    """Model class.
    """
    def __init__(self):
        self.layers = []
        self.training_losses = []
        self.validation_losses = []
        self.training_metrics = []
        self.validation_metrics = []

    @property
    def iterations(self):
        return len(self.training_losses)

    def add(self, layer):
        """Add new layer into the model.

        Arguments:
            layer: Layer instance, layer added into the neural network.
                If it's the first lay, it must have the `input_dim` attribute.

        Raises:
            ValueError if it's the first layer and `input_dim` is not set.
        """
        # The first layer in the model, must have `input_dim`.
        if not self.layers and layer.input_dim is None:
            raise ValueError('The first layer must have `input_dim`')
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics):
        """Compile the model.

        Arguments:
            optimizer: str or Optimizer instance, name of the optimizer
                or the optimizer instance itself.
            loss: str, the name of the loss function.
            metrics: list of str, a list of metric names.
        """
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)
        self.metrics = [metrics_module.get(metric) for metric in metrics]

        # Initialize weights, bias and optimizer accumulators
        for i, layer in enumerate(self.layers):
            if i == 0:
                # Use the `input_dim` attribute when it's the first layer.
                layer.build(layer.input_dim)
            else:
                # Use the `output_dim` of the previous layer.
                layer.build(self.layers[i-1].output_dim)
            if layer.trainable:
                self.optimizer.build(layer)

    def _forward(self, inputs, training=False):
        """Loop through the forward propagation method in each layer

        Arguments:
            inputs: `numpy.ndarray`, input data.
            training: boolean, whether it's training steps, for
                dropout layers and batch normalization layers, as
                they behave differently during training and inference.

        Returns:
            a `numpy.array` of output from the last layer.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs, training=training)
        return inputs

    def _backward(self, delta):
        """Loop through the backpropagation method in each layer
            in a reversed order.

        Arguments:
            delta: `numpy.ndarray`, the derivative of loss w.r.t the
                logits of the last layer.
        """
        # For the last layer, do not chain the derivatives from the
        # activation function, i.e. softmax, because the derivatives returned
        # from the loss function method is w.r.t the logits.
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in self.layers[-2::-1]:
            delta = layer.backward(delta)

    def _add_to_loss(self):
        """Calculate the loss from parameter regulizations.
        """
        loss = sum(layer.add_to_loss() for layer in self.layers)
        return loss

    def _get_loss(self, y, output):
        """Calculate the total loss.

        Arguments:
            y: `numpy.array`, ground truth label.
            output: `numpy.array`, predicted softmax output or logits.

        Returns:
            a `numpy.array` of losses.
        """
        # Calculate the loss from the loss function.
        loss = self.loss(y, output).mean()
        # Add the loss from regulizations.
        loss += self._add_to_loss()
        return loss

    def fit(self,
        x, y,
        batch_size,
        epochs,
        shuffle=True,
        validation_data=None,
        early_stopping_after=None,
        early_stopping_metric=None,
        verbose=1
    ):
        """Train the model.

        Arguments:
            x: `numpy.array`, training data.
            y: `numpy.array`, one hot encoded ground truth label.
            batch_size: integer, the number of records to use in each
                gradient descent step.
            epochs: integer, the number of times the training dataset
                will be passed through the network.
            shuffle: boolean, whether to shuffle the training data in
                each epoch.
            validation_data: tuple, (val_x, val_y), when set validation
                will be performed at the end of each epoch.
            early_stopping_after: integer, stop training if the validation
                loss hasn't been improved in the last n epochs.
                Must set validation data first.
            early_stopping_metric: str, the metric used for early stopping,
                if not set using losses.
            verbose: integer, 0, 1 or 2, the level of logs. It uses tqdm
                package to render the prgress bars. `0`: disable logs.
                `1`: only display the over progress bar. `2`: display
                progress bar for each epoch as well as losses and accuracies.
                This will affect the code performance, i.e. training
                takes longer.
        """
        # Loop through epochs.
        iterable = tqdm(range(epochs)) if verbose > 0 else range(epochs)
        for i in iterable:
            if verbose > 1:
                t = tqdm(total=len(x))
                # set progress bar prefix
                t.set_description('Epoch {}/{}'.format(i + 1, epochs))

            loss = 0
            outputs = []
            index_array = np.arange(len(x))
            if shuffle:
                np.random.shuffle(index_array)
            # Loop through batches.
            for index_start in range(0, len(x), batch_size):
                index_end = min(index_start+batch_size, len(x))
                batch_index = index_array[index_start:index_end]

                # Get the predictions by calling the forward method.
                output = self._forward(x[batch_index], training=True)
                outputs.append(output)
                loss += self.loss(y[batch_index], output).sum()
                # Get the derivatives for the backward step.
                delta = self.loss.delta(y[batch_index], output)
                # Update gradients in each layer by calling the backward method.
                self._backward(delta)
                # Update the parameters in each layer.
                for layer in self.layers:
                    if layer.trainable:
                        self.optimizer.update(layer, self.iterations)

                if verbose > 1:
                    # Update the displayed stats and loss after each batch
                    t.update(index_end - index_start)
                    t.set_postfix(loss=(loss / index_end + self._add_to_loss()))
            # Calculate the average for current epoch and
            # add loss from regulizations
            loss = loss / len(x) + self._add_to_loss()
            self.training_losses.append(loss)
            # Calculate accuracies.
            metrics = {
                metric.__name__: metric(y[index_array], np.vstack(outputs))
                for metric in self.metrics
            }
            self.training_metrics.append(metrics)
            # If validation data is present, calculate validation loss
            # and accuracy
            if validation_data is not None:
                val_x, val_y = validation_data

                val_metrics, val_loss = self.evaluate(val_x, val_y, batch_size)
                self.validation_losses.append(val_loss)
                self.validation_metrics.append(val_metrics)

            # Update the displayed stats for the current epoch.
            if verbose > 0:
                if validation_data is not None:
                    iterable.set_postfix(
                        loss=loss,
                        **metrics,
                        val_loss=val_loss,
                        **{'val_{}'.format(k):v for k, v in val_metrics.items()}
                    )
                else:
                    iterable.set_postfix(loss=loss, metrics=metrics)

            if verbose > 1:
                if validation_data is not None:
                    t.set_postfix(
                        loss=loss,
                        **metrics,
                        val_loss=val_loss,
                        **{'val_{}'.format(k):v for k, v in val_metrics.items()}
                    )
                else:
                    t.set_postfix(loss=loss, metrics=metrics)
                t.close()

            # Break the loop if the validation loss hasn't beem improved
            # after early_stopping_after epochs.
            if (
                validation_data is not None
                and early_stopping_after is not None
                and early_stopping_after > 0
            ):
                if early_stopping_metric is not None:
                    val_metric_socres = [
                        metric[early_stopping_metric]
                        for metric in self.validation_metrics
                    ]
                    if (
                        len(val_metric_socres) - early_stopping_after
                        > np.argmax(val_metric_socres)
                    ):
                        break
                elif (
                    len(self.validation_losses) - early_stopping_after
                    > np.argmin(self.validation_losses)
                ):
                    break

    def predict(self, x, batch_size):
        """Make predictions.

        Arguments:
            x: `numpy.array`, training data.
            batch_size: integer, the number of records to use in each
                gradient descent step.

        Returns:
            a `numpy.array` of softmax output.
        """
        outputs = []
        # Loop through batches.
        for index_start in range(0, len(x), batch_size):
            index_end = min(index_start+batch_size, len(x))
            output = self._forward(x[index_start:index_end])
            outputs.append(output)
        return np.vstack(outputs)

    def evaluate(self, x, y, batch_size):
        """Evaluate the model.

        Arguments:
            x: `numpy.array`, training data.
            y: `numpy.array`, one hot encoded ground truth label.
            batch_size: integer, the number of records to use in each
                gradient descent step.

        Returns:
            a tuple (metrics, loss), where metrics is a dict and
            loss is a number
        """
        output = self.predict(x, batch_size)

        metrics = {
            metric.__name__: metric(y, output) for metric in self.metrics
        }
        loss = self._get_loss(y, output)
        return metrics, loss
