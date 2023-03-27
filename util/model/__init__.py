from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
from keras.utils import plot_model


class Model(tf.keras.Model):
    def __init__(self, input_shape_structured: Tuple[int], input_shape_unstructured: Tuple[None | int, ...]):
        """
        Initializes the Model instance.

        Args:
        - input_shape_structured: A tuple representing the shape of structured input data.
        - input_shape_unstructured: A tuple representing the shape of unstructured input data.
        """
        super(Model, self).__init__()
        self.nn = NeuralNetwork(input_shape_structured, input_shape_unstructured)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Forward pass for the custom neural network.

        Args:
        - inputs: A tuple containing two input tensors (structured and unstructured data).

        Returns:
        - The output tensor from the neural network.
        """
        input_01, input_02 = inputs
        return self.nn.model([input_01, input_02])

    def summary(self) -> None:
        """
        Prints a summary of the custom neural network model.
        """
        self.nn.summary()

    def plot(self, filepath: str = './model.png') -> None:
        """
        Plots the custom neural network model.

        Args:
        - filepath: The file path to save the plotted model.
        """
        self.nn.plot(filepath)

    def train(self, x_train: List[np.ndarray], y_train: np.ndarray, batch_size: int = 32, epochs: int = 10, validation_split: float = 0.2, callbacks=None) -> tf.keras.callbacks.History:
        """
        Trains the custom neural network model.

        Args:
        - x_train: A list containing two tensors (structured and unstructured data) for training.
        - y_train: A tensor representing the training labels.
        - batch_size: The number of samples per gradient update.
        - epochs: The number of times to iterate over the entire dataset.
        - validation_split: The fraction of the training data to be used as validation data.
        - callbacks: A list of Keras callbacks to be called during training.

        Returns:
        - A Keras History object that contains training loss and metric values.
        """
        return self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)

    def test(self, x_test: List[np.ndarray], y_test: np.ndarray) -> Tuple[float, float]:
        """
        Tests the custom neural network model.

        Args:
        - x_test: A list containing two tensors (structured and unstructured data) for testing.
        - y_test: A tensor representing the test labels.

        Returns:
        - A tuple containing the test loss and test accuracy.
        """
        return self.evaluate(x_test, y_test)

    def save_weights(self, filepath: str) -> None:
        """
        Saves the custom neural network model weights.

        Args:
        - filepath: The file path to save the model weights.
        """
        self.nn.model.save_weights(filepath)

    def load_weights(self, filepath: str):
        """
        Loads the model weights from the specified file.

        Args:
        - filepath: The path to the file where the weights are stored.
        """
        self.nn.model.load_weights(filepath)


class NeuralNetwork:

    def __init__(self, input_shape_structured, input_shape_unstructured):

        self.input_01 = tf.keras.Input(shape=input_shape_structured)
        self.input_02 = tf.keras.Input(shape=input_shape_unstructured)

        self.build_model()

    def conv_layer(self,
                   input: tf.Tensor,
                   input_channel: int,
                   output_channel: int,
                   coef: tf.Tensor,
                   mean: float = 0.0,
                   std: float = 1.,
                   bias: float = 0.0,
                   filter_size: int = 3,
                   name: Optional[str] = None,
                   trainable: bool = True,
                   padding: str = "SAME",
                   strides: int = 1) -> tf.Tensor:
        """
        Creates a 3D convolutional layer using TensorFlow.

        Args:
        - input: The input tensor to the layer.
        - input_channel: The number of input channels (i.e., the depth of the input tensor).
        - output_channel: The number of output channels (i.e., the number of filters in the layer).
        - coef: The weight coefficients to use in the convolution operation.
        - mean: The mean of the initial normal distribution for the filter weights (default: 0.0).
        - std: The standard deviation of the initial normal distribution for the filter weights (default: 1.0).
        - bias: The initial value of the biases (default: 0.0).
        - filter_size: The size of the filters in the layer (default: 3).
        - name: The name of the layer (default: None). If not provided, a default name 'conv_layer' will be used.
        - trainable: Whether the variables in the layer should be trainable (default: True).
        - padding: The type of padding to use in the convolution operation (default: 'SAME').
        - strides: The stride length of the filters in the layer (default: 1).

        Returns:
        - The output tensor of the layer.
        """
        if name is None:
            name = 'conv_layer'

        with tf.name_scope(name):
            shape = [filter_size, filter_size, filter_size, input_channel, output_channel]

            W = tf.Variable(initial_value=tf.random.truncated_normal(shape=shape, mean=mean, stddev=std), trainable=trainable, name="W")
            B = tf.Variable(initial_value=tf.constant(bias, shape=[output_channel]), trainable=trainable, name="B")

            conv = tf.nn.conv3d(input, W * (1 + coef), strides=[1, strides, strides, strides, 1], padding=padding)
            conv = tf.nn.bias_add(conv, B)

        return conv

    def dense(self, input: tf.Tensor, output_channel: int, name: str, activation: Optional[str] = None) -> tf.Tensor:
        """
        Adds a fully connected (dense) layer to the network.

        Args:
        - input: The input tensor to the layer.
        - output_channel: The dimensionality of the output space (i.e., the number of neurons in the layer).
        - name: The name of the layer.
        - activation: The activation function to use. If None, no activation function will be used.

        Returns:
        - The output tensor of the layer.
        """
        with tf.name_scope(name):
            return tf.keras.layers.Dense(output_channel, activation=activation)(input)

    def conv(self, input: tf.Tensor, output_channel: int, coef: tf.Tensor, name: str, activation: str = 'relu') -> tf.Tensor:
        """
        Adds a convolutional layer to the network.

        Args:
        - input: The input tensor to the layer.
        - output_channel: The number of output channels (i.e., the number of filters in the layer).
        - coef: The weight coefficients to use in the convolution operation.
        - name: The name of the layer.
        - activation: The activation function to use.

        Returns:
        - The output tensor of the layer.
        """
        layer = self.conv_layer(input=input, input_channel=int(input.shape[-1]), output_channel=output_channel, coef=coef, name=name)
        return tf.keras.layers.Activation(activation, name='%s_activation' % name)(layer)

    def conv_only(self, input: tf.Tensor, output_channel: int, coef: tf.Tensor, name: str, activation: str = 'relu', iteration: int = 1) -> tf.Tensor:
        """
        Adds a stack of convolutional layers to the network.

        Args:
        - input: The input tensor to the layer.
        - output_channel: The number of output channels (i.e., the number of filters in each layer).
        - coef: The weight coefficients to use in the convolution operation.
        - name: The name of the layer.
        - activation: The activation function to use.
        - iteration: The number of convolutional layers to add to the stack.

        Returns:
        - The output tensor of the layer.
        """
        layer = self.conv(input=input, output_channel=output_channel, coef=coef, activation=activation, name='%s_layer01' % name)
        for n_iter in range(0, iteration - 1):
            layer = self.conv(input=layer, output_channel=output_channel, coef=coef, activation=activation, name='%s_layer_%02d' % (name, n_iter + 2))
        return layer

    def conv_with_polling(self, input: tf.Tensor, output_channel: int, coef: tf.Tensor, name: str, pool_size: int = 2, activation: str = 'relu', iteration: int = 1) -> tf.Tensor:
        """
        Adds a stack of convolutional layers with max pooling to the network.

        Args:
        - input: The input tensor to the layer.
        - output_channel: The number of output channels (i.e., the number of filters in each layer).
        - coef: The weight coefficients to use in the convolution operation.
        - name: The name of the layer.
        - pool_size: The size of the max pooling window.
        - activation: The activation function to use.
        - iteration: The number of convolutional layers to add to the stack.

        Returns:
        - The output tensor of the layer.
        """
        layer = self.conv_only(input=input, output_channel=output_channel, coef=coef, name=name, activation=activation, iteration=iteration)
        return tf.keras.layers.MaxPooling3D(pool_size=pool_size, padding="SAME")(layer)

    def build_model(self):
        structured_layer_01 = self.dense(self.input_01, output_channel=100, activation='relu', name='structured_layer_01')
        structured_layer_02 = self.dense(structured_layer_01, output_channel=200, activation='relu', name='structured_layer_02')
        structured_layer_03 = self.dense(structured_layer_02, output_channel=200, activation='relu', name='structured_layer_03')
        structured_layer_04 = self.dense(structured_layer_03, output_channel=100, activation='relu', name='structured_layer_04')
        structured_layer_05 = self.dense(structured_layer_04, output_channel=27, activation=None, name='structured_layer_05')
        weight_vector = tf.reduce_max(structured_layer_05, axis=0)
        weight_vector = tf.keras.layers.Normalization()(tf.reshape(weight_vector, shape=(3, 3, 3, 1, 1)))

        unstructured_layer_01 = self.conv_with_polling(input=self.input_02, output_channel=64, coef=weight_vector, name='unstructured_layer_01', iteration=2)
        unstructured_layer_02 = self.conv_with_polling(input=unstructured_layer_01, output_channel=128, coef=weight_vector, name='unstructured_layer_02', iteration=2)
        unstructured_layer_03 = self.conv_with_polling(input=unstructured_layer_02, output_channel=256, coef=weight_vector, name='unstructured_layer_03', iteration=4)
        unstructured_layer_04 = self.conv_with_polling(input=unstructured_layer_03, output_channel=512, coef=weight_vector, name='unstructured_layer_04', iteration=4)
        unstructured_layer_05 = self.conv_with_polling(input=unstructured_layer_04, output_channel=512, coef=weight_vector, name='unstructured_layer_05', iteration=4)

        layer = tf.reduce_mean(unstructured_layer_05, axis=1)
        layer = tf.keras.layers.Flatten()(layer)
        layer = self.dense(layer, 4096, name='dense_01')
        layer = self.dense(layer, 4096, name='dense_02')
        layer = self.dense(layer, 4, name='dense_03', activation='softmax')

        self.model = tf.keras.Model(inputs=[self.input_01, self.input_02], outputs=layer)

    def summary(self):
        self.model.summary()

    def plot(self, filepath: str = './model.png'):
        plot_model(self.model,
                   to_file=filepath,
                   show_shapes=True,
                   show_dtype=True,
                   show_layer_names=True,
                   expand_nested=True,
                   dpi=150,
                   show_layer_activations=True,
                   )
