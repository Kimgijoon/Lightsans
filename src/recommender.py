import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy


class SequentialRecommender(object):

    def __init__(self):
        pass

    def _calc_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> CategoricalCrossentropy:
        """calculate lightsans model's loss by using cross entropy

        Args:
            y_true (tf.Tensor): true value
            y_pred (tf.Tensor): predict value

        Returns:
            tf.Tensor: model loss
        """
        loss = CategoricalCrossentropy(from_logits=True)(y_true, y_pred)
        return loss

    def train(self):
        pass

    def predict(self):
        pass