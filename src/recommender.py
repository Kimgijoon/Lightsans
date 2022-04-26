import math
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.utils import Progbar
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_ranking as tfr

from src.model import LightSANs
import src.optimization as optimization


class SequentialRecommender(object):

    def __init__(self, config: Dict[str, any], batch_size: int, epochs: int, topk: int, is_training: bool=True):
        """blahblah

        Args:
            config: (Dict[str, any]): Configuration of model hyperparameters
            batch_size (int): Number of samples per gradient update
            epochs (int): Number of epochs to train the model
            topk (int): Cutoff of how many items are considered in the metric
            is_training (bool): Python boolean indicating whether the layer should behave 
	    	in training mode (adding dropout) or in inference mode (doing nothing)
        """
        self.config = config
        self.learning_rate = config['learning_rate']
        self.n_items = config['n_items']
        self.trainset_size = config['train']
        self.epochs = epochs

        self.model = LightSANs(config=self.config, is_training=is_training)
        self.steps_per_epoch = math.ceil(self.trainset_size / batch_size)
        num_train_steps = self.steps_per_epoch * self.epochs
        num_warmup_steps = int(0.1 * num_train_steps)
        self.optimizer = optimization.create_optimizer(init_lr=self.learning_rate,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        
        self.train_loss, self.valid_loss = Mean(), Mean()
        self.train_ndcg = tfr.keras.metrics.NDCGMetric(topn=topk)
        self.valid_ndcg = tfr.keras.metrics.NDCGMetric(topn=topk)
        self.train_mrr = tfr.keras.metrics.MRRMetric(topn=topk)
        self.valid_mrr = tfr.keras.metrics.MRRMetric(topn=topk)
        
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

    @tf.function(jit_compile=True)
    def _compiled_step(self,
                       features: Dict[str, tf.data.Dataset],
                       labels: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        with tf.GradientTape() as tape:
            logits = self.model(features)
            loss = self._calc_loss(labels, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        return loss, logits, gradients

    def train_step(self,
                   features: Dict[str, tf.data.Dataset],
                   labels: tf.data.Dataset):

        loss, logits, gradients = self._compiled_step(features, labels)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_ndcg(labels, logits)
        self.train_mrr(labels, logits)

    @tf.function(jit_compile=True)
    def valid_step(self,
                   features: Dict[str, tf.data.Dataset],
                   labels: tf.data.Dataset):

        logits = self.model(features)
        loss = self._calc_loss(labels, logits)

        self.valid_loss(loss)
        self.valid_ndcg(labels, logits)
        self.valid_mrr(labels, logits)

    def train(self,
              trainset: tf.data.Dataset,
              validset: tf.data.Dataset):
        """train model

        Args:
            trainset (tf.data.Dataset): Tensorflow dataset class (train data)
            validset (tf.data.Dataset): Tensorflow dataset class (test data)
        """
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch+1}/{self.epochs}')
            
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.train_ndcg.reset_states()
            self.valid_ndcg.reset_states()
            self.train_mrr.reset_state()
            self.valid_mrr.reset_state()
            
            p_bar = Progbar(self.steps_per_epoch)
            for step, (train_x, train_y) in enumerate(trainset):
                self.train_step(train_x, train_y)
                values = [
		            ('loss', self.train_loss.result()),
                    ('ndcg', self.train_ndcg.result()),
                    ('mrr', self.train_mrr.result())]
                p_bar.update(step, values=values)
                
            for valid_x, valid_y in validset:
                self.valid_step(valid_x, valid_y)
            values = [
                ('val_loss', self.valid_loss.result()),
                ('val_ndcg', self.valid_ndcg.result()),
                ('val_mrr', self.valid_mrr.result())]
            p_bar.add(1, values=values)

    def predict(self):
        pass