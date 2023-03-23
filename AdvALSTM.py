import tensorflow as tf
from keras.engine import data_adapter

def hinge_acc(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(
        y_true, y_pred, threshold= 0
    )

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.av_W = self.add_weight(
            name='att_W', dtype=tf.float32,
            shape=[self.units, self.units],
            initializer="glorot_uniform",
            trainable= True
        )
        self.av_b = self.add_weight(
            name='att_h', dtype=tf.float32,
            shape=[self.units],
            initializer="zeros",
            trainable= True
        )
        self.av_u = self.add_weight(
            name='att_u', dtype=tf.float32,
            shape=[self.units],
            initializer="glorot_uniform",
            trainable= True
        )
        super().build(input_shape)
    def call(self, h):
        a_laten = tf.tanh(
            tf.tensordot(h, self.av_W,
                         axes=1) + self.av_b)
        a_scores = tf.tensordot(a_laten, self.av_u,
                                     axes=1,
                                     name='scores')
        a_alphas = tf.nn.softmax(a_scores, name='alphas')

        a_con = tf.reduce_sum(
            h * tf.expand_dims(a_alphas, -1), 1)
        
        fea_con = tf.concat(
            [h[:, -1, :], a_con],
            axis=1)
        return fea_con
    
class AdvALSTM(tf.keras.models.Model):
    def __init__(self, units, epsilon = 1E-3, beta =  5E-2, learning_rate = 1E-2, dropout = None, l2 = None, attention = True, hinge = True, adversarial_training = True, random_perturbations = False):
        super().__init__()
        self.epsilon = tf.constant(epsilon)
        self.beta = tf.constant(beta)
        self.hinge = hinge
        self.adversarial_training = adversarial_training
        self.random_perbutations = random_perturbations
        
        
        if attention:
            self.model_latent_rep = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units, activation = "tanh"),
                tf.keras.layers.Dropout(dropout), 
                tf.keras.layers.LSTM(units, return_sequences = True, dropout = dropout),
                TemporalAttention(units)
            ])
        else:
            self.model_latent_rep = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units, activation = "tanh"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.LSTM(units, return_sequences = False, dropout = dropout),
            ])
        self.model_prediction = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation=None if hinge else "sigmoid",
                kernel_regularizer = tf.keras.regularizers.L2(l2),
                bias_regularizer = tf.keras.regularizers.L2(l2))
        ])
        
        
        self.compile(
            loss = "hinge" if hinge else "binary_crossentropy",
            optimizer = tf.keras.optimizers.Adam(learning_rate),
            metrics = [hinge_acc if hinge else "acc"]
        )
            
    def call(self, x):
        x = self.model_latent_rep(x)
        x = self.model_prediction(x)
        return x
    
    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            e = self.model_latent_rep(x, training=True)
            with tf.GradientTape(watch_accessed_variables = False) as tape_pertubations:
                tape_pertubations.watch(e)
                y_pred = self.model_prediction(e, training = True)
                loss = self.compiled_loss(y, y_pred, sample_weight)

            
            if self.adversarial_training:
                with tape.stop_recording():
                    if self.random_perbutations:
                        perturbations = tf.random.normal(shape=tf.shape(e))
                    else:
                        perturbations = tf.math.sign(tape_pertubations.gradient(loss, e))
                    tf.stop_gradient(perturbations)

                
                e_adv = e + self.epsilon * tf.norm(e, ord = "euclidean", axis= -1, keepdims = True) * tf.math.l2_normalize(perturbations, axis = -1, epsilon=1e-8)
                y_pred_adv = self.model_prediction(e_adv, training = True)

                adv_loss = self.compiled_loss(y, y_pred_adv, sample_weight)
                loss += self.beta*adv_loss + tf.add_n(self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)
        
