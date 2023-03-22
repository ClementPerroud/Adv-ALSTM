import tensorflow as tf
from keras.engine import data_adapter

def hinge_acc(y_true, y_pred):
    y_pred = tf.math.sign(y_pred)
    return tf.keras.metrics.binary_accuracy(
        (y_true + 1)/2, (y_pred + 1) / 2
    )

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.av_W = self.add_weight(
            name='att_W', dtype=tf.float32,
            shape=[self.units, self.units],
            initializer="glorot_uniform"
        )
        self.av_b = self.add_weight(
            name='att_h', dtype=tf.float32,
            shape=[self.units],
            initializer="zeros"
        )
        self.av_u = self.add_weight(
            name='att_u', dtype=tf.float32,
            shape=[self.units],
            initializer="glorot_uniform"
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
    def __init__(self, units, epsilon, beta, learning_rate = 1E-2, l2 = None, attention = True, hinge = True, adversarial_training = True):
        super().__init__()
        self.epsilon = epsilon
        self.beta = beta
        self.hinge = hinge
        self.adversarial_training = adversarial_training
        
        if attention:
            self.model_latent_rep = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units, activation = "tanh"),
                tf.keras.layers.LSTM(units, return_sequences = True),
                TemporalAttention(units),
            ])
        else:
            self.model_latent_rep = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units, activation = "tanh"),
                tf.keras.layers.LSTM(units, return_sequences = False),
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
    
    @tf.function
    def get_perturbation(self, e, y, sample_weight):
        y_pred = self.model_prediction(e, training = True)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=None)
        perturbations = tf.gradients(loss, [e])[0]
        tf.stop_gradient(perturbations)
        perturbations = perturbations / tf.norm(perturbations, ord='euclidean', axis=-1, keepdims = True)
        return perturbations
        
    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape(persistent = True) as tape:
            e = self.model_latent_rep(x, training=True)
            
            y_pred = self.model_prediction(e, training = True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=None)
            
            with tape.stop_recording():
                perturbations = tape.gradient(loss, e)#self.get_perturbation(e, y, sample_weight)
            
            perturbations = perturbations / tf.norm(perturbations, ord='euclidean', axis=-1, keepdims = True)
                
            if self.adversarial_training:
                 # Normalize
                e_adv = e + self.epsilon* perturbations
                y_pred_adv = self.model_prediction(e_adv, training = True)

                loss += self.beta*self.compiled_loss(y, y_pred_adv, sample_weight, regularization_losses=None) + tf.add_n(self.losses)
        
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        del tape
        return self.compute_metrics(x, y, y_pred, sample_weight)
    
    def fit(self, *args, **kwargs):
        if self.hinge:
            args = list(args)
            if len(args) > 1:
                args[1] = args[1]*2 - 1
            elif "y" in kwargs.keys():
                kwargs['y'] = kwargs['y']*2 - 1

            if "validation_data" in kwargs.keys():
                kwargs['validation_data'] = (kwargs['validation_data'][0], kwargs['validation_data'][1]*2 -1)
        super().fit(*args, **kwargs)
        
