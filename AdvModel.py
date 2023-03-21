import tensorflow as tf

class AdvModel(tf.keras.Model):
    def __init__(self, epsilon, beta, input_shape, model_latent_space, model_classifier, *args, **kwargs):
        

        inputs = tf.keras.layers.Input(shape= input_shape)
        x = model_latent_space(inputs)
        x = model_classifier(x)
        super().__init__(inputs = inputs, outputs = x, **kwargs)
    
        self.epsilon = epsilon
        self.beta = beta
        self.model_latent_space = model_latent_space
        self.model_classifier = model_classifier
        
         
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        trainable_vars = self.trainable_variables
        
        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch(trainable_vars)
            e_s = self.model_latent_space(x, training=True)
            
            
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(e_s)
                y_pred = self.model_classifier(e_s, training = True)
                loss = self.compiled_loss(y, y_pred)
            
            g_s = tf.stop_gradient( tape2.gradient(loss, e_s) )
        
            e_adv = e_s + self.epsilon *g_s / tf.norm(g_s, ord=2, axis=1)
            y_adv = self.model_classifier(e_adv, training = True)
            
            adv_loss +=   self.beta * self.compiled_loss(y, y_adv)
            loss +=  self.losses + adv_loss


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape1.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)