import tensorflow as tf


class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup):
        super(LrSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup = warmup

    def __call__(self, step):
        param_1 = tf.math.rsqrt(step)
        param_2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(param_1, param_2)
