import tensorflow as tf
import tensorflow.python.keras.backend as K

class Embed_layer(tf.keras.Layer):
    def __init__(self, k, sparse_feature_columns):
        super(Embed_layer, self).__init__()
        self.emb_layers = [tf.keras.layers.Embedding(feat['feat_onehot_dim'], k) for feat in sparse_feature_columns]

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("The dim of inputs should be 2, not %d" % (K.ndim(inputs)))

        emb = tf.transpose(
            tf.convert_to_tensor([layer(inputs[:, i])
                                  for i, layer in enumerate(self.emb_layers)]),
            [1, 0, 2])
        emb = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        return emb

class Res_layer(tf.keras.Layer):
    def __init__(self, hidden_units):
        super(Res_layer, self).__init__()
        self.dense_layer = [tf.keras.layers.Dense(i, activation='relu') for i in hidden_units]

    def build(self, input_shape):
        self.output_layer = tf.keras.layers.Dense(input_shape[-1], activation=None)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("The dim of inputs should be 2, not %d" % (K.ndim(inputs)))

        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        x = self.output_layer(x)

        output = inputs + x
        return tf.nn.relu(output)

class deep_crossing(tf.keras.Model):
    def __init__(self, feature_columns, k, hidden_units, res_layer_num):
        super(deep_crossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = Embed_layer(k, self.sparse_feature_columns)
        self.res_layer = [Res_layer(hidden_units) for _ in range(res_layer_num)]
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        emb = self.embed_layer(sparse_inputs)
        x = tf.concat([dense_inputs, emb], axis=-1)
        for layer in self.res_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output