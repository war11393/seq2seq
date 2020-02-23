import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, encoder_units, embedding_matrix):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.encoder_units = encoder_units
        self.gru = tf.keras.layers.GRU(self.encoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, word, hidden):
        x = self.embedding(word)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def init_hidden(self, batch_size):
        return tf.zeros((batch_size, self.encoder_units))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, decoder_units, embedding_matrix):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.decoder_units = decoder_units
        self.gru = tf.keras.layers.GRU(self.decoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.biGru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, word, hidden, enc_output, context_vector):
        # x嵌入完成后形状 == (batch_size, 1, embedding_dim)
        x = self.embedding(word)
        # x连接后形状 == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        # output 形状 == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output 形状 == (batch_size, vocab)
        x = self.fc(output)
        return x, state


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output):
        # 调整decoder中h的形状
        dec_hidden_with_x = tf.expand_dims(dec_hidden, 1)

        # 计算score，公式为 v^T*tanh(w1*h_t + w2*h_s)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(dec_hidden_with_x)))

        # attention_weights 形状 == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector 求和后形状 == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
