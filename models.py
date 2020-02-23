import tensorflow as tf

from layers import Encoder, Decoder, Attention
from utils.utils import load_embedding_matrix


class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.params = params
        self.embedding_matrix = load_embedding_matrix()
        self.encoder = Encoder(params["vocab_size"], params["vector_dim"], params["encoder_units"],
                               self.embedding_matrix)
        self.attention = Attention(params["attn_units"])
        self.decoder = Decoder(params["vocab_size"], params["vector_dim"], params["decoder_units"],
                               self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.init_hidden(self.params['batch_size'])
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call_decoder(self, dec_input, dec_hidden, enc_output, target):
        predictions = []

        for t in range(1, self.params['max_y_length'] + 2):
            context_vector, _ = self.attention(dec_hidden, enc_output)
            predict, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output, context_vector)
            dec_input = tf.expand_dims(target[:, t], 1)  # 使用teach forcing

            predictions.append(predict)

        return tf.stack(predictions, 1), dec_hidden
