import tensorflow as tf

class RNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the whether trading occurs. Serve as a baseline.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(units=rnn_size, return_sequences=False, return_state=False)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.dense2 = tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        
    def call(self, inputs):
        
        _ = self.flatten(inputs)
        _ = self.embedding_layer(_)
        _ = self.lstm(_)
        _ = self.dense(_)
        _ = self.dense2(_)
        
        return _

def train_RNN(model,X0, Y0,X1, Y1):
    
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
        loss=loss_metric, 
        metrics=[acc_metric])
    
    hist = model.fit(
        X0, Y0,
        batch_size=100,
        epochs=10,
        validation_data=(X1, Y1),
        verbose=0)
    
    return hist