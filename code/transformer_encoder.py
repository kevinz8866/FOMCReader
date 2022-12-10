import tensorflow as tf
from attention_utils import TransformerBlockRankThree, TransformerBlockRankFour

#####################################################################################################################################   
class TransformerEncoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.encoder = TransformerBlockRankFour(hidden_size, useMultiHeadedAttention=False)
        self.classifier = tf.keras.layers.Dense(2,'softmax')
    
    def call(self, transcript):
        
        sentence = self.embedding_layer(transcript)
        _ = self.encoder(sentence,0)
        _ = tf.keras.layers.Flatten()(_)
        _ = tf.keras.layers.Dropout(0.5)(_)
        probs = self.classifier(_)
        return probs
    
def train_TransformerEncoder(model,X0, Y0,X1, Y1):
    
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

#####################################################################################################################################
class TransformerEncoderStatement(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.encoder = TransformerBlockRankFour(hidden_size, useMultiHeadedAttention=False)
        self.encoder2 = TransformerBlockRankFour(hidden_size, useMultiHeadedAttention=False)
        self.classifier = tf.keras.layers.Dense(2,'softmax')
    
    def call(self, inputs):
        transcript,statement=inputs[:,:4,:],inputs[:,4:,:]
        sentence = self.embedding_layer(transcript)
        statement = self.embedding_layer(statement)
        _ = self.encoder(sentence,0)
        _2 = self.encoder2(statement,0)
        _ = tf.keras.layers.Dropout(0.5)(tf.concat([_,_2],axis=1))
        _ = tf.keras.layers.Flatten()(_)
        probs = self.classifier(_)
        return probs

def train_TransformerEncoderStatement(model,X0, Y0,X1, Y1):
    
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
        loss=loss_metric, 
        metrics=[acc_metric])
    
    hist = model.fit(
        X0, Y0,
        batch_size=100,
        epochs=5,
        validation_data=(X1, Y1),
        verbose=0)
    
    return hist

#####################################################################################################################################
class TransformerEncoderStatementTone(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.encoder = TransformerBlockRankFour(hidden_size, useMultiHeadedAttention=False)
        self.encoder2 = TransformerBlockRankFour(hidden_size, useMultiHeadedAttention=False)
        self.encoder3 = TransformerBlockRankThree(window_size, useMultiHeadedAttention=False)
        self.classifier = tf.keras.layers.Dense(2,'softmax')
    
    def call(self, inputs):
        transcript,statement,tone=inputs[:,:4,:],inputs[:,4:804,:],inputs[:,804:,:]
        sentence = self.embedding_layer(transcript)
        statement = self.embedding_layer(statement)
        _ = self.encoder(sentence,0)
        _2 = self.encoder2(statement,0)
        _3 = self.encoder3(tone,0)
        _ = tf.keras.layers.Dropout(0.5)(tf.concat([_,_2,],axis=1))
        _ = tf.keras.layers.Flatten()(_)
        _3 = tf.keras.layers.Flatten()(_3)
        probs = self.classifier(tf.concat([_,_3],axis=1))
        return probs

def train_TransformerEncoderStatementTone(model,X0, Y0,X1, Y1):
    
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss=loss_metric, 
        metrics=[acc_metric])
    
    hist = model.fit(
        X0, Y0,
        batch_size=100,
        epochs=5,
        validation_data=(X1, Y1),
        verbose=0)
    
    return hist
    
    