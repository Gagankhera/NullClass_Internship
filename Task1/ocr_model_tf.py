# ocr_model_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def build_crnn(input_shape=(32,128,1), num_classes=28):
    """
    Simple CRNN with convolutional feature extractor + BiLSTM + softmax.
    num_classes should include the blank label (for CTC) if you prefer;
    our training uses CTC where logits size = num_chars + 1 (blank).
    """
    inputs = layers.Input(shape=input_shape, name="image")
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2,1))(x)
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2,1))(x)

    # shape: (batch, h, w, c) -> collapse h to 1; get sequence along width
    shape = tf.shape(x)
    x = layers.Reshape((-1, x.shape[-1]))(x)  # (batch, time, channels)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="crnn")
    return model

class CTCModel(models.Model):
    def __init__(self, net, chars):
        super().__init__()
        self.net = net
        self.chars = chars
        self.blank_index = len(chars)  # blank at end
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        # no metrics here (can add CER/WER later)

    def train_step(self, data):
        # data: (images, labels, input_length, label_length)
        images, labels, input_length, label_length = data
        with tf.GradientTape() as tape:
            y_pred = self.net(images, training=True)  # [batch, time, num_classes]
            # Keras ctc expects shape (batch, time, num_classes)
            logit_length = tf.cast(input_length, dtype="int32")
            loss = K.ctc_batch_cost(labels, y_pred, logit_length, label_length)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        return {"loss": loss}

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    def predict_texts(self, images, max_len=32):
        preds = self.net.predict(images)
        # preds shape: (batch, time, classes)
        texts = []
        for p in preds:
            seq = p.argmax(axis=-1)
            # collapse repeats and remove blank
            collapsed = []
            prev = -1
            for c in seq:
                if c == prev: 
                    prev = c
                    continue
                if c == self.blank_index:
                    prev = c
                    continue
                collapsed.append(c)
                prev = c
            # map to chars
            text = "".join([self.chars[c] for c in collapsed if c < len(self.chars)])
            texts.append(text)
        return texts
