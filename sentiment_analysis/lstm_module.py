from functools import reduce

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score

from src.paul_sentiment_analysis.prepare_data import prepare_lstm_datasets


def lstm_layers(vocab_size, max_tokens,
                n_embedding_features, dropout=0.1):
    return Sequential([
        Input(max_tokens),
        Embedding(vocab_size + 1, n_embedding_features, input_length=max_tokens),
        Bidirectional(layer=LSTM(max_tokens, dropout=dropout))
    ])


def dense_layers(input_dim, hidden_layers=(64,), dropout=0.1):
    return Sequential([
                          Input(input_dim),
                          Dropout(dropout),
                      ] + reduce(lambda a, b: a + list(b),
                                 zip([Dense(hidden_layer, activation='relu') for hidden_layer in hidden_layers],
                                     [Dropout(dropout) for _ in hidden_layers]), []) \
                      + [Dense(3, activation='softmax')])


def tensorflow_train(train, test, vocab_size=256, max_tokens=64, dropout=0.1):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score',
        min_delta=0,
        patience=4,
        verbose=0,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )

    model = Sequential([
        lstm_layers(vocab_size, max_tokens,
                    n_embedding_features=vocab_size//16, dropout=dropout),
        dense_layers(2 * max_tokens,
                     dropout=dropout,
                     hidden_layers=(max_tokens,)
                     )
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy', F1Score(3, average='weighted')])
    print(model.summary())
    print(tf.config.list_physical_devices('GPU'))
    model.fit(train, validation_data=test, epochs=20,
              batch_size=4, verbose=1, shuffle=True,
              callbacks=[early_stopping])
    model.save('lstm.h5')


if __name__ == '__main__':
    train, test = prepare_lstm_datasets(
        save_vectorize_layer='tf_vectorizer',
        vocab_size=2048,
        max_tokens=64,
    )
    tensorflow_train(
        train, test,
        vocab_size=2048,
        max_tokens=64
    )
