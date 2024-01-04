import pickle

from sklearn.model_selection import train_test_split

from .tokenize_data import create_sklearn_vectorizer, create_tensorflow_vectorize_layer
from .tokenize_data import save_tensorflow_vectorize_layer, load_tensorflow_vectorize_layer
from .tokenize_data import score_to_sentiment, read_raw_data, tokenize_text, filter_thai
from .tokenize_data import sklearn_vectorize, tensorflow_tokenize


def prepare_train_test(test_size=0.2, seed=333):
    data = score_to_sentiment(read_raw_data().drop_duplicates(ignore_index=True))
    data = data[data['review'].apply(tokenize_text).apply(lambda x: len(x) > 0)]
    train, test = train_test_split(data, test_size=test_size,
                                   stratify=data['type'],
                                   random_state=seed)
    return train, test, data


def upsample_train(train):
    return train.loc[train.index.repeat(((3 - train.rating) ** 1).astype(int))]


def prepare_tfidf_data(test_size=0.2, upsample=True, save_vectorizer=False):
    train, test, data = prepare_train_test(test_size)
    train['review'] = train['review'].apply(filter_thai)
    test['review'] = test['review'].apply(filter_thai)
    train = train.sort_values('rating')
    sklearn_vectorizer = create_sklearn_vectorizer(data['review'])
    if save_vectorizer:
        with open(save_vectorizer + '.pkl', 'wb') as f:
            pickle.dump(sklearn_vectorizer, f)
    if upsample:
        print('before upsampling', train.groupby('rating').count())
        train = upsample_train(train)
        print('after upsampling', train.groupby('rating').count())
    x_train = sklearn_vectorize(train['review'], sklearn_vectorizer)
    x_test = sklearn_vectorize(test['review'], sklearn_vectorizer)
    y_train, y_test = train['rating'], test['rating']
    return x_train, x_test, y_train, y_test


def random_review_slice(reviews):
    import tensorflow as tf
    '''
    a data augmentation method I devised
    randomly cut the starting part of review,
    so long reviews exceeding token limit has a random chance of the later part to be read
    '''
    # random position to cut review string
    pos = tf.random.uniform(shape=[],
                            minval=0,
                            maxval=tf.math.maximum(tf.strings.length(reviews)[-1] - 160, 1),
                            dtype=tf.int32)
    # cut review string based on randomized positiion
    substr = tf.strings.substr(reviews, pos=pos, len=-1)
    # split string into words
    splitted = tf.strings.split(substr, sep=' ')
    # if original review string is long enough, cut off position 0, which is likely partial word
    # if original review string is short, keep everything
    selected = splitted[:, tf.cast(tf.math.greater(tf.strings.length(reviews)[-1] - pos, 160), tf.int32):]

    # rejoin string into format usable by tensorflow vectorize layer
    return tf.strings.reduce_join(selected, separator=' ', axis=-1)


def prepare_lstm_datasets(test_size=0.2, upsample=True, random_slice_string_data=True,
                          vocab_size=256, max_tokens=64, tensorflow_vectorize_layer=None,
                          save_vectorize_layer=False):
    import tensorflow as tf
    from tensorflow.data import Dataset

    train, test, data = prepare_train_test(test_size)

    if tensorflow_vectorize_layer is None:
        tensorflow_vectorize_layer = create_tensorflow_vectorize_layer(
            train['review'].apply(tokenize_text),
            vocab_size=vocab_size, max_tokens=max_tokens)
        if save_vectorize_layer:
            save_tensorflow_vectorize_layer(tensorflow_vectorize_layer, save_vectorize_layer)
    else:
        assert isinstance(tensorflow_vectorize_layer, str)
        tensorflow_vectorize_layer = load_tensorflow_vectorize_layer(tensorflow_vectorize_layer)

    train['review'] = train['review'].apply(tokenize_text)
    test['review'] = test['review'].apply(tokenize_text)

    if upsample:
        print('before upsampling', train.groupby('rating').count())
        train = upsample_train(train)
        print('after upsampling', train.groupby('rating').count())

    if random_slice_string_data:
        train = Dataset.from_tensor_slices((train[['review']], train[['rating']])) \
            .map(lambda x, y: (
            tensorflow_tokenize(random_review_slice(x), tensorflow_vectorize_layer),
            tf.one_hot(y, depth=3, axis=-1)))
    else:
        train = Dataset.from_tensor_slices((train[['review']], train[['rating']])) \
            .map(lambda x, y: (
            tensorflow_tokenize(x, tensorflow_vectorize_layer),
            tf.one_hot(y, depth=3, axis=-1)))

    test = Dataset.from_tensor_slices((test[['review']], test[['rating']])) \
        .map(lambda x, y:
             (tensorflow_tokenize(x, tensorflow_vectorize_layer),
              tf.one_hot(y, depth=3, axis=-1)))

    return train, test
