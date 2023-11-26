

def common_preprocess(data):
    '''
    placeholder for feature engineering for shared features
    some scaling change added to facilitate debug
    '''
    return data.map(lambda x: x * 10)

def batch_preprocess(data):
    '''
    placeholder for feature engineering for batch features
    some scaling change added to facilitate debug
    '''
    return data.map(lambda x: x * 10)

def stream_preprocess(data):
    '''
    placeholder for feature engineering for stream features
    some scaling change added to facilitate debug
    '''
    return data.map(lambda x: x / 10)


