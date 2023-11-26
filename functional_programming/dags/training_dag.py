import datetime

from airflow.decorators import task_group, task, dag

from constants import DATA_MODE, TRAINING_SIZE, EVALUATION_SIZE, \
    METRIC_MINIMUM_THRESHOLD, DAG_DEFAULT_ARGS


@task
def prepare_data(date):
    '''
    with Airflow module import inside task function is common and best practice,
    to minimize computing power loss at each DAG refresh
    '''
    from utils.functional_tools import make_functional
    from model_scripts.preprocess import common_preprocess, batch_preprocess, stream_preprocess

    preprocess_methods = (common_preprocess, batch_preprocess, stream_preprocess)

    if DATA_MODE == 'synthesize':
        '''
        (csv is used in experimental stage, to facilitate debug)
        synthesize raw data, save synthesized data as csv
        run common preprocess, then save as csv 
        then run both batch and stream preprocess, then save both as csv
        '''
        from utils.synthesize_user_data import normal_user_data
        for dset, size in zip(['train', 'val'], [TRAINING_SIZE, EVALUATION_SIZE]):
            make_functional(normal_user_data(n_user=size), preprocess_methods) \
                .to_csv(f'data/{dset}_{date}.csv', index=False) \
                .common_preprocess() \
                .to_csv(f'data/{dset}_common_processed_{date}.csv', index=False) \
                .freeze() \
                .batch_preprocess() \
                .to_csv(f'data/{dset}_batch_processed_{date}.csv', index=False) \
                .restore() \
                .stream_preprocess() \
                .to_csv(f'data/{dset}_stream_processed_{date}.csv', index=False)
    else:
        '''
        get data from database, then save processed data somewhere
        this processed data can go to temporary or permanent storage depending on final design
        
        filter to separate data into groups for train/test or A/B testing can be included
        '''
        raise NotImplementedError('Unimplemented data mode')


@task_group
def train_and_evaluate(date, model_type):
    import pickle

    @task
    def train_model(date, model_type):
        from model_scripts.train import train

        if DATA_MODE == 'synthesize':
            import pandas as pd
            with open(f'pkl/{model_type}_model_{date}.pkl', 'wb') as f:
                pickle.dump(train(pd.read_csv(f'data/train_{model_type}_processed_{date}.csv')), f)

        else:
            raise NotImplementedError('Unimplemented data mode')

    @task
    def evaluate_model(date, model_type):
        from model_scripts.evaluation import evaluate
        if DATA_MODE == 'synthesize':
            '''
            this demo mode only compares performance with minimum threshold, not with current best model
            '''
            import pandas as pd
            import shutil
            with open(f'pkl/{model_type}_model_{date}.pkl', 'rb') as f:
                metric = evaluate(pickle.load(f))
            if metric > METRIC_MINIMUM_THRESHOLD:
                shutil.copy(f'pkl/{model_type}_model_{date}.pkl',
                            f'pkl/{model_type}_model_latest.pkl')

        else:
            raise NotImplementedError('Unimplemented data mode')

    train_model(date, model_type) >> \
    evaluate_model(date, model_type)


@dag(dag_id=f'weekly_training_run',
     start_date=datetime.datetime(2023, 11, 7), schedule="0 17 * * 2",
     default_args={'owner': 'Paul', **DAG_DEFAULT_ARGS},
     tags=['weekly_training_run'])
def train_dag():
    prepare_data('{{ ds }}') >> \
    train_and_evaluate \
        .partial(date='{{ ds }}') \
        .expand(model_type=['batch', 'stream'])


train_dag()
