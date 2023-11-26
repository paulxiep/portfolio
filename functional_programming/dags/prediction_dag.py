import datetime

from airflow.decorators import task, dag
from airflow.sensors.external_task import ExternalTaskSensor

from constants import DATA_MODE, DAG_DEFAULT_ARGS


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
        then run batch preprocess and save as csv
        '''
        from utils.synthesize_user_data import normal_user_data
        make_functional(normal_user_data(n_user=20),
                        additional_methods=preprocess_methods) \
            .to_csv(f'data/predict_{date}.csv', index=False) \
            .common_preprocess() \
            .to_csv(f'data/predict_common_processed_{date}.csv', index=False) \
            .batch_preprocess() \
            .to_csv(f'data/predict_batch_processed_{date}.csv', index=False)
    else:
        '''
        get data from HDFS, then save intermediate processed data somewhere
        
        The black-box API for A/B testing can be called here to split into groups,
        so batch prediction results for each group can use different models
        '''
        raise NotImplementedError('Unimplemented data mode')


@task
def predict(date):
    from model_scripts.prediction import predict
    from model_scripts.train import train
    '''
    Naturally, running daily for batch predictions only
    '''
    if DATA_MODE == 'synthesize':
        import pandas as pd
        predict(train(None),
                pd.read_csv(f'data/predict_batch_processed_{date}.csv')) \
            .to_csv(f'results/batch_prediction_{date}.csv', index=False)

    else:
        '''
        Can be written to run different models for different data groups in A/B testing
        '''
        raise NotImplementedError('Unimplemented data mode')


def get_preceding_tuesday(date):
    '''
    datetime weekday runs from 0 as Monday
    cronjob schedule runs from 0 as Sunday
    training DAG was set to run on Tuesday, so we need Tuesday.
    '''
    from utils.functional_tools import make_functional

    def get_preceding_tuesday_delta(weekday):
        '''
        -7 added because Airflow date argument
        is the last scheduled date preceding the date the run actually triggers
        '''
        return -((weekday + 6) % 7) - 7

    def get_timedelta(delta):
        return datetime.timedelta(days=delta, hours=-3)

    def plus_frozen(obj):
        return obj._content + obj._frozen_content

    return make_functional(date,
                            additional_methods=[get_preceding_tuesday_delta, get_timedelta],
                            meta_methods=[plus_frozen])\
                .freeze()\
                .weekday()\
                .get_preceding_tuesday_delta()\
                .get_timedelta()\
                    .restore()\
                    .plus_frozen()\
                    .return_content()


@dag(dag_id=f'daily_batch_prediction_run',
     start_date=datetime.datetime(2023, 11, 14), schedule="0 20 * * *",
     default_args={'owner': 'Paul', **DAG_DEFAULT_ARGS},
     max_active_runs=2,
     tags=['daily_batch_prediction_run'])
def prediction_dag():
    ExternalTaskSensor(
        task_id='wait_for_train',
        external_task_group_id='train_and_evaluate',
        external_dag_id='weekly_training_run',
        allowed_states=['success'],
        #failed_states=['failed'],
        execution_date_fn=get_preceding_tuesday,
        poke_interval=300
    ) >> \
        prepare_data('{{ ds }}') >> \
            predict('{{ ds }}')


prediction_dag()
