import datetime
import logging

from model_scripts.preprocess import common_preprocess, batch_preprocess, stream_preprocess
from utils.functional_tools import make_functional
from utils.synthesize_user_data import normal_user_data

logging.root.setLevel(logging.DEBUG)


def preprocess_demo():
    '''
    common and batch preprocess multiply data by 10
    stream preprocess divide data by 10
    '''
    make_functional(normal_user_data()) \
        .to_csv('data/data_step_1.csv',
                logs='step 1 data should be 1x standard normal') \
        .pipe(common_preprocess,
              comment='multiply data by 10') \
        .to_csv('data/data_step_2.csv',
                logs='step 2 data should be 10x standard normal') \
        .freeze(comment='freeze the 10x data') \
        .pipe(batch_preprocess,
              comment='multiply data by 10') \
        .to_csv('data/data_step_3.csv',
                logs='step 3 data should be 100x standard normal') \
        .restore(comment='restore the frozen 10x data') \
        .pipe(stream_preprocess,
              comment='multiply data by 10') \
        .to_csv('data/data_step_4.csv',
                logs='step 4 data should be 1x standard normal')


def datetime_demo():
    def get_preceding_tuesday_delta(weekday):
        '''
        get timedelta object for the preceding Tuesday
        '''
        return datetime.timedelta(days=-((weekday + 6) % 7) - (7 * (weekday == 1)))

    return make_functional(datetime.datetime.today()) \
                .freeze(logs='saved today datetime') \
                .weekday(comment='now it becomes Integer') \
                .pipe(get_preceding_tuesday_delta, comment='now it becomes timedelta') \
                .meta_pipe(lambda self: self.restore().pipe(print, comment='should print datetime') + \
                                        self.pipe(print, comment='should print timedelta'),
                           comment='Subtract timedelta from frozen datetime object') \
                .pipe(print) \
                .return_content(logs='return value as datetime of last Tuesday')


if __name__ == '__main__':
    print('\npreprocess_demo()---------------\n')
    preprocess_demo()

    print('\ndatetime_demo()-----------------\n')
    datetime_demo()
