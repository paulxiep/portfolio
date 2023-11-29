import datetime
import logging
import random

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
                log='step 1 data should be 1x standard normal') \
        .pipe(common_preprocess,
              comment='multiply data by 10') \
        .to_csv('data/data_step_2.csv',
                log='step 2 data should be 10x standard normal') \
        .freeze(comment='freeze the 10x data') \
        .pipe(batch_preprocess,
              comment='multiply data by 10') \
        .to_csv('data/data_step_3.csv',
                log='step 3 data should be 100x standard normal') \
        .restore(comment='restore the frozen 10x data') \
        .pipe(stream_preprocess,
              comment='divide data by 10') \
        .to_csv('data/data_step_4.csv',
                log='step 4 data should be 1x standard normal')


def datetime_demo():
    def get_preceding_tuesday_delta(weekday):
        '''
        get timedelta object for the preceding Tuesday
        '''
        return datetime.timedelta(days=-((weekday + 6) % 7) - (7 * (weekday == 1)))

    return make_functional(datetime.datetime.today()) \
        .freeze(log='saved today datetime') \
        .weekday(comment='calling datetime method, now it becomes Integer') \
        .pipe(get_preceding_tuesday_delta, comment='now it becomes timedelta') \
        .meta_pipe(lambda self: self.restore().pipe(logging.info, comment='should log datetime') + \
                                self.pipe(logging.info, comment='should log timedelta'),
                   comment='Subtract timedelta from frozen datetime object', content_log_level=logging.INFO) \
        .return_content(log='return value as datetime of last Tuesday')


def list_demo():
    return make_functional([1, 2, 3, 4, 5, 6, 7, 8]) \
        .pipe(random.shuffle, content_log_level=logging.INFO, comment='shuffle') \
        .pipe(random.shuffle, content_log_level=logging.INFO, comment='shuffle') \
        .sort(content_log_level=logging.INFO, log='calling list method sort') \
        .sort(reverse=True, content_log_level=logging.INFO, comment='calling list method') \
        .reverse(content_log_level=logging.INFO, log='calling list method reverse') \
        .pipe(lambda x: x[0], content_log_level=logging.INFO, comment='getting 1st element') \
        .pipe(lambda x: [x] * x, content_log_level=logging.INFO) \
        .return_content()


if __name__ == '__main__':
    logging.info('\n\npreprocess_demo()---------------\n')
    preprocess_demo()

    logging.info('\n\ndatetime_demo()-----------------\n')
    datetime_demo()

    logging.info('\n\nlist_demo()-----------------\n')
    list_demo()
