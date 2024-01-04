from concurrent.futures import ThreadPoolExecutor
from itertools import product

from data_scrape.data_parameters import VENUES, LOCATIONS, N_PAGES
from data_scrape.scrape_reviews import scrape_reviews
from parameters import N_THREADS

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=N_THREADS)
    executor.map(lambda venue, location: scrape_reviews(venue, location,
                                                        N_PAGES, version_name='0.0.1'),
                 *zip(*list(product(VENUES, LOCATIONS))))
