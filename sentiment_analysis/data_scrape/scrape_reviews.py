import _thread
import os
import threading
import time
from contextlib import contextmanager

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

'''
timeout mechanism was 'borrowed' from Stackoverflow
'''


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def scrape_reviews(venue='restaurants', location='bangkok', n_pages=4, path='raw_data',
                   version_name=None, google_map_path='https://www.google.co.th/maps',
                   review_key='รีวิว'):
    '''
    used to scrape google map reviews of venue type
    at location,
    with maximum pages of results
    '''
    with sync_playwright() as pw:
        # creates an instance of the Chromium browser and launches it
        browser = pw.chromium.launch(headless=False)

        # creates a new browser page (tab) within the browser instance
        page = browser.new_page()

        # go to url with Playwright page element
        page.goto(f'{google_map_path}/search/{venue}+{location}')

        time.sleep(4)

        # scrolling
        for i in range(n_pages):
            # tackle the body element
            html = page.inner_html('body')

            # create beautiful soup element
            soup = BeautifulSoup(html, 'html.parser')

            # select items
            categories = soup.select('.hfpxzc')
            last_category_in_page = categories[-1].get('aria-label')

            # scroll to the last item
            last_category_location = page.locator(
                f"text={last_category_in_page}")
            last_category_location.scroll_into_view_if_needed()

        # get links of all categories after scroll
        links = [item.get('href') for item in soup.select('.hfpxzc')]

        if version_name is None:
            version_name = time.time()
        out = []

        page.close()

        for link in links:
            # go to subject link
            page = browser.new_page()
            try:
                with time_limit(30):
                    page.goto(link)
                    time.sleep(4)

                    locator = page.locator(f"text='{review_key}'")
                    if locator.count() > 0:
                        locator.first.click()
                    else:
                        page.close()
                        continue

                    time.sleep(4)

                    # create new soup
                    html = page.inner_html('body')

                    # create beautiful soup element
                    soup = BeautifulSoup(html, 'html.parser')

                    try:
                        stars = soup.select('.DU9Pgb')
                        stars = [star.find('span')['aria-label'][0] for star in stars]
                        reviews = soup.select('.MyEned')
                        reviews = [review.find('span').text for review in reviews]
                        out.append(pd.DataFrame(list(map(list, zip(reviews, stars)))))
                    except Exception as e:
                        print('Error!!', e)
                        continue
            except TimeoutException as e:
                print("Timed out!")
                continue
            page.close()

    pd.concat(out, axis=0).to_csv(os.path.join(path,
                                               f'review_data_{venue}_{location}_{version_name}.csv'),
                                  index=False)


if __name__ == '__main__':
    scrape_reviews()
