from locust import HttpUser, task, between
import random
import pandas as pd
import environ

environ.Env.read_env()

class SentimentUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def hello_world(self):
        self.client.post(f'/predict',
                         headers={'aikey': environ.Env()('AIKEY')}, json={
                'json_data': pd.DataFrame([[random.choice([
                    'วันนี้เป็นวันดีมีงานรื่นเริง',
                    'ลูกเราเคยเรียนที่นี่ ต้องย้ายออกมาเพราะครูตีลูกเราจนก้นลาย',
                    'เนื้อทำมาสุกพอดี แต่ปรุงรสอ่อนไปหน่อย',
                    'ทรายขาว ฟ้าใส ลมโชยเย็นๆ สบายตัวจริงๆ',
                    'อันนี้อะไรก็ไม่รู้'
                ])]]).to_json(),
                'model_choice': random.choice(['LSTM', 'Linear Regression', 'Naive Bayes'])
            })