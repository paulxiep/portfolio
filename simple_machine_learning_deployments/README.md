# machine-learning-operations-test
Originally a test for mid level Machine learning Engineer for a company in Thailand.

Since no data and model were actually provided, I synthesized them and adapted the test into my portfolio.

```.env``` file has been adjusted accordingly and for the purpose of this portfolio, will be included in git.

### Instruction

A _junior_ data scientist has prepared a prediction model ready to predict client's orders for a restaurant - given that a scientist has no prior knowledge on machine learning operations and bringing the models up for production, you, as a machine learning engineer, has to bring this model to live by yourself. Please *demonstrate* how would you do it. 

Fork this repo and make this model ready to deploy on GCP on any suitable service of your choice

### Important

Will require ```.env``` file with corresponding keys to run

### Methods

1. Streamlit: Simple, easy to use, web application tool suitable for small data application.
2. Airflow Docker: While normally associated with data pipeline, Airflow can be adapted to schedule regular model predict/retraining service.
3. Flask API: I once used the company's own application in place of Flask to enable API on cloud. I'm a newbie to Flask itself and may not be familiar with security protocol. Nevertheless I made a working Flask API application, and can be tested with your own API call program or my streamlit program.

### How to run

0. All methods: ```sh 00_prepare_demo.sh``` to prepare dummy models and synthetic data for demo runs on all methods.
1. Streamlit: ```streamlit run simple_streamlit.py```, then insert test data with categorical features already encoded (can be obtained from ```/data``` after preparing demo in step 0). The results will be displayed on the streamlit web app.
2. Airflow Docker: (requires running docker engine) ```sh 02_prepare_airflow_demo.sh```. Go to ```localhost:{AIRFLOW_PORT}``` on your browser (requires said port to be vacant), and login with user and password specified in .env. Click to enable the daily_prediction_run DAG and wait for scheduled run or activate more runs manually. The prediction results and generated test data will be in ```/results``` folder. When done run ```sh 10_clean_docker.sh```.
3. Flask API: (requires running docker engine) ```sh 03_prepare_flask_demo.sh```. Then run ```streamlit run simple_streamlit.py```, except this time choose 'Call Flask API' on the radio button, then uploaded test data. When done run ```sh 10_clean_docker.sh```.

### Mid-Development Updates Log

2023-11-17 21:53: Added data synthesizer to somewhat mimic the data in jupyter notebook.

2023-11-17 22:18: Added module to mimic the preprocessing in the jupyter notebook.

2023-11-17 22.40: Added 2nd part of preprocess for feature set 2.

2023-11-18 01:06: Prepared dummy model modules and model saving functionalities (Stacking doesn't work yet)

2023-11-18 13:35: Method 1 (Streamlit) up (still without Stacking model until further notice)

2023-11-18 16:20: Method 2 (Airflow Docker) up (with only prediction DAG).

2023-11-19 11:15: Restructured project in preparation for Method 3.

2023-11-19 16:10: Method 3 (Flask Web API) up and simplified running steps.
