## Data Assistance Wizard

### Keywords
*Extracted by GitHub Copilot*

- **Language:** `Python`
- **LLM & NLP:** `LangChain` · `OpenAI API` · `GPT-4` · `GPT-3.5` · `LLM Integration` · `Prompt Engineering` · `Natural Language to SQL` · `Natural Language to Pandas` · `Text-to-Query` · `Token Management`
- **LangChain Components:** `DataFrame Agent` · `SQL Database Chain`
- **Web App:** `Streamlit` · `Chat Interface` · `Session State Management` · `Cloud Deployment`
- **Data & Database:** `Pandas` · `PostgreSQL` · `SQLAlchemy` · `psycopg2` · `Data Analysis`

---

[Preview application on cloud](https://pxp-data-chat.streamlit.app/)

### What it is
 - A user-friendly web data application that allows users to ask questions about the data.

### Dependencies
 - Python 3.9+
 - With or without virtualenv, run ```pip install -r requirements.txt```.
 - If connection to SQL database is required, also run ```pip install -r db_requirements.txt```. (tested with Postgresql)

### Usage Instructions
 - Start application with ```streamlit run main.py```
 - (if using csv data) Upload the input csv. The encoding might need to be manually configured.
 - (if using Postgresql) Set the connection details in the provided text boxes.
   - Or set the default values in constants.py
 - Different LLM models have different capabilities. This is set in constants.py
   - gpt-4 may be the most capable overall
   - gpt-3.5-turbo-16k may be needed to increase token limit
   - Temperature of 0 or close to it is advised so model doesn't make things up.
 - After uploading csv or establishing connection, previews will be available in preview tab.
 - Enter anything in the chat input box at the bottom.
 - Chat history is stored in the app, but not sent to the LLM, this is done considering token limit.
   - Each new prompt is a new LLM conversation.
 - Clear chat history button is at the bottom on the chat tab.


### Implementation Details
 - Streamlit
   - Streamlit's primary design is that it always reruns code from top to bottom with any user interaction.
     - It used to be stateless too, meaning everything had to be reloaded and re-computed each time.
     - Statefulness has since been added to the tool.
       - But one still needs to be careful to design it to not redo expensive computations or API call.
 - Overall Design
   - Although probably unnecessary, the app is designed to not persist any unnecessary object in memory.
     - So LLM and class objects are instantiated to perform its functions, then recycled away, functional programming style.
 - NaturalQuery
   - Though the Streamlit app calls it by creating new recyclable instances for each call, the Class is designed so that persisting Class Instance is a valid usage too.
     - So LLM engine is a Class attribute.
   - 2 methods
     - df_run connects to the inputted pandas.DataFrame object
     - db_run connects to the pre-connected Database connection

### Known Issues
 - If csv is poorly formatted, the app will fail at csv upload.
   - LLM is not yet involved in csv load, which still uses ```pd.read_csv()```
 - LLM may be capable of interpreting column names and data, or may not.
   - Getting accurate answers or any answer at all relies on the capabilities of the LLM.
   - Sometimes rewording the question can get the correct answer if the original one cannot.
 - Any database system other than Postgres was not tested, and even Postgres was tested only on my configurations.