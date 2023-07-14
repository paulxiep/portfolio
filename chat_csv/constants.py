# Model parameters
MODEL_NAME = 'gpt-4'
TEMPERATURE = 0
VERBOSE = False

# try latin1 for csv with European characters
DEFAULT_ENCODING = 'utf-8'

# DB connection parameters
DB_USER = 'db_user_here'
DB_PASSWORD = 'db_password_here'
DB_NAME = 'db_name_here'
DB_ADDRESS = 'db_address_here'
DB_PORT = 'db_port_here'
DB_TABLE_NAME = 'db_table_name_here'

# used for db_run, table name will be added to prompt to tell LLM to look at specific table
APPEND_TABLE_NAME_TO_PROMPT = True