import pendulum
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

#from bluetooth_tracker.aggregator import event_aggregator
from python_operators.soldering_cells_functions import sc_eval_pvalue

local_timezone = pendulum.timezone('Europe/Madrid')
args = {
    'owner': 'Airflow',
    'start_date': datetime(2021, 4, 12, tzinfo=local_timezone),
    "depends_on_past": False,
    'email_on_failure': False,
    'email_on_retry': False,
    'email': ['agarro@ikerlan.es'],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# schedule_interval = every hour from 07:00 to 20:00 and every day from Monday to Friday
with DAG(dag_id='sc_eval_pvalue', default_args=args, schedule_interval='0 * * * 1-5', catchup=False) as dag:
    insert_data_task = PythonOperator(task_id='sc_eval_pvalue',
                                     provide_context=True,
                                     python_callable=sc_eval_pvalue)
