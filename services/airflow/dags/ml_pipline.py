from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add code directory to path
sys.path.insert(0, '/opt/airflow/code')

from datasets.process_data import load_and_process_data
from models.train_model import train_and_evaluate

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    catchup=False,
) as dag:

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=load_and_process_data,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_and_evaluate,
    )

    deploy_services_task = BashOperator(
        task_id='deploy_services',
        bash_command='cd /opt/airflow/code/deployment && docker-compose up -d --build',
    )

    process_data_task >> train_model_task >> deploy_services_task