from airflow.utils.dates import days_ago
from airflow import DAG

from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from python_functions import (
    data_processing_fn,
    download_dataset_fn,
    ml_training_RandomForest_fn,
)

args = {"owner": "airflow", "retries": 1, "start_date": days_ago(1)}

with DAG(
    dag_id="airflow_ml_pipeline",
    default_args=args,
    description="ML pipeline simple",
    schedule=None,
) as dag:
    dummy_task = EmptyOperator(task_id="Starting the process", retries=2)

    task_extract_data = PythonOperator(
        task_id="download_dataset", python_callable=download_dataset_fn
    )

    task_process_data = PythonOperator(
        task_id="data_processing", python_callable=data_processing_fn
    )

    task_train_RF_model = PythonOperator(
        task_id="ml_training_RandomForest", python_callable=ml_training_RandomForest_fn
    )

    task_train_logistic_model = PythonOperator(
        task_id="ml_training_Logistic", python_callable=ml_training_Logistic_fn
    )

    task_identify_best_model = PythonOperator(task_id="identify_best_model",python_callable=identify_best_model_fn)

dummy_task >> task_extract_data >> task_process_data >> [task_train_RF_model,task_train_logistic_model] >> task_identify_best_model
