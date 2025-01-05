import textwrap
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.etl.process_tic_tac_toe_results import TicTacToeResultsProcessor
from src.machine_learning.build_model import TicTacToeModelBuilder


with DAG(
    "train_tic_tac_model",
    schedule="@daily",
    start_date=pendulum.datetime(day=1, month=1, year=2025, tz="UTC"),
    description="Train the Tic-tac-toe's machine learning model.",
    doc_md=__doc__,
) as dag:
    process_dataset_task = PythonOperator(
        task_id="process_dataset",
        python_callable=TicTacToeResultsProcessor.process,
    )

    process_dataset_task.doc_md = textwrap.dedent(
        """\
        ### Process dataset task
        Executes the process method from the TicTacToeResultsProcessor class.
        This method loads and cleans the src/data/tic_tac_results.csv and outputs
        processed_tic_tac_results.csv.
        """
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=TicTacToeModelBuilder.build,
    )

    train_model_task.doc_md = textwrap.dedent(
        """\
        ### Train model task
        Executes the build method from the TicTacToeModelBuilder class.
        It trains a machine learning model using the processed_tic_tac_results.csv
        and saves the model in src/machine_learning/tic_tac_model.z
        """
    )

    process_dataset_task >> train_model_task
