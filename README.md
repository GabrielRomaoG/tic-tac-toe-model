# Teste vaga Cientista de dados Pleno - L5 Networks


## Dataset 

[Resultados Jogo da Velha - Kaggle](https://www.kaggle.com/datasets/fabdelja/tictactoe)

You can see the EDA of the dataset on src/machine_learning/eda.ipynb.

## Project Structure

The dag is available on airflow/dags.

The scripts for etl is on src/elt.

The machine learning code is on src/machine_learning.

## Setup

### Manually

For a faster setup, read the Docker setup section.

Create a virtual environment using the python version given by the .python-version file.

```
python -m venv .venv
```

Install the dependencies in the requirements.txt and setup.py:
```
pip install -r requirements.txt
pip intall -e .
```

Additional command for the airflow home path:
```
export AIRFLOW_HOME=$(pwd)/airflow/
```

Now you can run airflow in a local environment:
```
airflow standalone
```

### Docker 

execute these commands to build the container and run it:
```
docker build -t tic_tac_toe_model:v1 .
docker run -it -p 8080:8080 tic_tac_toe_model:v1
```

### Accessing the web server

After executing the run command, the airflow webserver will be available on the 
port 8080 and the login access is going to be on the terminal.

You can execute the dag train_tic_tac_model manually in there.
