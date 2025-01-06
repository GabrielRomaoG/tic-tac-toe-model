FROM apache/airflow:2.10.4-python3.11

ENV AIRFLOW_HOME=/opt/app/
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false

WORKDIR /opt/app

COPY airflow/dags/ dags
COPY src src
COPY resources resources
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

USER root

RUN chown airflow resources/ 

USER airflow

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

EXPOSE 8080

CMD ["airflow", "standalone"]
