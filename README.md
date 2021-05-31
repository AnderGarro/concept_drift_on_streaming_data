# Concept Drift on Streaming Data
Concept drift evaluation over time series in streaming data on a industrial use case. 

Data is generated using a digital twin out of a small sample data. sample data comes from an indsutrial use case where a sensor captures a especific magnitude for an interval of 2-3 seconds on a 10 ms sample freq. Then, each time series contains 190-200 points. This process might simulate magnitdues such as pressure on a stamp or temperature on a drill process. Twin is deployed on a flask endpoint. 

* A data simulator instantiates twin to generate data from three work stations. We use a jupyter notebooik to run the simulator. 
* Data and its features are stored on real time on a postgress database. 
* An airflow server orchestrates concept drift evaluation on a batch process
* Results might be seen onreal time on a dash front.

## Getting Started:

The arquitecture is ready to be deployed using docker and docker-compose. Before following next steps, make sure you have both technologies installed and ready to use. The ``docker-compose.yml`` file will build a [Flask](https://flask.org) API service for data simulations with [PostgeSQL](https://www.postgresql.org/) as the data store. Note that Flask will represent the twin of the industrial process. An [AirFlow](http://airflow.apache.org) container will also be deployed which will use [PostgeSQL](https://www.postgresql.org/) as the metadata store too. Moreover, a [Jupyter](https://jupyter.org/) instance will be deployed to run simulations. Finally, a [Dash](https://plotly.com/dash/) server is used for the visualization of the results on real time.

Follow these steps to build the MLflow-AirFlow-AutoML stack:

1. Install docker (docker & docker-compose commands must be installed).
2. git clone
3. docker-compose up -d
4. Open JupyterLab UI at http://your-docker-machine-ip:1997 and run all cells in simulator_run.py for starting the digital twin.
5. Open Dash UI/Fornt at http://your-docker-machine-ip:8050 to see real time visualization.
6. Open Airflow UI at http://your-docker-machine-ip:8080 and turn dag On for cocnept drift evaluation.
7. Enjoy!
