# concept_drift_on_streaming_data
Cocnept drift evaluation over time series in streaming data on a industrial use case.. Data is generated using a digital twin generated from a small sample data. Twin is deployed ona flask endpoint and then a data simualtor instantiates twin to generate data from three work stations. Data and its features are stored on real time, an airflow server orchestrates concept drift evaluation on a batch process and results might be seen onreal time on a dash front.