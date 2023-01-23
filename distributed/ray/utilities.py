import logging
import joblib
import mlflow
import numpy as np
from mlflow import MlflowClient
import pandas as pd
import os
import re


def mlflow_log_artifact(parent_run_id, artifact, local_path, **kwargs):
    logging.info(f"In log_artifact...run id = {parent_run_id}, local_path")
    mlflow.set_tags({'mlflow.parentRunId': parent_run_id})

    with open(local_path, "wb") as artifact_handle:
        joblib.dump(artifact, artifact_handle)
    MlflowClient().log_artifact(parent_run_id, local_path, **kwargs)
    logging.info("Logging was successful.")


def mlflow_load_artifact(parent_run_id, artifact_name, **kwargs):
    try:
        logging.info(f"In load_artifact...run id = {parent_run_id}, {kwargs}")
        mlflow.set_tags({'mlflow.parentRunId': parent_run_id})

        artifact_list = MlflowClient().list_artifacts(parent_run_id)
        artifact_match = next((artifact for artifact in artifact_list if artifact.path == artifact_name), None)
        artifact = None

        if artifact_match:
            download_path = mlflow.artifacts.download_artifacts(**kwargs)
            logging.info(f"Artifact downloaded to...{download_path}")
            with open(f"{download_path}", "rb") as artifact_handle:
                artifact = joblib.load(artifact_handle)
        else:
            logging.info(f"Artifact {artifact_name} cannot be loaded (has not yet been saved).")

        return artifact
    except Exception as e:
        logging.info(f'Could not complete execution for load_artifact - {kwargs}- error occurred: ', exc_info=True)


def mlflow_log_text(parent_run_id, **kwargs):
    logging.info(f"In log_text...run id = {parent_run_id}, {kwargs}")
    mlflow.set_tags({'mlflow.parentRunId': parent_run_id})

    MlflowClient().log_text(parent_run_id, **kwargs)

    logging.info("Logging was successful.")


def mlflow_log_metric(parent_run_id, **kwargs):
    logging.info(f"In log_metric...run id = {parent_run_id}")
    mlflow.set_tags({'mlflow.parentRunId': parent_run_id})

    MlflowClient().log_metric(parent_run_id, **kwargs)

    logging.info("Logging was successful.")


def mlflow_generate_autolog_metrics(flavor=None):
    getattr(mlflow, flavor).autolog(log_models=False) if flavor is not None else mlflow.autolog(log_models=False)


def text_to_numpy(textfile):
    records = np.genfromtxt(textfile, delimiter=',')
    return records


def get_root_run_id(experiment_names=['Default']):
    runs = mlflow.search_runs(experiment_names=experiment_names, filter_string="tags.runlevel='root'", max_results=1,
                              output_format='list')
    logging.debug(f"Parent run is...{runs}")
    return runs[0].info.run_id if len(runs) else None


def get_next_rolling_window(current_dataset, num_shifts):
    if not len(current_dataset):
        logging.error("Error: Cannot get the next rolling window for an empty dataset")
    else:
        new_dataset = pd.concat(
            [current_dataset[num_shifts % len(current_dataset):], current_dataset[:num_shifts % len(current_dataset)]])
        new_dataset.index = current_dataset.index + (current_dataset.index.freq * num_shifts)
        return new_dataset


def filter_rows_by_head_or_tail(df, head=True, num_rows_in_head=None, num_rows_in_tail=None):
    if (num_rows_in_head is not None) != (num_rows_in_tail is not None):
        raise ValueError(
            f"Exactly one of num_rows_head({num_rows_in_head}) and num_rows_tail({num_rows_in_tail}) must be passed, not both")
    if num_rows_in_head is not None:
        return df[:num_rows_in_head] if head else df[num_rows_in_head:]
    else:
        return df[:-num_rows_in_tail] if head else df[-num_rows_in_tail:]


def get_env_var(name):
    if name in os.environ:
        value = os.environ[name]
        return int(value) if re.match("\d+$", value) else value
    else:
        logging.info('Unknown environment variable requested: {}'.format(name))

