import os
import sys
from collections import defaultdict
import logging
import traceback
import joblib
import tempfile
import mlflow
import json
import pandas as pd
from os.path import exists
from datetime import datetime, timedelta, timezone
import pytz
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset
from mlmetrics import exporter
import numpy as np
import re
from mlflow import MlflowClient
from prodict import Prodict
from multiprocessing import Process, Lock
from filelock import FileLock, Timeout
from mlflow.models import MetricThreshold


def mlflow_log_artifact(parent_run_id, artifact, local_path, **kwargs):
    logging.info(f"In log_artifact...run id = {parent_run_id}, local_path")
    mlflow.set_tags({'mlflow.parentRunId': parent_run_id})

    with open(local_path, "wb") as artifact_handle:
        joblib.dump(artifact, artifact_handle)
    # synchronize_file_write(file=artifact, file_path=local_path)
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


def mlflow_generate_autolog_metrics(flavor=None):
    getattr(mlflow, flavor).autolog(log_models=False) if flavor is not None else mlflow.autolog(log_models=False)

