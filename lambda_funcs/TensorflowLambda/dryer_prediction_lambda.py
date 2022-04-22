# import tempfile
from logging import getLogger
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from modeling_features import (
    distribution_x_feats,
    modeling_features_together,
    moisture_dist_columns,
    remaining_time_x_feats,
)

# Use this to retrain model if I want
# from dryerprocessing.process_iot import ProcessIoT
# ProcessIoT(local=False, save=True, train_models=True)

logger = getLogger()
logger.setLevel(level="DEBUG")


def get_models(reload_models: bool = False) -> Tuple:
    """
    Gets the remaining drying time and the distribution prediction algorithms

    Returns:
        Tuple: (remaining drying time model, distribution prediction model)
    """
    models_ = []
    for model_name in [
        "predict_drying_time.sklearn",
        "predict_distribution.tensorflow",
    ]:
        local_path = Path(f"/mnt/models/{model_name}")
        try:
            if reload_models:
                raise FileNotFoundError
            logger.debug(f"Trying to load {model_name}")
            with open(local_path, "rb") as f:
                f.seek(0)
                models_.append(joblib.load(f))
                logger.debug(f"Successfully loaded {model_name} from EFS")
        except FileNotFoundError:
            logger.debug(f"{model_name} file not found on EFS. Load from s3")

            import boto3

            client = boto3.client("s3")
            s3_uri = f"s3://dryer-data/Meta/Models/new_models/{model_name}"
            client.download_file(
                "dryer-data",
                ("/").join(Path(s3_uri).parts[2:]),
                str(local_path),
            )

            logger.debug(
                f"Downloaded {model_name} from S3, saved to EFS: {str(local_path)}"
            )
            with open(local_path, "rb") as f:
                f.seek(0)
                models_.append(joblib.load(f))
                logger.debug(
                    f"Successfully loaded {model_name} from EFS after loading from s3"
                )

    return models_


drying_time_model, distribution_model = get_models(reload_models=False)


class Modeling:
    def __init__(self, data: pd.Series, drying_time_model, distribution_model):
        self.data = data

        (
            self.remaining_drying_time,
            self.predicted_distribution,
        ) = self.get_prediction(
            data=self.data,
            drying_time_model=drying_time_model,
            distribution_model=distribution_model,
        )

    def get_prediction(
        self, data: pd.Series, drying_time_model, distribution_model
    ) -> Tuple[float, pd.Series]:
        """
        Predict the remaining drying time and distribution from a data series

        Args:
            data (pd.Series): Most recent drying data

        Returns:
            float: the predicted remaining drying time
            pd.Series: the predicted drying distribution after predicted remaining drying time
        """
        remaining_drying_time = drying_time_model.predict(
            data.loc[remaining_time_x_feats]
            .dropna()
            .to_numpy()
            .reshape(-1, len(remaining_time_x_feats))
        )
        data["elapsed_time"] = float(
            data["elapsed_time"] + remaining_drying_time
        )
        # This is the distribution predicted in the future
        predicted_distribution = pd.Series(
            distribution_model.predict(
                data.loc[distribution_x_feats]
                .dropna()
                .to_numpy()
                .reshape(-1, len(distribution_x_feats))
            ).reshape(-1),
            index=[f"{col}_model" for col in moisture_dist_columns],
        )

        return remaining_drying_time, predicted_distribution


def handler(event, context):
    data = event["queryStringParameters"]["q"]
    split_str = data.split(",")
    formatted_data = (
        [float(split_str[0].split("[")[-1])]
        + [float(y) for y in split_str[1:-1]]
        + [float(split_str[-1].split("]")[0])]
    )
    assert isinstance(formatted_data, list)
    input_data = pd.Series(
        data=formatted_data, index=modeling_features_together
    )

    model_obj = Modeling(
        data=input_data,
        drying_time_model=drying_time_model,
        distribution_model=distribution_model,
    )
    remaining_drying_time = model_obj.remaining_drying_time
    predicted_distribution = model_obj.predicted_distribution
    logger.debug(msg=f"Initial event: {event}")
    nl = "\n"
    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
        },
        "body": (
            f"The predicted remaining drying time is "
            f"{remaining_drying_time[0]} hrs {nl}{nl}"
            f"The predicted distribution after this time is {nl}{predicted_distribution}"
        ),
    }

    return response
