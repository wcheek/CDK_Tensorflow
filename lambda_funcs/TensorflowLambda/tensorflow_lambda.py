# import tempfile
from pathlib import Path
from typing import Tuple

import boto3
import joblib


def get_model() -> Tuple:
    """
    Gets model from EFS if exists. Otherwise load model from S3, save to EFS
    """
    local_path = Path(f"/mnt/models/model.tensorflow")
    try:
        with open(local_path, "rb") as f:
            f.seek(0)
            model = joblib.load(f)
    except FileNotFoundError:
        client = boto3.client("s3")
        # Save model to EFS
        client.download_file(
            "models-bucket",
            "model.tensorflow",
            str(local_path),
        )

        with open(local_path, "rb") as f:
            f.seek(0)
            model = joblib.load(f)

    return model


model = get_model()


def get_prediction(model, input_data):
    # Do what you need to do to feed input data to your model
    return 1
    # return output_data


def handler(event, context):
    # This is the data we get from the client query
    data = event["queryStringParameters"]["q"]
    # I pass the data as a list to the API, but it gets converted into a string.
    # This is some fancy way to get back the list from the str(list)
    split_str = data.split(",")
    formatted_data = (
        [float(split_str[0].split("[")[-1])]
        + [float(y) for y in split_str[1:-1]]
        + [float(split_str[-1].split("]")[0])]
    )
    assert isinstance(formatted_data, list)

    # Get a prediction by feeding your formatted input data into model
    prediction = get_prediction(model=model, input_data=formatted_data)
    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
        },
        "body": f"The predicted value is {prediction}",
    }

    return response
