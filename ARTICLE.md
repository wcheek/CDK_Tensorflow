# Put a Tensorflow model into production with AWS Lambda and AWS CDK

To allow us to deploy a Tensorflow model on Lambda, I will pull concepts together from my previous articles. This article is about deploying a model on Lambda, so I will not be talking about training the model or using it.

-   To build a Lambda function large enough to hold the `Tensorflow` library, we will need to [deploy our Lambda function using a Docker container stored on ECR](https://dev.to/wesleycheek/deploy-a-docker-built-lambda-function-with-aws-cdk-fio). 
-   To improve prediction times, we can [store our models in a filesystem attached to our Lambda function](https://dev.to/wesleycheek/lambda-function-with-persistent-file-store-using-aws-cdk-and-aws-efs-45h8). This mostly avoids having to load the models from `S3`.
-   To get prediction results, we will use [Lambda function URLs](https://dev.to/wesleycheek/aws-lambda-function-urls-with-aws-cdk-58ih) to expose an HTTPS endpoint we can query using HTTP GET ([another option is to use an API Gateway](https://dev.to/wesleycheek/deploy-an-api-fronted-lambda-function-using-aws-cdk-2nch)).
-   [To save and load our models](https://dev.to/wesleycheek/saveload-tensorflow-sklearn-pipelines-from-local-and-aws-s3-34dc), we will use Joblib.

The Github repository can be found [here](https://github.com/wcheek/CDK_Tensorflow).

Let’s get started building our stack and function!

## Stack Design

I’ve tried to make this design as minimal as possible while keeping the features we are looking for: a docker deployed Lambda function with `Tensorflow` installed, including an attached file system, and with an HTTPS endpoint which can be queried with an HTTP GET message.

```python
from aws_cdk import CfnOutput as Output
from aws_cdk import CfnResource, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_efs as efs
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from constructs import Construct


class CdkTensorflowStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Let's list all of our physical resources getting deployed
        self.vpc = None
        self.access_point = None
        self.prediction_lambda = None
        self.models_bucket = None

        # convenient deployment
        self.build_infrastructure()

    def build_infrastructure(self):
        self.build_vpc()
        self.build_filesystem()
        self.build_lambda()
        self.build_bucket()
        self.build_function_url()

    def build_vpc(self):
        # Need the VPC for the lambda filesystem
        self.vpc = ec2.Vpc(scope=self, id="VPC", vpc_name="ExampleVPC")

    def build_filesystem(self):
        file_system = efs.FileSystem(
            scope=self,
            id="ExampleEFS",
            vpc=self.vpc,
            file_system_name="ExampleEFS",
            removal_policy=RemovalPolicy.DESTROY,
        )
        # create a new access point from the filesystem
        self.access_point = file_system.add_access_point(
            "AccessPoint",
            # set /export/lambda as the root of the access point
            path="/export/lambda",
            # as /export/lambda does not exist in a new efs filesystem, the efs will create the directory with the following createAcl
            create_acl=efs.Acl(
                owner_uid="1001", owner_gid="1001", permissions="750"
            ),
            # enforce the POSIX identity so lambda function will access with this identity
            posix_user=efs.PosixUser(uid="1001", gid="1001"),
        )

    def build_lambda(self):
        self.prediction_lambda = _lambda.DockerImageFunction(
            scope=self,
            id="TensorflowLambda",
            function_name="TensorflowLambda",
            code=_lambda.DockerImageCode.from_image_asset(
                directory="lambda_funcs/TensorflowLambda"
            ),
            # I've found inferences can be made with my simple model in < 20 sec
            timeout=Duration.seconds(60 * 0.5),
            memory_size=128 * 6 * 1,  # mb
            # Attach the EFS file system
            filesystem=_lambda.FileSystem.from_efs_access_point(
                ap=self.access_point, mount_path="/mnt/models"
            )
            if self.access_point
            else None,
            # Needs to be placed in the same VPC as the EFS file system
            vpc=self.vpc,
        )

    def build_bucket(self):
        self.models_bucket = s3.Bucket(
            scope=self,
            id="ExampleModelsBucket",
            bucket_name="models-bucket",
            # These settings will make sure things get deleted when we take down the stack
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        # We can add files to our new bucket from a local source
        s3_deployment.BucketDeployment(
            self,
            "save_model_to_s3",
            sources=[s3_deployment.Source.asset("model_files")],
            destination_bucket=self.models_bucket,
        )
        # Make sure to give the lambda permission to retrieve the model file
        self.models_bucket.grant_read(identity=self.prediction_lambda)

    def build_function_url(self):
        # Set up the Lambda Function URL
        cfnFuncUrl = CfnResource(
            scope=self,
            id="lambdaFuncUrl",
            type="AWS::Lambda::Url",
            properties={
                "TargetFunctionArn": self.prediction_lambda.function_arn,
                "AuthType": "NONE",
                "Cors": {"AllowOrigins": ["*"]},
            },
        )

        # Give everyone permission to invoke the Function URL
        CfnResource(
            scope=self,
            id="funcURLPermission",
            type="AWS::Lambda::Permission",
            properties={
                "FunctionName": self.prediction_lambda.function_name,
                "Principal": "*",
                "Action": "lambda:InvokeFunctionUrl",
                "FunctionUrlAuthType": "NONE",
            },
        )

        # Get the Function URL as output
        Output(
            scope=self,
            id="funcURLOutput",
            value=cfnFuncUrl.get_att(attribute_name="FunctionUrl").to_string(),
        )

```

## Lambda Function Design

### cdk_tensorflow/lambda_funcs/TensorflowLambda/tensorflow_lambda.py

Again, trying to show a minimum working example. This Lambda function will try to load the model from the EFS. If it can’t be found (like on first run), it copies the model from the S3 bucket we created, saves it to EFS, and loads it again. Once we have the model loaded we can use it to make inferences. The handler will send the inference back to the client which queried it.

```python
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

```

### cdk_tensorflow/lambda_funcs/TensorflowLambda/Dockerfile

Provides build instructions to build the Lambda function.

```dockerfile
FROM amazon/aws-lambda-python:latest

LABEL maintainer="Wesley Cheek"
RUN yum update -y && \
    yum install -y python3 python3-dev python3-pip gcc && \
    rm -Rf /var/cache/yum
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY tensorflow_lambda.py ./

CMD ["tensorflow_lambda.handler"]
```

### cdk_tensorflow/lambda_funcs/TensorflowLambda/requirements.txt

```
joblib
# I use tensorflow-cpu because it's half the size of the gpu version and Lambda doesn't have a GPU anyway.
tensorflow-cpu
boto3
```

## Deployment

We can run `cdk deploy` and our infrastructure will get autmatically deployed. Any files in the folder `model_files` will get uploaded to the s3 bucket we created. The Lambda function will be bundled using Docker and saved to `ECR`.

## Testing & Querying the Lambda function

Testing will depend on your model. I hope I have given you enough of an outline to get started. To send data to your newly deployed lambda function, you can find the Function URL either in the outputs after CDK has finished deploying or on the AWS Lambda console.

![image-20220422101510034](D:\Projects\Notes\My Articles\5_CDK_Tensorflow\Assets\image-20220422101510034.png)

The data shown below is "hello!" - for you it could be anything. My models would receive a list of values, for instance.

![image-20220422101530720](D:\Projects\Notes\My Articles\5_CDK_Tensorflow\Assets\image-20220422101530720.png)

Once you’re finished make sure to `cdk destroy` to avoid any charges!
