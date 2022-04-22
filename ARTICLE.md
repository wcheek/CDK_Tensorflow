# Put a Tensorflow model into production with AWS Lambda and AWS CDK

To allow us to deploy a Tensorflow model on Lambda, I will pull concepts together from my previous articles.

-   To build a Lambda function large enough to hold the `Tensorflow` library, we will need to [deploy our Lambda function using a Docker container stored on ECR](https://dev.to/wesleycheek/deploy-a-docker-built-lambda-function-with-aws-cdk-fio). 
-   To improve prediction times, we can [store our models in a filesystem attached to our Lambda function](https://dev.to/wesleycheek/lambda-function-with-persistent-file-store-using-aws-cdk-and-aws-efs-45h8). This mostly avoids having to load the models from `S3`.
-   To get prediction results, we will use [Lambda function URLs](https://dev.to/wesleycheek/aws-lambda-function-urls-with-aws-cdk-58ih) to expose an HTTPS endpoint we can query using HTTP GET ([another option is to use an API Gateway](https://dev.to/wesleycheek/deploy-an-api-fronted-lambda-function-using-aws-cdk-2nch)).
-   [To save and load our models](https://dev.to/wesleycheek/saveload-tensorflow-sklearn-pipelines-from-local-and-aws-s3-34dc), we will use Joblib.

The Github repository can be found [here](https://github.com/wcheek/CDK_Tensorflow).

Let’s get started building our stack!

