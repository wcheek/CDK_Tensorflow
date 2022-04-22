# Put a Tensorflow model into production with AWS Lambda and AWS CDK

To allow us to deploy a Tensorflow model on Lambda, I will pull concepts together from my previous articles.

-   To build a Lambda function large enough to hold the `Tensorflow` library, we will need to [deploy our Lambda function using a Docker container stored on ECR](https://dev.to/wesleycheek/deploy-a-docker-built-lambda-function-with-aws-cdk-fio). 
-   To improve prediction times, we can [store our models in a filesystem attached to our Lambda function](https://dev.to/wesleycheek/lambda-function-with-persistent-file-store-using-aws-cdk-and-aws-efs-45h8). This mostly avoids having to load the models from `S3`.
-   To get prediction results, we will use [Lambda function URLs](https://dev.to/wesleycheek/aws-lambda-function-urls-with-aws-cdk-58ih) to expose an HTTPS endpoint we can query using HTTP GET ([another option is to use an API Gateway](https://dev.to/wesleycheek/deploy-an-api-fronted-lambda-function-using-aws-cdk-2nch)).
-   [To save and load our models](https://dev.to/wesleycheek/saveload-tensorflow-sklearn-pipelines-from-local-and-aws-s3-34dc), we will use Joblib.

## If starting from scratch (not cloning this project)

1) `mkdir project && cd project`
2) `cdk init --language python`
3) Follow instructions below to activate venv, install libraries.
4) Make sure you have activated your AWS credentials and `cdk deploy`

# Welcome to your CDK Python project!

This is a blank project for Python development with CDK.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To manually create a virtualenv on MacOS and Linux:

```
$ python -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
