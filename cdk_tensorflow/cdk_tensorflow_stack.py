from aws_cdk import CfnOutput as Output
from aws_cdk import CfnResource, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_efs as efs
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment
from constructs import Construct


class CdkTensorflowStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.vpc = None
        self.access_point = None
        self.prediction_lambda = None
        self.models_bucket = None

        self.build_infrastructure()

    def build_infrastructure(self):
        self.build_vpc()
        self.build_filesystem()
        self.build_lambda()
        self.build_bucket()
        self.build_function_url()

    def build_vpc(self):
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
            timeout=Duration.seconds(60 * 0.5),
            memory_size=128 * 6 * 1,  # mb
            filesystem=_lambda.FileSystem.from_efs_access_point(
                ap=self.access_point, mount_path="/mnt/models"
            )
            if self.access_point
            else None,
            vpc=self.vpc,
        )

    def build_bucket(self):
        self.models_bucket = s3.Bucket(
            scope=self,
            id="ModelsBucket",
            versioned=True,
            bucket_name="models-bucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        aws_s3_deployment.BucketDeployment(
            self,
            "database_for_lex_lambda",
            sources=[aws_s3_deployment.Source.asset("model_files")],
            destination_bucket=self.my_bucket,
        )
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
