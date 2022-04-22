import aws_cdk as cdk

from cdk_tensorflow.cdk_tensorflow_stack import CdkTensorflowStack

app = cdk.App()
CdkTensorflowStack(
    app,
    "CdkTensorflowStack",
)

app.synth()
