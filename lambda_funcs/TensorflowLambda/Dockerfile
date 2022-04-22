FROM amazon/aws-lambda-python:latest

LABEL maintainer="Wesley Cheek"
RUN yum update -y && \
    yum install -y python3 python3-dev python3-pip gcc && \
    rm -Rf /var/cache/yum
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY dryer_prediction_lambda.py ./
COPY modeling_features.py ./
COPY processing ./
RUN pip install .
RUN pip freeze | grep tensorboard | xargs pip uninstall -y
RUN pip freeze | grep google | xargs pip uninstall -y
RUN pip cache purge
CMD ["dryer_prediction_lambda.handler"]