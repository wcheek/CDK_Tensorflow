FROM amazon/aws-lambda-python:latest

LABEL maintainer="Wesley Cheek"
RUN yum update -y && \
    yum install -y python3 python3-dev python3-pip gcc && \
    rm -Rf /var/cache/yum
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY tensorflow_lambda.py ./

CMD ["tensorflow_lambda.handler"]