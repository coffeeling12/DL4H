#!/usr/bin/env bash

# REMEMBER TO SETUP AWS_PROFILE environment variable before running this script!
# For example:
# export AWS_PROFILE=vts-dev

# Based on https://github.com/aws-samples/amazon-sagemaker-custom-container/blob/master/build_and_push.sh
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

image="dlh/preprocess"

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


region=$(aws configure get region)
region=${region:-us-east-1}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Use https://github.com/awslabs/amazon-ecr-credential-helper
# instead of
# $(aws ecr get-login --region ${region} --no-include-email)

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}