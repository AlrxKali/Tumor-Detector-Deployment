import mlflow.sagemaker
from mlflow.deployments import get_deploy_client

experiment_id = '368512154669720787'
run_id = '18dcce44f3a943d38a26712cdd42c0ea'
region = 'us-east-1'
aws_id = '569380489595'
arn = 'arn:aws:iam::569380489595:role/aws-sagemaker-for-deploy-ml-model'
app_name = 'avasure-model-application'
model_uri =  f'mlruns/{experiment_id}/{run_id}/artifacts/vgg16_model'
tag_id = '2.1.1'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

mlflow.sagemaker.push_model_to_sagemaker(
        model_name='vgg16',
        model_uri=model_uri,
        execution_role_arn='arn:aws:iam::569380489595:role/aws-sagemaker-for-deploy-ml-model',
        bucket='mlflow-sagemaker-us-east-1-569380489595',
        image_url=image_url,
        region_name=region,
        #vpc_config=vpc_config,
        flavor=None,
    )