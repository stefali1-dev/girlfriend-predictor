"""Everything the web page needs on AWS, in one stack in eu-west-1.

    BUDGET_EMAIL=you@example.com make deploy    # build + push the image, create or update all
    BUDGET_EMAIL=you@example.com make destroy   # remove it all (the budget too)

Needs models/partnered.joblib and models/typical.json (make all), Docker, and once per
account and region: make bootstrap.
"""

import os
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sagemaker as sagemaker

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "girlfriend-predictor"
VERCEL_TEAM = "stefali1-devs-projects"
VERCEL_PROJECT = "girlfriend-predictor"


class WebModel(cdk.Stack):
    def __init__(self, scope, name, **kwargs):
        super().__init__(scope, name, **kwargs)

        # SageMaker only takes the classic Docker manifest, not OCI (Docker's default since v29):
        # hence oci-mediatypes=false here and BUILDX_NO_DEFAULT_ATTESTATIONS=1 in `make deploy`.
        image = ecr_assets.DockerImageAsset(self, "Image", directory=str(ROOT), file="sagemaker/Dockerfile",
                                            platform=ecr_assets.Platform.LINUX_AMD64,
                                            outputs=["type=docker,oci-mediatypes=false"])
        # SageMaker writes the container's output here. Made first, so it gets a short retention.
        log_group = logs.LogGroup(self, "Logs", log_group_name=f"/aws/sagemaker/Endpoints/{ENDPOINT}",
                                  retention=logs.RetentionDays.ONE_WEEK,
                                  removal_policy=cdk.RemovalPolicy.DESTROY)

        role = iam.Role(self, "SageMakerRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        image.repository.grant_pull(role)
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
            resources=[log_group.log_group_arn]))

        model = sagemaker.CfnModel(self, "Model", execution_role_arn=role.role_arn,
                                   primary_container={"image": image.image_uri})
        model.node.add_dependency(role)  # SageMaker checks it can pull the image when the model is made
        config = sagemaker.CfnEndpointConfig(self, "Config", production_variants=[{
            "modelName": model.attr_model_name,
            "variantName": "AllTraffic",
            # Scales to zero when idle. 3 at once is plenty and caps what abuse could cost.
            "serverlessConfig": {"maxConcurrency": 3, "memorySizeInMb": 2048},
        }])
        endpoint = sagemaker.CfnEndpoint(self, "Endpoint", endpoint_name=ENDPOINT,
                                         endpoint_config_name=config.attr_endpoint_config_name)
        endpoint.node.add_dependency(log_group)

        # The Vercel function signs in through Vercel's OIDC: no AWS keys stored anywhere.
        issuer = f"oidc.vercel.com/{VERCEL_TEAM}"
        vercel = iam.OidcProviderNative(self, "Vercel", url=f"https://{issuer}",
                                        client_ids=[f"https://vercel.com/{VERCEL_TEAM}"])
        vercel_role = iam.Role(self, "VercelRole", assumed_by=iam.WebIdentityPrincipal(
            vercel.oidc_provider_arn, conditions={
                "StringEquals": {f"{issuer}:aud": f"https://vercel.com/{VERCEL_TEAM}"},
                "StringLike": {f"{issuer}:sub": [
                    f"owner:{VERCEL_TEAM}:project:{VERCEL_PROJECT}:environment:preview",
                    f"owner:{VERCEL_TEAM}:project:{VERCEL_PROJECT}:environment:production"]},
            }))
        vercel_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"],
            resources=[self.format_arn(service="sagemaker", resource="endpoint", resource_name=ENDPOINT)]))

        # For the whole account, not only this stack.
        budgets.CfnBudget(self, "Budget", budget={
            "budgetName": "monthly-10-usd", "budgetType": "COST", "timeUnit": "MONTHLY",
            "budgetLimit": {"amount": 10, "unit": "USD"},
        }, notifications_with_subscribers=[{
            "notification": {"notificationType": kind, "comparisonOperator": "GREATER_THAN",
                             "threshold": percent, "thresholdType": "PERCENTAGE"},
            "subscribers": [{"subscriptionType": "EMAIL", "address": os.environ["BUDGET_EMAIL"]}],
        } for kind, percent in [("ACTUAL", 80), ("FORECASTED", 100)]])

        cdk.CfnOutput(self, "EndpointName", value=endpoint.endpoint_name)
        cdk.CfnOutput(self, "VercelRoleArn", value=vercel_role.role_arn)


app = cdk.App()
WebModel(app, "GirlfriendPredictor", env=cdk.Environment(region="eu-west-1"))
app.synth()
