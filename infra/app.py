"""CDK entry point: `cdk synth` writes cdk.out; nothing here deploys or calls AWS."""

import os
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import Validations
from cdk_nag import AwsSolutionsChecks

from kyb_infra.stack import KybStack

REPO_ROOT = Path(__file__).resolve().parents[1]  # the Dockerfile and .dockerignore live here

app = cdk.App()
KybStack(
    app,
    "KybStack",
    repo_root=REPO_ROOT,
    # Region-bound, not account-bound: synth needs no credentials, ELB access logs need a region.
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)
Validations.of(app).add_plugins(AwsSolutionsChecks(app))
app.synth()
