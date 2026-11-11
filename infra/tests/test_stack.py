"""Assertions on the synthesized KybStack template (ARCHITECTURE §4.6).

Nothing here touches AWS: `Template.from_stack` renders CloudFormation locally. IAM checks compare
the *exact* set of (action, resource) pairs each component role holds, so adding a permission, or
widening a resource, fails a test and has to be argued for.
"""

import json
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template
from kyb_infra.stack import POLICY_OBJECT_KEY, KybStack

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV = cdk.Environment(account="987654321098", region="us-east-1")


@pytest.fixture(scope="module")
def stack() -> KybStack:
    return KybStack(cdk.App(), "KybStack", repo_root=REPO_ROOT, env=ENV)


@pytest.fixture(scope="module")
def template(stack: KybStack) -> dict:
    return Template.from_stack(stack).to_json()


def render(value) -> str:
    """CloudFormation intrinsics as one comparable string: `${Logical.Attr}`, `${Ref}`."""
    if isinstance(value, str):
        return value
    if "Ref" in value:
        return "${" + value["Ref"] + "}"
    if "Fn::GetAtt" in value:
        return "${" + ".".join(value["Fn::GetAtt"]) + "}"
    if "Fn::Join" in value:
        sep, parts = value["Fn::Join"]
        return sep.join(render(p) for p in parts)
    raise AssertionError(f"unrendered intrinsic: {value}")


def listed(value) -> list:
    return value if isinstance(value, list) else [value]


def resources_of(template: dict, type_: str) -> dict[str, dict]:
    return {k: v for k, v in template["Resources"].items() if v["Type"] == type_}


def role_pairs(stack: KybStack, template: dict, role) -> set[tuple[str, str]]:
    """Every (action, resource) an Allow in the role's policies grants."""
    role_id = stack.get_logical_id(role.node.default_child)
    pairs = set()
    for policy in resources_of(template, "AWS::IAM::Policy").values():
        if {"Ref": role_id} not in policy["Properties"]["Roles"]:
            continue
        for st in policy["Properties"]["PolicyDocument"]["Statement"]:
            assert st["Effect"] == "Allow"
            for action in listed(st["Action"]):
                for resource in listed(st["Resource"]):
                    pairs.add((action, render(resource)))
    return pairs


def lid(stack: KybStack, construct) -> str:
    return stack.get_logical_id(construct.node.default_child)


def arn(stack, construct) -> str:
    return "${" + lid(stack, construct) + ".Arn}"


def db_user(stack: KybStack, user: str) -> str:
    return (
        "arn:${AWS::Partition}:rds-db:${AWS::Region}:${AWS::AccountId}:dbuser:"
        f"${{{lid(stack, stack.db)}.DbiResourceId}}/{user}"
    )


def secret_refs(stack: KybStack) -> set[str]:
    return {
        "${" + stack.get_logical_id(stack.db.secret.node.default_child) + "}",
        "${" + lid(stack, stack.model_keys) + "}",
    }


def test_api_role_has_exactly_its_actions_on_scoped_resources(stack, template):
    archive, audit, tenant = (
        arn(stack, stack.archive),
        arn(stack, stack.audit_key),
        arn(stack, stack.tenant_key),
    )
    expected = {
        ("rds-db:connect", db_user(stack, "kyb_api")),
        ("s3:ListBucket", archive),
        ("s3:GetObject", archive + "/*"),
        ("s3:PutObject", archive + "/*"),
        ("kms:Sign", audit),
        ("kms:Encrypt", tenant),
        ("kms:Decrypt", tenant),
        ("kms:GenerateDataKey", tenant),
        *(("secretsmanager:GetSecretValue", s) for s in secret_refs(stack)),
    }
    assert role_pairs(stack, template, stack.roles.api) == expected


def test_verifier_role_is_read_only(stack, template):
    archive, anchor = arn(stack, stack.archive), arn(stack, stack.anchor)
    expected = {
        ("rds-db:connect", db_user(stack, "kyb_auditor")),
        ("s3:ListBucket", archive),
        ("s3:ListBucket", anchor),
        ("s3:GetObject", archive + "/*"),
        ("s3:GetObject", anchor + "/*"),
        ("kms:GetPublicKey", arn(stack, stack.audit_key)),
        ("kms:Decrypt", arn(stack, stack.tenant_key)),  # the archive is SSE-KMS
    }
    assert role_pairs(stack, template, stack.roles.verifier) == expected


def test_anchor_writer_can_only_put_to_the_anchor_bucket(stack, template):
    expected = {("s3:PutObject", arn(stack, stack.anchor) + "/*")}
    assert role_pairs(stack, template, stack.roles.anchor_writer) == expected


def test_compliance_editor_can_only_put_the_policy_key(stack, template):
    expected = {("s3:PutObject", f"{arn(stack, stack.policy)}/{POLICY_OBJECT_KEY}")}
    assert role_pairs(stack, template, stack.roles.compliance_editor) == expected


def test_component_roles_carry_no_managed_or_inline_policies(stack, template):
    for role in (
        stack.roles.api,
        stack.roles.verifier,
        stack.roles.anchor_writer,
        stack.roles.compliance_editor,
    ):
        props = template["Resources"][lid(stack, role)]["Properties"]
        assert "ManagedPolicyArns" not in props
        assert "Policies" not in props


def test_trust_policies(stack, template):
    def trust(role) -> dict:
        return template["Resources"][lid(stack, role)]["Properties"]["AssumeRolePolicyDocument"][
            "Statement"
        ]

    for role in (stack.roles.api, stack.roles.verifier, stack.roles.anchor_writer):
        assert [s["Principal"] for s in trust(role)] == [{"Service": "ecs-tasks.amazonaws.com"}]
    (editor,) = trust(stack.roles.compliance_editor)
    assert editor["Condition"] == {"Bool": {"aws:MultiFactorAuthPresent": "true"}}
    assert render(editor["Principal"]["AWS"]) == "arn:${AWS::Partition}:iam::${AWS::AccountId}:root"


def test_no_wildcard_action_and_no_wildcard_resource_on_writes(template):
    statements = []
    for type_ in ("AWS::IAM::Policy", "AWS::IAM::ManagedPolicy"):
        for policy in resources_of(template, type_).values():
            statements += policy["Properties"]["PolicyDocument"]["Statement"]
    for role in resources_of(template, "AWS::IAM::Role").values():
        for inline in role["Properties"].get("Policies", []):
            statements += inline["PolicyDocument"]["Statement"]
    for policy in resources_of(template, "AWS::S3::BucketPolicy").values():
        statements += policy["Properties"]["PolicyDocument"]["Statement"]
    for policy in resources_of(template, "AWS::SecretsManager::ResourcePolicy").values():
        statements += policy["Properties"]["ResourcePolicy"]["Statement"]
    assert statements
    read_only_on_star = {"ecr:GetAuthorizationToken"}  # no resource-level scoping exists in IAM
    for st in statements:
        if st["Effect"] != "Allow":
            continue
        for action in listed(st["Action"]):
            assert action != "*" and not action.endswith(":*"), st
            if "*" in listed(st["Resource"]):
                assert action in read_only_on_star, st


def test_secrets_are_denied_to_everyone_but_the_api_role(stack, template):
    policies = resources_of(template, "AWS::SecretsManager::ResourcePolicy")
    assert len(policies) == 2  # database credentials and model API keys
    api_arn = "${" + lid(stack, stack.roles.api) + ".Arn}"
    for policy in policies.values():
        (st,) = policy["Properties"]["ResourcePolicy"]["Statement"]
        assert st["Effect"] == "Deny" and st["Action"] == "secretsmanager:GetSecretValue"
        assert st["Principal"] == {"AWS": "*"}
        condition = st["Condition"]["ArnNotEquals"]["aws:PrincipalArn"]
        assert render(condition) == api_arn


def test_secrets_use_the_tenant_key(stack, template):
    tenant = "${" + lid(stack, stack.tenant_key) + ".Arn}"
    for secret in resources_of(template, "AWS::SecretsManager::Secret").values():
        assert render(secret["Properties"]["KmsKeyId"]) == tenant


def buckets(stack: KybStack) -> dict[str, object]:
    return {
        "access_logs": stack.access_logs,
        "archive": stack.archive,
        "anchor": stack.anchor,
        "policy": stack.policy,
    }


def test_every_bucket_blocks_public_access_is_encrypted_and_tls_only(stack, template):
    found = resources_of(template, "AWS::S3::Bucket")
    assert len(found) == 4
    bucket_policies = resources_of(template, "AWS::S3::BucketPolicy")
    for bucket in found.values():
        props = bucket["Properties"]
        assert props["PublicAccessBlockConfiguration"] == {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        }
        assert props["BucketEncryption"]["ServerSideEncryptionConfiguration"]
    for name, bucket in buckets(stack).items():
        lid_ = lid(stack, bucket)
        policy = next(
            p for p in bucket_policies.values() if p["Properties"]["Bucket"] == {"Ref": lid_}
        )
        denies = [
            s for s in policy["Properties"]["PolicyDocument"]["Statement"] if s["Effect"] == "Deny"
        ]
        assert any(s["Condition"] == {"Bool": {"aws:SecureTransport": "false"}} for s in denies), (
            name
        )


def test_archive_is_kms_encrypted_with_the_tenant_key_and_versioned(stack, template):
    props = template["Resources"][lid(stack, stack.archive)]["Properties"]
    (rule,) = props["BucketEncryption"]["ServerSideEncryptionConfiguration"]
    default = rule["ServerSideEncryptionByDefault"]
    assert default["SSEAlgorithm"] == "aws:kms"
    assert render(default["KMSMasterKeyID"]) == "${" + lid(stack, stack.tenant_key) + ".Arn}"
    assert props["VersioningConfiguration"] == {"Status": "Enabled"}


def test_anchor_bucket_has_object_lock_in_compliance_mode(stack, template):
    props = template["Resources"][lid(stack, stack.anchor)]["Properties"]
    assert props["ObjectLockEnabled"] is True
    retention = props["ObjectLockConfiguration"]["Rule"]["DefaultRetention"]
    assert retention["Mode"] == "COMPLIANCE"
    assert retention["Days"] >= 1
    assert props["VersioningConfiguration"] == {"Status": "Enabled"}


def test_policy_bucket_is_versioned_and_has_no_object_lock(stack, template):
    props = template["Resources"][lid(stack, stack.policy)]["Properties"]
    assert props["VersioningConfiguration"] == {"Status": "Enabled"}
    assert "ObjectLockEnabled" not in props


def test_buckets_survive_stack_deletion(template):
    for bucket in resources_of(template, "AWS::S3::Bucket").values():
        assert bucket["DeletionPolicy"] == "Retain"


def test_rds_is_encrypted_private_and_uses_iam_auth(stack, template):
    (db,) = resources_of(template, "AWS::RDS::DBInstance").values()
    props = db["Properties"]
    assert props["Engine"] == "postgres"
    assert props["StorageEncrypted"] is True
    assert render(props["KmsKeyId"]) == "${" + lid(stack, stack.tenant_key) + ".Arn}"
    assert props["PubliclyAccessible"] is False
    assert props["EnableIAMDatabaseAuthentication"] is True
    assert props["DeletionProtection"] is True
    assert props["MultiAZ"] is True
    assert db["DeletionPolicy"] == "Retain"


def test_rds_sits_in_isolated_subnets_and_only_the_service_reaches_it(stack, template):
    (group,) = resources_of(template, "AWS::RDS::DBSubnetGroup").values()
    subnets = group["Properties"]["SubnetIds"]
    isolated = {
        lid_
        for lid_, s in resources_of(template, "AWS::EC2::Subnet").items()
        if "data" in json.dumps(s["Properties"]["Tags"])
    }
    assert {s["Ref"] for s in subnets} == isolated
    # only the service security group may connect: one 5432 rule, sourced from a security group
    ingress_5432 = [
        i
        for i in resources_of(template, "AWS::EC2::SecurityGroupIngress").values()
        if i["Properties"].get("FromPort") == 5432
    ]
    assert len(ingress_5432) == 1
    source = ingress_5432[0]["Properties"]["SourceSecurityGroupId"]
    assert source == {"Fn::GetAtt": [lid(stack, stack.service_sg), "GroupId"]}


def test_server_access_logs_can_be_delivered_to_the_sink(stack, template):
    policy = next(
        p
        for p in resources_of(template, "AWS::S3::BucketPolicy").values()
        if p["Properties"]["Bucket"] == {"Ref": lid(stack, stack.access_logs)}
    )
    (st,) = [
        s
        for s in policy["Properties"]["PolicyDocument"]["Statement"]
        if s.get("Principal") == {"Service": "logging.s3.amazonaws.com"}
    ]
    assert st["Effect"] == "Allow" and st["Action"] == "s3:PutObject"
    assert st["Condition"]["StringEquals"] == {"aws:SourceAccount": "987654321098"}
    assert len(st["Condition"]["ArnLike"]["aws:SourceArn"]) == 3  # archive, anchor, policy only
    assert len(st["Resource"]) == 3


def test_task_definition_is_non_root_with_read_only_root_filesystem(stack, template):
    (task,) = resources_of(template, "AWS::ECS::TaskDefinition").values()
    props = task["Properties"]
    (container,) = props["ContainerDefinitions"]
    assert container["User"] == "10001:10001"
    assert container["ReadonlyRootFilesystem"] is True
    assert {m["ContainerPath"] for m in container["MountPoints"]} == {"/tmp", "/data"}
    assert all(not m.get("ReadOnly") for m in container["MountPoints"])
    assert props["RuntimePlatform"]["CpuArchitecture"] == "ARM64"
    assert render(props["TaskRoleArn"]) == "${" + lid(stack, stack.roles.api) + ".Arn}"


def test_container_is_not_privileged_and_holds_no_secret_values(template):
    (task,) = resources_of(template, "AWS::ECS::TaskDefinition").values()
    (container,) = task["Properties"]["ContainerDefinitions"]
    assert container.get("Privileged") in (None, False)
    assert "Secrets" not in container
    literals = {
        e["Name"]: e["Value"] for e in container["Environment"] if isinstance(e["Value"], str)
    }
    assert literals == {"KYB_WORKSPACE": "/data"}  # everything else is an ARN or bucket reference


def test_service_runs_in_private_subnets_without_a_public_ip(template):
    (service,) = resources_of(template, "AWS::ECS::Service").values()
    vpc_config = service["Properties"]["NetworkConfiguration"]["AwsvpcConfiguration"]
    assert vpc_config["AssignPublicIp"] == "DISABLED"
    app_subnets = {
        lid_
        for lid_, s in resources_of(template, "AWS::EC2::Subnet").items()
        if "app" in json.dumps(s["Properties"]["Tags"])
    }
    assert {s["Ref"] for s in vpc_config["Subnets"]} == app_subnets
    assert service["Properties"]["LaunchType"] == "FARGATE"
    assert service["Properties"]["DeploymentConfiguration"]["MinimumHealthyPercent"] == 100


def test_alb_terminates_tls_and_redirects_http(template):
    listeners = resources_of(template, "AWS::ElasticLoadBalancingV2::Listener").values()
    by_port = {listener["Properties"]["Port"]: listener["Properties"] for listener in listeners}
    assert by_port[443]["Protocol"] == "HTTPS"
    assert by_port[443]["SslPolicy"].startswith("ELBSecurityPolicy-TLS13")
    assert by_port[443]["Certificates"][0]["CertificateArn"] == {"Ref": "CertificateArn"}
    (redirect,) = by_port[80]["DefaultActions"]
    assert redirect["Type"] == "redirect"
    assert redirect["RedirectConfig"]["Protocol"] == "HTTPS"
    (alb,) = resources_of(template, "AWS::ElasticLoadBalancingV2::LoadBalancer").values()
    attrs = {a["Key"]: a["Value"] for a in alb["Properties"]["LoadBalancerAttributes"]}
    assert attrs["routing.http.drop_invalid_header_fields.enabled"] == "true"
    assert attrs["access_logs.s3.enabled"] == "true"
    assert alb["Properties"]["Scheme"] == "internal"  # public exposure is an explicit context flag


def test_private_network_has_the_five_endpoints(template):
    endpoints = resources_of(template, "AWS::EC2::VPCEndpoint").values()
    names = {render(e["Properties"]["ServiceName"]).rsplit(".", 1)[-1] for e in endpoints}
    assert names == {"s3", "kms", "secretsmanager", "rds", "logs"}


def test_keys(stack, template):
    audit = template["Resources"][lid(stack, stack.audit_key)]["Properties"]
    assert audit["KeyUsage"] == "SIGN_VERIFY"
    assert audit["KeySpec"] == "ECC_NIST_EDWARDS25519"
    tenant = template["Resources"][lid(stack, stack.tenant_key)]["Properties"]
    assert tenant["EnableKeyRotation"] is True
    for key in (audit, tenant):
        for st in key["KeyPolicy"]["Statement"]:
            # the default policy only delegates to IAM through the account root
            assert render(st["Principal"]["AWS"]).endswith(":root")


def test_deploy_needs_a_certificate_parameter_not_a_baked_in_arn(template):
    assert template["Parameters"]["CertificateArn"]["Type"] == "String"
    assert "Default" not in template["Parameters"]["CertificateArn"]
