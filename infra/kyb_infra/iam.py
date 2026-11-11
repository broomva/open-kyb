"""One IAM role per component, least privilege (ARCHITECTURE §4.6 table).

Statements are written out by hand rather than with `grant_*` helpers, so the action list of each
role is visible in one place and `tests/test_stack.py` can pin it exactly. No role gets a wildcard
action, `s3:Delete*`, or `kms:*`.
"""

from aws_cdk import Aws
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_rds as rds
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

ECS_TASKS = iam.ServicePrincipal("ecs-tasks.amazonaws.com")


class ComponentRoles(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        archive: s3.IBucket,
        anchor: s3.IBucket,
        policy: s3.IBucket,
        policy_object_key: str,
        db: rds.DatabaseInstance,
        audit_key: kms.IKey,
        tenant_key: kms.IKey,
        secrets: list[secretsmanager.ISecret],
    ) -> None:
        super().__init__(scope, construct_id)

        def db_user(user: str) -> str:
            prefix = f"arn:{Aws.PARTITION}:rds-db:{Aws.REGION}:{Aws.ACCOUNT_ID}"
            return f"{prefix}:dbuser:{db.instance_resource_id}/{user}"

        def allow(actions: list[str], resources: list[str]) -> iam.PolicyStatement:
            return iam.PolicyStatement(
                effect=iam.Effect.ALLOW, actions=actions, resources=resources
            )

        # api: the Fargate task role. Writes the ledger (INSERT-only, RLS in Postgres), reads
        # and writes sealed archives, signs checkpoints, reads its own secrets.
        self.api = iam.Role(
            self, "Api", assumed_by=ECS_TASKS, description="kyb API task role (ARCHITECTURE §4.6)"
        )
        self.api.add_to_policy(allow(["rds-db:connect"], [db_user("kyb_api")]))
        self.api.add_to_policy(allow(["s3:ListBucket"], [archive.bucket_arn]))
        self.api.add_to_policy(
            allow(["s3:GetObject", "s3:PutObject"], [archive.arn_for_objects("*")])
        )
        self.api.add_to_policy(allow(["kms:Sign"], [audit_key.key_arn]))
        # GenerateDataKey: tenant-key envelope encryption and S3 SSE-KMS writes both need it.
        self.api.add_to_policy(
            allow(["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"], [tenant_key.key_arn])
        )
        self.api.add_to_policy(
            allow(["secretsmanager:GetSecretValue"], [s.secret_arn for s in secrets])
        )
        # Secrets are readable by the api role only: everyone else is denied at the resource.
        for secret in secrets:
            secret.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    principals=[iam.AnyPrincipal()],
                    actions=["secretsmanager:GetSecretValue"],
                    resources=["*"],
                    conditions={"ArnNotEquals": {"aws:PrincipalArn": self.api.role_arn}},
                )
            )

        # verifier: read-only audit: SELECT-only DB user, archive and anchor reads, public key.
        self.verifier = iam.Role(
            self,
            "Verifier",
            assumed_by=ECS_TASKS,
            description="kyb audit verifier: read-only (ARCHITECTURE §4.6)",
        )
        self.verifier.add_to_policy(allow(["rds-db:connect"], [db_user("kyb_auditor")]))
        self.verifier.add_to_policy(
            allow(["s3:ListBucket"], [archive.bucket_arn, anchor.bucket_arn])
        )
        self.verifier.add_to_policy(
            allow(["s3:GetObject"], [archive.arn_for_objects("*"), anchor.arn_for_objects("*")])
        )
        self.verifier.add_to_policy(allow(["kms:GetPublicKey"], [audit_key.key_arn]))
        # Decrypt: the archive is SSE-KMS under the tenant key, so reading it needs this.
        self.verifier.add_to_policy(allow(["kms:Decrypt"], [tenant_key.key_arn]))

        # anchor-writer: append-only. PutObject on the Object Lock bucket; no delete, no read.
        self.anchor_writer = iam.Role(
            self,
            "AnchorWriter",
            assumed_by=ECS_TASKS,
            description="kyb anchor writer: PutObject on the anchor bucket only",
        )
        self.anchor_writer.add_to_policy(allow(["s3:PutObject"], [anchor.arn_for_objects("*")]))

        # compliance-editor: a human role (MFA). Writes the one policy key, nothing else.
        self.compliance_editor = iam.Role(
            self,
            "ComplianceEditor",
            assumed_by=iam.AccountPrincipal(Aws.ACCOUNT_ID).with_conditions(
                {"Bool": {"aws:MultiFactorAuthPresent": "true"}}
            ),
            description="kyb compliance editor: PutObject on compliance_rules.md only",
        )
        self.compliance_editor.add_to_policy(
            allow(["s3:PutObject"], [policy.arn_for_objects(policy_object_key)])
        )
