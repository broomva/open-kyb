"""The open-kyb KYB service on AWS, exactly ARCHITECTURE §4.6 (spec, spec).

Postgres is authoritative; S3 holds only sealed archives, the WORM audit anchor and the policy file.
One IAM role per component, each with only the actions the §4.6 table gives it (see `iam.py`).
The image is the service image built from the root Dockerfile, never the agent sandbox (spec).
"""

from pathlib import Path

from aws_cdk import Acknowledgment, CfnParameter, Duration, RemovalPolicy, Stack, Validations
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

from kyb_infra.iam import ComponentRoles

SERVICE_PORT = 8080
SERVICE_UID_GID = "10001:10001"
DB_PORT = 5432
POLICY_OBJECT_KEY = "compliance_rules.md"  # the one key the compliance-editor may write
# `kyb serve` (the chat/API tier) is planned, not on main: override with -c kyb:serviceCommand=...
DEFAULT_SERVICE_COMMAND = ["serve", "--host", "0.0.0.0", "--port", str(SERVICE_PORT)]


class KybStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, repo_root: Path, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        node = self.node
        internet_facing = bool(node.try_get_context("kyb:internetFacing") or False)
        retention_days = int(node.try_get_context("kyb:anchorRetentionDays") or 365)
        command = node.try_get_context("kyb:serviceCommand") or DEFAULT_SERVICE_COMMAND

        # TLS certificate: an ACM ARN supplied at deploy time, never invented here.
        certificate_arn = CfnParameter(
            self,
            "CertificateArn",
            type="String",
            description="ARN of the ACM certificate for the ALB HTTPS listener.",
            allowed_pattern=r"arn:aws[a-z-]*:acm:[a-z0-9-]+:[0-9]{12}:certificate/.+",
        )

        # --- keys ------------------------------------------------------------------------------
        self.audit_key = kms.Key(
            self,
            "AuditSigningKey",
            description="Signs per-tenant Merkle checkpoints (spec). Sign/verify only.",
            key_spec=kms.KeySpec.ECC_NIST_EDWARDS25519,
            key_usage=kms.KeyUsage.SIGN_VERIFY,
            removal_policy=RemovalPolicy.RETAIN,
            pending_window=Duration.days(30),
        )
        self.tenant_key = kms.Key(
            self,
            "TenantKey",
            description="Envelope encryption for tenant data: RDS, archive, secrets.",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
            pending_window=Duration.days(30),
        )

        # --- network ---------------------------------------------------------------------------
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=1,  # egress is for the model API only; one NAT is the sample-scale choice
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="edge", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="app", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="data", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24
                ),
            ],
        )
        self.vpc.add_flow_log("FlowLogs")
        self.endpoint_sgs: list[ec2.ISecurityGroup] = []
        private = [
            ec2.SubnetSelection(subnet_group_name="app"),
            ec2.SubnetSelection(subnet_group_name="data"),
        ]
        self.vpc.add_gateway_endpoint(
            "S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3, subnets=private
        )
        for name, service in (
            ("Kms", ec2.InterfaceVpcEndpointAwsService.KMS),
            ("SecretsManager", ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER),
            ("Rds", ec2.InterfaceVpcEndpointAwsService.RDS),
            ("Logs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
        ):
            endpoint = self.vpc.add_interface_endpoint(
                f"{name}Endpoint",
                service=service,
                subnets=ec2.SubnetSelection(subnet_group_name="app"),
            )
            self.endpoint_sgs.extend(endpoint.connections.security_groups)

        # --- object storage --------------------------------------------------------------------
        self.access_logs = self._bucket(
            "AccessLogs", encryption=s3.BucketEncryption.S3_MANAGED, versioned=False
        )
        self.access_logs.add_lifecycle_rule(expiration=Duration.days(400))
        self.archive = self._bucket(
            "Archive",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.tenant_key,
            bucket_key_enabled=True,
            logs=self.access_logs,
        )
        self.anchor = self._bucket(
            "AuditAnchor",
            # signed checkpoints only, so SSE-S3: the anchor writer needs no KMS grant
            encryption=s3.BucketEncryption.S3_MANAGED,
            object_lock_default_retention=s3.ObjectLockRetention.compliance(
                Duration.days(retention_days)
            ),
            logs=self.access_logs,
        )
        self.policy = self._bucket(
            "Policy", encryption=s3.BucketEncryption.S3_MANAGED, logs=self.access_logs
        )

        # --- database --------------------------------------------------------------------------
        db_sg = ec2.SecurityGroup(self, "DbSg", vpc=self.vpc, allow_all_outbound=False)
        self.db = rds.DatabaseInstance(
            self,
            "Db",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_17),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MEDIUM),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_group_name="data"),
            security_groups=[db_sg],
            port=DB_PORT,
            multi_az=True,
            storage_encrypted=True,
            storage_encryption_key=self.tenant_key,
            publicly_accessible=False,
            iam_authentication=True,
            credentials=rds.Credentials.from_generated_secret(
                "kyb_admin", encryption_key=self.tenant_key
            ),
            backup_retention=Duration.days(14),
            deletion_protection=True,
            removal_policy=RemovalPolicy.RETAIN,
            cloudwatch_logs_exports=["postgresql"],
            auto_minor_version_upgrade=True,
            allocated_storage=50,
        )
        self.model_keys = secretsmanager.Secret(
            self,
            "ModelApiKeys",
            description="Model API keys. Values are set out of band; sandboxes never hold them.",
            encryption_key=self.tenant_key,
            removal_policy=RemovalPolicy.RETAIN,
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}',
                generate_string_key="unused",
            ),
        )

        # --- IAM: one role per component -------------------------------------------------------
        self.roles = ComponentRoles(
            self,
            "Roles",
            archive=self.archive,
            anchor=self.anchor,
            policy=self.policy,
            policy_object_key=POLICY_OBJECT_KEY,
            db=self.db,
            audit_key=self.audit_key,
            tenant_key=self.tenant_key,
            secrets=[self.db.secret, self.model_keys],
        )

        # --- runtime ---------------------------------------------------------------------------
        cluster = ecs.Cluster(
            self, "Cluster", vpc=self.vpc, container_insights_v2=ecs.ContainerInsights.ENABLED
        )
        self.task = task = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            cpu=512,
            memory_limit_mib=1024,
            task_role=self.roles.api,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )
        for volume in ("tmp", "data"):  # the only writable paths; the root filesystem is read-only
            task.add_volume(name=volume)
        container = task.add_container(
            "Service",
            image=ecs.ContainerImage.from_asset(
                str(repo_root), file="Dockerfile", platform=ecr_assets.Platform.LINUX_ARM64
            ),
            command=list(command),
            user=SERVICE_UID_GID,
            readonly_root_filesystem=True,
            essential=True,
            environment={
                "KYB_WORKSPACE": "/data",
                "KYB_DB_SECRET_ARN": self.db.secret.secret_arn,
                "KYB_MODEL_KEYS_SECRET_ARN": self.model_keys.secret_arn,
                "KYB_AUDIT_KEY_ARN": self.audit_key.key_arn,
                "KYB_TENANT_KEY_ARN": self.tenant_key.key_arn,
                "KYB_ARCHIVE_BUCKET": self.archive.bucket_name,
                "KYB_ANCHOR_BUCKET": self.anchor.bucket_name,
                "KYB_POLICY_BUCKET": self.policy.bucket_name,
            },  # references only: the service fetches secret values at runtime with the api role
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="kyb",
                log_group=logs.LogGroup(
                    self,
                    "ServiceLogs",
                    retention=logs.RetentionDays.ONE_YEAR,
                    removal_policy=RemovalPolicy.RETAIN,
                ),
            ),
            health_check=ecs.HealthCheck(
                command=["CMD", "kyb", "--help"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(10),
            ),
            port_mappings=[ecs.PortMapping(container_port=SERVICE_PORT)],
        )
        container.add_mount_points(
            ecs.MountPoint(container_path="/tmp", source_volume="tmp", read_only=False),
            ecs.MountPoint(container_path="/data", source_volume="data", read_only=False),
        )

        self.service_sg = service_sg = ec2.SecurityGroup(
            self, "ServiceSg", vpc=self.vpc, allow_all_outbound=False
        )
        service_sg.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "model API and AWS APIs")
        db_sg.connections.allow_from(service_sg, ec2.Port.tcp(DB_PORT), "service to Postgres")
        service = ecs.FargateService(
            self,
            "Service",
            cluster=cluster,
            task_definition=task,
            desired_count=2,
            assign_public_ip=False,
            vpc_subnets=ec2.SubnetSelection(subnet_group_name="app"),
            security_groups=[service_sg],
            min_healthy_percent=100,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=Duration.seconds(60),
            enable_execute_command=False,
        )

        self.alb_sg = alb_sg = ec2.SecurityGroup(
            self, "AlbSg", vpc=self.vpc, allow_all_outbound=False
        )
        source = ec2.Peer.any_ipv4() if internet_facing else ec2.Peer.ipv4(self.vpc.vpc_cidr_block)
        alb_sg.add_ingress_rule(source, ec2.Port.tcp(443), "HTTPS")
        alb_sg.add_ingress_rule(source, ec2.Port.tcp(80), "HTTP, redirected to HTTPS")
        self.alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            vpc=self.vpc,
            internet_facing=internet_facing,
            vpc_subnets=ec2.SubnetSelection(subnet_group_name="edge" if internet_facing else "app"),
            security_group=alb_sg,
            drop_invalid_header_fields=True,
            deletion_protection=True,
        )
        self.alb.log_access_logs(self.access_logs, prefix="alb")
        self._allow_s3_server_access_logs()
        https = self.alb.add_listener(
            "Https",
            port=443,
            protocol=elbv2.ApplicationProtocol.HTTPS,
            certificates=[elbv2.ListenerCertificate.from_arn(certificate_arn.value_as_string)],
            ssl_policy=elbv2.SslPolicy.RECOMMENDED_TLS,
            open=False,
        )
        https.add_targets(
            "Api",
            port=SERVICE_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[service],
            health_check=elbv2.HealthCheck(path="/healthz", healthy_http_codes="200"),
            deregistration_delay=Duration.seconds(30),
        )
        self.alb.add_redirect(open=False)

        self._suppress()

    def _bucket(
        self,
        construct_id: str,
        *,
        encryption: s3.BucketEncryption,
        versioned: bool = True,
        logs: s3.IBucket | None = None,
        **kwargs,
    ) -> s3.Bucket:
        """A private, TLS-only, encrypted bucket that is kept on stack deletion."""
        return s3.Bucket(
            self,
            construct_id,
            encryption=encryption,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=versioned,
            removal_policy=RemovalPolicy.RETAIN,
            server_access_logs_bucket=logs,
            server_access_logs_prefix=f"{construct_id.lower()}/" if logs else None,
            **kwargs,
        )

    def _allow_s3_server_access_logs(self) -> None:
        """S3 server access logs need a bucket-policy grant for the logging service: the sink
        blocks public ACLs, so the legacy log-delivery ACL grant alone does not deliver."""
        sources = [self.archive, self.anchor, self.policy]
        self.access_logs.add_to_resource_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("logging.s3.amazonaws.com")],
                actions=["s3:PutObject"],
                resources=[
                    self.access_logs.arn_for_objects(f"{bucket.node.id.lower()}/*")
                    for bucket in sources
                ],
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {"aws:SourceArn": [bucket.bucket_arn for bucket in sources]},
                },
            )
        )

    def _suppress(self) -> None:
        """Every cdk-nag acknowledgement, with the reason an infrastructure reviewer can check."""

        def ack(construct: Construct, rule: str, reason: str) -> None:
            Validations.of(construct).acknowledge(
                Acknowledgment(id=f"AwsSolutions::{rule}", reason=reason)
            )

        def bucket_objects(bucket: s3.Bucket) -> str:
            logical = self.get_logical_id(bucket.node.default_child)
            return f"AwsSolutions-IAM5[Resource::<{logical}.Arn>/*]"

        ack(
            self.access_logs,
            "AwsSolutions-S1",
            "This is the server-access-log sink itself; logging it to itself would loop.",
        )
        ack(
            self.db,
            "AwsSolutions-RDS11",
            "Default port kept: the instance is in isolated subnets and reachable only from the "
            "service security group.",
        )
        ack(
            self.model_keys,
            "AwsSolutions-SMG4",
            "Third-party model API keys cannot be rotated by Secrets Manager; rotation is an "
            "operator procedure (docs/design/deployment.md).",
        )
        ack(
            self.db.node.find_child("Secret"),
            "AwsSolutions-SMG4",
            "The master secret only bootstraps the schema; the service connects with IAM database "
            "authentication as kyb_api / kyb_auditor.",
        )
        for group in [*self.endpoint_sgs, self.alb_sg]:
            ack(
                group,
                "AwsSolutions-EC23",
                "Ingress is 443 (and 80, redirected) from the VPC CIDR, which the rule cannot "
                "evaluate as a token; it is 0.0.0.0/0 only with -c kyb:internetFacing=true.",
            )
        ack(
            self.task,
            "AwsSolutions-ECS2",
            "The environment holds ARNs and bucket names, not secret values; the service reads "
            "secret values at runtime with the api role.",
        )
        # Wildcards: object paths under one named bucket, and the execution role's ECR token call.
        for role, buckets in (
            (self.roles.api, [self.archive]),
            (self.roles.verifier, [self.archive, self.anchor]),
            (self.roles.anchor_writer, [self.anchor]),
        ):
            for bucket in buckets:
                ack(
                    role.node.find_child("DefaultPolicy"),
                    bucket_objects(bucket),
                    "Object-level access needs the /* path suffix; the bucket is named and the "
                    "action list is fixed.",
                )
        ack(
            self.task.execution_role.node.find_child("DefaultPolicy"),
            "AwsSolutions-IAM5[Resource::*]",
            "ecr:GetAuthorizationToken has no resource-level scoping in IAM; it is the only "
            "wildcard and a read-class call.",
        )
