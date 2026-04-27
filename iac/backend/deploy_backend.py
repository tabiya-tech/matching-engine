import base64
import os
from dataclasses import dataclass

import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp

from env_vars import EnvVars

API_GATEWAY_CONFIG_TEMPLATE_FILE = "api-gateway-cfg.yaml"

def _setup_api_gateway(
        *,
        environment_name: str,
        gcp_project_id: str,
        gcp_project_region: str,
        cloudrun: gcp.cloudrunv2.Service,
):
    """
    Set up a GCP API Gateway in front of the Cloud Run service.

    The API Gateway routes requests to the Cloud Run instance. To prevent direct
    calls to Cloud Run from the internet, a dedicated service account with
    'roles/run.invoker' is created for the API Gateway and granted sole access.
    """
    apigw_sa = gcp.serviceaccount.Account(
        f"{environment_name}-api-gateway-sa",
        account_id=f"{environment_name}-api-gw-sa",
        project=gcp_project_id,
        display_name=f"API Gateway service account ({environment_name})",
        create_ignore_already_exists=True,
    )

    apigw_api = gcp.apigateway.Api(
        f"{environment_name}-api-gateway-api",
        api_id=f"{environment_name}-matching-api",
        project=gcp_project_id,
    )

    template_path = os.path.join(os.path.dirname(__file__), API_GATEWAY_CONFIG_TEMPLATE_FILE)
    with open(template_path, "r") as f:
        template = f.read()

    apigw_config_yaml = cloudrun.uri.apply(
        lambda cloud_run_url: template
        .replace("__CLOUD_RUN_URL__", cloud_run_url)
        .replace("__ENVIRONMENT__", environment_name)
    )
    apigw_config_b64 = apigw_config_yaml.apply(
        lambda yml: base64.b64encode(yml.encode()).decode()
    )

    apigw_config = gcp.apigateway.ApiConfig(
        f"{environment_name}-api-gateway-cfg",
        api=apigw_api.api_id,
        project=gcp_project_id,
        openapi_documents=[
            gcp.apigateway.ApiConfigOpenapiDocumentArgs(
                document=gcp.apigateway.ApiConfigOpenapiDocumentDocumentArgs(
                    path=API_GATEWAY_CONFIG_TEMPLATE_FILE,
                    contents=apigw_config_b64,
                ),
            )
        ],
        gateway_config=gcp.apigateway.ApiConfigGatewayConfigArgs(
            backend_config=gcp.apigateway.ApiConfigGatewayConfigBackendConfigArgs(
                google_service_account=apigw_sa.email,
            )
        ),
    )

    api_gateway = gcp.apigateway.Gateway(
        f"{environment_name}-api-gateway",
        api_config=apigw_config.id,
        display_name=f"Matching API Gateway ({environment_name})",
        gateway_id=f"{environment_name}-matching-gw",
        project=gcp_project_id,
        region=gcp_project_region,
    )

    # Only grant roles/run.invoker to the API Gateway service account so the
    # Cloud Run service cannot be called directly from the internet.
    gcp.cloudrun.IamMember(
        f"{environment_name}-api-gw-invoker",
        project=gcp_project_id,
        location=gcp_project_region,
        service=cloudrun.name,
        role="roles/run.invoker",
        member=apigw_sa.email.apply(lambda email: f"serviceAccount:{email}"),
    )

    # Enable the managed API service so the gateway can serve requests.
    gcp.projects.Service(
        f"{environment_name}-api-gateway-managed-service",
        project=gcp_project_id,
        service=apigw_api.managed_service,
        opts=pulumi.ResourceOptions(depends_on=[api_gateway]),
    )

    pulumi.export("api_gateway_url", api_gateway.default_hostname.apply(lambda h: f"https://{h}"))
    pulumi.export("api_gateway_id", api_gateway.gateway_id)
    return api_gateway


def deploy_backend(*,
                   gcp_project_id: str,
                   gcp_project_region: str,
                   docker_repository_id: str,
                   environment_name: str,
                   env_vars: EnvVars):
    image_id = f"matching-algorithm-api"
    pulumi.log.info(f"Building image {image_id}")

    image_name = f"{gcp_project_region}-docker.pkg.dev/{gcp_project_id}/{docker_repository_id}/{image_id}:latest"
    pulumi.log.info(f"Image name: {image_name}")

    image = docker.Image(
        f"{image_id}-{environment_name}",
        image_name=image_name,
        build=docker.DockerBuildArgs(
            context="../../backend",
            platform="linux/amd64"),
        registry=None,  # use gcloud for authentication.
        opts=pulumi.ResourceOptions(depends_on=[]),
    )

    service = gcp.cloudrunv2.Service(
        f"{environment_name}-cloudrun-service",
        name=f"{environment_name}-cloudrun-service",
        project=gcp_project_id,
        location=gcp_project_region,
        ingress="INGRESS_TRAFFIC_ALL",
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=image.repo_digest,
                    resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                        limits={
                            'memory': "2Gi",
                            'cpu': "4",
                        },
                    ),
                    envs=env_vars.get_env_vars(),
                )
            ],
        ),
        opts=pulumi.ResourceOptions(depends_on=[image]),
    )

    pulumi.export("cloud_run_url", service.uri)

    _setup_api_gateway(
        environment_name=environment_name,
        gcp_project_id=gcp_project_id,
        gcp_project_region=gcp_project_region,
        cloudrun=service,
    )
