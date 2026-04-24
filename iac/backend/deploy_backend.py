from dataclasses import dataclass

import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp

@dataclass(frozen=True)
class EnvironmentVariables:
    mongodb_uri: str
    mongodb_name: str

    def get_env_vars(self) -> list[gcp.cloudrunv2.ServiceTemplateContainerEnvArgs]:
        return [
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="MONGO_URL", value=self.mongodb_uri),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name="MONGO_DB_NAME", value=self.mongodb_name),
        ]


def deploy_backend(*,
                   gcp_project_id: str,
                   gcp_project_region: str,
                   docker_repository_id: str,
                   environment_name: str,
                   env_vars: EnvironmentVariables):
    image_id = f"matching-algorithm-{environment_name}"
    pulumi.log.info(f"Building image {image_id}")

    image_name = f"{gcp_project_region}-docker.pkg.dev/{gcp_project_id}/{docker_repository_id}/{image_id}:latest"
    pulumi.log.info(f"Image name: {image_name}")

    image = docker.Image(
        image_id,
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
                    image=image_name,
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
