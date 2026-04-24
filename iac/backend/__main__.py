import os
import pulumi
from deploy_backend import deploy_backend
from dotenv import load_dotenv

from env_vars import EnvVars

def main():
    load_dotenv()
    stack = pulumi.get_stack()
    pulumi.log.info(f"using stack: {stack}")

    docker_repository_name = pulumi.Config().require("docker-repository_name")
    pulumi.log.info(f"using docker repository: {docker_repository_name}")

    project_id = pulumi.Config("gcp").require("project")
    pulumi.log.info(f"using project: {project_id}")

    region = pulumi.Config("gcp").require("region")
    pulumi.log.info(f"using region: {region}")

    deploy_backend(
        gcp_project_id=project_id,
        gcp_project_region=region,
        docker_repository_id=docker_repository_name,
        environment_name=stack,
        env_vars=EnvVars.construct_from_env()
    )

if __name__ == "__main__":
    main()
