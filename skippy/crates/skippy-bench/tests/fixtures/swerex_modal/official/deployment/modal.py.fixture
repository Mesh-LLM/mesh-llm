import asyncio
import logging
import os
import time
import uuid
from pathlib import Path, PurePath
from typing import Any

import boto3
import modal
from botocore.exceptions import NoCredentialsError
from typing_extensions import Self

from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import ModalDeploymentConfig
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import IsAliveResponse
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.log import get_logger
from swerex.utils.wait import _wait_until_alive

__all__ = ["ModalDeployment"]


def _get_modal_user() -> str:
    # not sure how to get the user from the modal api
    return modal.config._profile  # type: ignore


class _ImageBuilder:
    """_ImageBuilder.auto() is used by ModalDeployment"""

    def __init__(self, *, install_pipx: bool = True, logger: logging.Logger | None = None):
        self.logger = logger or get_logger("rex_image_builder")
        self._install_pipx = install_pipx

    def from_file(self, image: PurePath, *, build_context: PurePath | None = None) -> modal.Image:
        self.logger.info(f"Building image from file {image}")
        if build_context is None:
            build_context = Path(image).resolve().parent
        build_context = Path(build_context)
        self.logger.debug(f"Using build context {build_context}")
        return modal.Image.from_dockerfile(
            str(image),
            context_dir=str(build_context),
        )

    def from_registry(self, image: str) -> modal.Image:
        self.logger.info(f"Building image from docker registry {image}")
        if os.environ.get("DOCKER_USERNAME") and os.environ.get("DOCKER_PASSWORD"):
            secret = modal.Secret.from_dict(
                {
                    "DOCKER_USERNAME": os.environ["DOCKER_USERNAME"],
                    "DOCKER_PASSWORD": os.environ["DOCKER_PASSWORD"],
                }
            )
            secrets = [secret]
            self.logger.debug("Docker login credentials were provided")
        else:
            self.logger.warning("DOCKER_USERNAME and DOCKER_PASSWORD not set. Using public images.")
            secrets = None
        return modal.Image.from_registry(image, secrets=secrets)

    def from_ecr(self, image: str) -> modal.Image:
        self.logger.info(f"Building image from ECR {image}")
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            aws_access_key_id = credentials.access_key
            aws_secret_access_key = credentials.secret_key
            secret = modal.Secret.from_dict(
                {
                    "AWS_ACCESS_KEY_ID": aws_access_key_id,
                    "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
                }
            )
            return modal.Image.from_ecr(  # type: ignore
                image,
                secrets=[secret],
            )
        except NoCredentialsError as e:
            msg = "AWS credentials not found. Please configure your AWS credentials."
            raise ValueError(msg) from e

    def ensure_pipx_installed(self, image: modal.Image) -> modal.Image:
        image = image.apt_install("pipx")
        return image.run_commands("pipx ensurepath")

    def auto(self, image_spec: str | modal.Image | PurePath) -> modal.Image:
        if isinstance(image_spec, modal.Image):
            image = image_spec
        elif isinstance(image_spec, PurePath) and not Path(image_spec).is_file():
            msg = f"File {image_spec} does not exist"
            raise FileNotFoundError(msg)
        elif Path(image_spec).is_file():
            image = self.from_file(Path(image_spec))
        elif "amazonaws.com" in image_spec:  # type: ignore
            image = self.from_ecr(image_spec)  # type: ignore
        else:
            image = self.from_registry(image_spec)  # type: ignore

        if self._install_pipx:
            image = self.ensure_pipx_installed(image)

        return image


class ModalDeployment(AbstractDeployment):
    # Leave the constructor args for now, because image can take a modal.Image
    # but we don't want to make this part of the config class because it would
    # force us to have modal installed/import it...
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        image: str | modal.Image | PurePath,
        startup_timeout: float = 0.4,
        runtime_timeout: float = 3600.0,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        install_pipx: bool = True,
        deployment_timeout: float = 3600.0,
    ):
        """Deployment for modal.com. The deployment will only start when the
        `start` method is being called.

        Args:
            image: Image to use for the deployment. One of the following:
                1. `modal.Image` object
                2. Path to a Dockerfile
                3. Dockerhub image name (e.g. `python:3.11-slim`)
                4. ECR image name (e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:tag`)
            startup_timeout: The time to wait for the runtime to start.
            runtime_timeout: The runtime timeout.
            deployment_timeout: The deployment timeout.
            modal_sandbox_kwargs: Additional arguments to pass to `modal.Sandbox.create`
        """
        self._image = _ImageBuilder(install_pipx=install_pipx, logger=logger).auto(image)
        self._runtime: RemoteRuntime | None = None
        self._startup_timeout = startup_timeout
        self._sandbox: modal.Sandbox | None = None
        self._port = 8880
        self.logger = logger or get_logger("rex-deploy")
        self._app = modal.App.lookup("swe-rex", create_if_missing=True)
        self._user = _get_modal_user()
        self._runtime_timeout = runtime_timeout
        self._deployment_timeout = deployment_timeout
        if modal_sandbox_kwargs is None:
            modal_sandbox_kwargs = {}
        self._modal_kwargs = modal_sandbox_kwargs
        self._hooks = CombinedDeploymentHook()

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: ModalDeploymentConfig) -> Self:
        return cls(
            image=config.image,
            install_pipx=config.install_pipx,
            startup_timeout=config.startup_timeout,
            runtime_timeout=config.runtime_timeout,
            deployment_timeout=config.deployment_timeout,
            modal_sandbox_kwargs=config.modal_sandbox_kwargs,
        )

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive. The return value can be
        tested with bool().

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        if self._runtime is None or self._sandbox is None:
            raise DeploymentNotStartedError()
        exit_code = await self._sandbox.poll.aio()
        if exit_code is not None:
            msg = "Container process terminated."
            output = "stdout:\n" + await self._sandbox.stdout.read.aio()  # type: ignore
            output += "\nstderr:\n" + await self._sandbox.stderr.read.aio()  # type: ignore
            msg += "\n" + output
            raise RuntimeError(msg)
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        assert self._runtime is not None
        return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=self._runtime._config.timeout)

    def _start_swerex_cmd(self, token: str) -> str:
        """Start swerex-server on the remote. If swerex is not installed arelady,
        install pipx and then run swerex-server with pipx run
        """
        rex_args = f"--port {self._port} --auth-token {token}"
        return f"{REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args}"

    async def get_modal_log_url(self) -> str:
        """Returns URL to modal logs

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        return f"https://modal.com/apps/{self._user}/main/deployed/{self.app.name}?activeTab=logs&taskId={await self.sandbox._get_task_id.aio()}"

    async def start(
        self,
    ):
        """Starts the runtime."""
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started. Ignoring duplicate start() call.")
            return

        self.logger.info("Starting modal sandbox")
        self._hooks.on_custom_step("Starting modal sandbox")
        t0 = time.time()
        token = self._get_token()
        self._sandbox = await modal.Sandbox.create.aio(
            "/usr/bin/env",
            "bash",
            "-c",
            self._start_swerex_cmd(token),
            image=self._image,
            timeout=int(self._deployment_timeout),
            unencrypted_ports=[self._port],
            app=self._app,
            **self._modal_kwargs,
        )
        tunnels = await self._sandbox.tunnels.aio()
        tunnel = tunnels[self._port]
        elapsed_sandbox_creation = time.time() - t0
        self.logger.info(f"Sandbox ({self._sandbox.object_id}) created in {elapsed_sandbox_creation:.2f}s")
        self.logger.info(f"Check sandbox logs at {await self.get_modal_log_url()}")
        self.logger.info(f"Sandbox created with id {self._sandbox.object_id}")
        await asyncio.sleep(1)
        self.logger.info(f"Starting runtime at {tunnel.url}")
        self._hooks.on_custom_step("Starting runtime")
        self._runtime = RemoteRuntime(
            host=tunnel.url, timeout=self._runtime_timeout, auth_token=token, logger=self.logger
        )
        remaining_startup_timeout = max(0, self._startup_timeout - elapsed_sandbox_creation)
        t1 = time.time()
        await self._wait_until_alive(timeout=remaining_startup_timeout)
        self.logger.info(f"Runtime started in {time.time() - t1:.2f}s")

    async def stop(self):
        """Stops the runtime."""
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None
        if self._sandbox is not None:
            exit_code = await self._sandbox.poll.aio()
            if exit_code is not None:
                await self._sandbox.terminate.aio()
        self._sandbox = None
        self._app = None

    @property
    def runtime(self) -> RemoteRuntime:
        """Returns the runtime if running.

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    @property
    def app(self) -> modal.App:
        """Returns the modal app

        Raises:
            DeploymentNotStartedError: If the deployment is not started.
        """
        if self._app is None:
            raise DeploymentNotStartedError()
        return self._app

    @property
    def sandbox(self) -> modal.Sandbox:
        """Returns the modal sandbox

        Raises:
            DeploymentNotStartedError: If the deployment is not started.
        """
        if self._sandbox is None:
            raise DeploymentNotStartedError()
        return self._sandbox
