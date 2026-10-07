//! Finite old-intent edits on the exact official1.4 SDK, preserving its async interfaces.
use anyhow::{Result, bail};
const MODAL_CHANGES: &[(&str, &str)] = &[
    (
        "    def ensure_pipx_installed(self, image: modal.Image) -> modal.Image:\n        image = image.apt_install(\"pipx\")\n        return image.run_commands(\"pipx ensurepath\")\n\n",
        "    def ensure_pipx_installed(self, image: modal.Image) -> modal.Image:\n\n        image = image\\\n            .run_commands(\"pip config unset global.index-url || true\") \\\n            .run_commands(\"(apt update && apt install -y curl) || (apk update && apk add --no-cache curl bash)\") \\\n            .run_commands(\"curl https://pyenv.run | bash\") \\\n            .run_commands(\"(apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev) || (apk add --no-cache make build-base openssl-dev zlib-dev bzip2-dev readline-dev sqlite-dev git ncurses-dev xz tk-dev libxml2-dev xmlsec-dev libffi-dev xz-dev)\") \\\n            .run_commands(\"~/.pyenv/bin/pyenv install 3.11.13\") \\\n            .run_commands(\"~/.pyenv/versions/3.11.13/bin/python3.11 -m pip install pipx\") \\\n            .run_commands(\"~/.pyenv/versions/3.11.13/bin/python3.11 -m pipx ensurepath\") \\\n            .entrypoint([])\n\n        return image\n\n",
    ),
    (
        "startup_timeout: float = 0.4,",
        "startup_timeout: float = 1800.0,",
    ),
    (
        "return f\"{REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args}\"",
        "return f\"{REMOTE_EXECUTABLE_NAME} {rex_args} || ~/.pyenv/versions/3.11.13/bin/python3.11 -m pipx run {PACKAGE_NAME} {rex_args}\"",
    ),
];
const REMOTE_CHANGES: &[(&str, &str)] = &[
    (
        "    async def _request(self, endpoint: str, payload: BaseModel | None, output_class: Any, num_retries: int = 0):\n        \"\"\"Small helper to make requests to the server and handle errors and output.\"\"\"\n        request_url = f\"{self._api_url}/{endpoint}\"\n        request_id = str(uuid.uuid4())\n        headers = self._headers.copy()\n        headers[\"X-Request-ID\"] = request_id  # idempotency key for the request\n\n        retry_count = 0\n        last_exception: Exception | None = None\n        retry_delay = 0.1\n        backoff_max = 5\n\n        while retry_count <= num_retries:\n            try:\n                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:\n                    async with session.post(\n                        request_url,\n                        json=payload.model_dump() if payload else None,\n                        headers=headers,\n                    ) as resp:\n                        await self._handle_response_errors(resp)\n                        return output_class(**await resp.json())\n            except Exception as e:\n                last_exception = e\n                retry_count += 1\n                if retry_count <= num_retries:\n                    await asyncio.sleep(retry_delay)\n                    retry_delay *= 2\n                    retry_delay += random.uniform(0, 0.5)\n                    retry_delay = min(retry_delay, backoff_max)\n                    continue\n                self.logger.error(\"Error making request %s after %d retries: %s\", request_id, num_retries, e)\n        raise last_exception  # type: ignore\n\n",
        "    async def _request(self, endpoint: str, payload: BaseModel | None, output_class: Any, num_retries: int = 0):\n        \"\"\"Make a request, retrying only acquisition and transient HTTP status failures.\"\"\"\n        request_url = f\"{self._api_url}/{endpoint}\"\n        request_id = str(uuid.uuid4())\n        headers = self._headers.copy()\n        headers[\"X-Request-ID\"] = request_id  # idempotency key for the request\n\n        retry_count = 0\n        retry_delay = 0.1\n        backoff_max = 5\n        while retry_count <= num_retries:\n            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:\n                response = None\n                try:\n                    response = await session.post(\n                        request_url,\n                        json=payload.model_dump() if payload else None,\n                        headers=headers,\n                        timeout=aiohttp.ClientTimeout(total=self._get_timeout()),\n                    )\n                    if response.status in (408, 429) or (500 <= response.status < 600 and response.status != 511):\n                        response.raise_for_status()\n                except (aiohttp.ClientError, asyncio.TimeoutError) as error:\n                    if response is not None:\n                        response.release()\n                    if isinstance(error, aiohttp.ClientResponseError) and (\n                        error.status not in (408, 429) and not 500 <= error.status < 600\n                    ):\n                        raise\n                    if isinstance(error, aiohttp.ClientResponseError) and error.status == 511:\n                        raise\n                    retry_count += 1\n                    if retry_count > num_retries:\n                        self.logger.error(\"Error making request %s after %d retries: %s\", request_id, num_retries, error)\n                        raise\n                    await asyncio.sleep(retry_delay)\n                    retry_delay *= 2\n                    retry_delay += random.uniform(0, 0.5)\n                    retry_delay = min(retry_delay, backoff_max)\n                    continue\n\n                # Response processing is outside the transport retry catch. In particular,\n                # HTTP511 transferred TimeoutError/ClientError exceptions propagate once.\n                try:\n                    await self._handle_response_errors(response)\n                    return output_class(**await response.json())\n                finally:\n                    response.release()\n        raise RuntimeError(\"invalid negative retry count\")\n\n",
    ),
    (
        "return await self._request(\"create_session\", request, CreateSessionResponse)",
        "return await self._request(\"create_session\", request, CreateSessionResponse, num_retries=4)",
    ),
    (
        "return await self._request(\"close_session\", request, CloseSessionResponse)",
        "return await self._request(\"close_session\", request, CloseSessionResponse, num_retries=4)",
    ),
    (
        "return await self._request(\"read_file\", request, ReadFileResponse)",
        "return await self._request(\"read_file\", request, ReadFileResponse, num_retries=4)",
    ),
    (
        "return await self._request(\"close\", None, CloseResponse)",
        "return await self._request(\"close\", None, CloseResponse, num_retries=4)",
    ),
    (
        "f\"{self._api_url}/upload\", data=data, headers=self._headers\n                        ) as response:",
        "f\"{self._api_url}/upload\", data=data, headers=self._headers,\n                            timeout=aiohttp.ClientTimeout(total=self._get_timeout()),\n                        ) as response:",
    ),
    (
        "async with session.post(f\"{self._api_url}/upload\", data=data, headers=self._headers) as response:",
        "async with session.post(\n                        f\"{self._api_url}/upload\", data=data, headers=self._headers,\n                        timeout=aiohttp.ClientTimeout(total=self._get_timeout()),\n                    ) as response:",
    ),
];
pub(super) fn rewrite(destination: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    let mut text = std::str::from_utf8(bytes)?.to_owned();
    let edits = match destination {
        "deployment/modal.py" => MODAL_CHANGES,
        "deployment/config.py" => return Ok(bytes.to_vec()),
        "runtime/remote.py" => REMOTE_CHANGES,
        _ => bail!("unknown finite SWE-ReX Modal destination"),
    };
    for (original, replacement) in edits {
        if text.matches(original).count() != 1 {
            bail!("unknown or ambiguous official SWE-ReX patch anchor");
        }
        text = text.replacen(original, replacement, 1);
    }
    Ok(text.into_bytes())
}
