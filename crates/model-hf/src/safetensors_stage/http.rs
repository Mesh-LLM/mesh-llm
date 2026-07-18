use std::{
    io::{Read, Write},
    ops::Range,
    path::{Component, Path},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, ensure};
use reqwest::{
    StatusCode, Url,
    blocking::{Client, RequestBuilder, Response},
    header::{
        ACCEPT_ENCODING, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, ETAG, IF_RANGE, RANGE,
    },
    redirect::Policy,
};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

#[derive(Clone)]
pub(crate) struct RemoteSource {
    client: Client,
    endpoint: Url,
    token: Option<String>,
}

pub(crate) struct ExactRangeResponse {
    response: Response,
    pub total_file_bytes: u64,
    expected_bytes: u64,
    etag: Option<String>,
}

pub(crate) struct RemoteFile {
    pub bytes: Vec<u8>,
    pub etag: Option<String>,
}

impl RemoteSource {
    pub fn new(endpoint: Option<&str>, token: Option<String>) -> Result<Self> {
        let endpoint = Url::parse(endpoint.unwrap_or(DEFAULT_ENDPOINT))
            .context("parse Hugging Face endpoint")?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .redirect(Policy::limited(10))
            .user_agent("mesh-llm-safetensors-stage/1")
            .build()
            .context("build SafeTensors stage HTTP client")?;
        Ok(Self {
            client,
            endpoint,
            token,
        })
    }

    pub fn endpoint(&self) -> &str {
        self.endpoint.as_str()
    }

    pub fn url(&self, repo: &str, revision: &str, file: &str) -> Result<Url> {
        validate_relative_file(file)?;
        let mut url = self.endpoint.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| anyhow!("Hugging Face endpoint cannot be a base URL"))?;
            segments.pop_if_empty();
            segments.extend(repo.split('/'));
            segments.push("resolve");
            segments.push(revision);
            segments.extend(file.split('/'));
        }
        Ok(url)
    }

    pub fn optional_small_file(&self, url: Url, max_bytes: u64) -> Result<Option<RemoteFile>> {
        let response = self
            .authorized(self.client.get(url))
            .send()
            .context("send Hugging Face metadata request")?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        self.read_small_response(response, max_bytes).map(Some)
    }

    pub fn small_file(&self, url: Url, max_bytes: u64) -> Result<RemoteFile> {
        let response = self
            .authorized(self.client.get(url))
            .send()
            .context("send Hugging Face metadata request")?;
        self.read_small_response(response, max_bytes)
    }

    pub fn exact_range(&self, url: Url, range: Range<u64>) -> Result<ExactRangeResponse> {
        self.exact_range_with_identity(url, range, None)
    }

    pub fn exact_range_if_range(
        &self,
        url: Url,
        range: Range<u64>,
        etag: &str,
    ) -> Result<ExactRangeResponse> {
        validate_strong_etag(etag)?;
        self.exact_range_with_identity(url, range, Some(etag))
    }

    fn exact_range_with_identity(
        &self,
        url: Url,
        range: Range<u64>,
        etag: Option<&str>,
    ) -> Result<ExactRangeResponse> {
        ensure!(range.start < range.end, "HTTP byte range must not be empty");
        let expected_bytes = range.end - range.start;
        let range_header = format!("bytes={}-{}", range.start, range.end - 1);
        let mut request = self
            .client
            .get(url)
            .header(RANGE, range_header.clone())
            .header(ACCEPT_ENCODING, "identity");
        if let Some(etag) = etag {
            request = request.header(IF_RANGE, etag);
        }
        let response = self
            .authorized(request)
            .send()
            .with_context(|| format!("request HTTP range {range_header}"))?;
        ensure!(
            response.status() == StatusCode::PARTIAL_CONTENT,
            "server did not honor {range_header}; status was {} (refusing a possible full-shard download)",
            response.status()
        );
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .context("206 response omitted Content-Range")?
            .to_str()
            .context("Content-Range is not valid ASCII")?;
        let parsed = parse_content_range(content_range)?;
        ensure!(
            parsed.start == range.start && parsed.end_exclusive == range.end,
            "Content-Range {content_range:?} did not match requested {range_header}"
        );
        if let Some(length) = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        {
            ensure!(
                length == expected_bytes,
                "HTTP range Content-Length was {length}, expected {expected_bytes}"
            );
        }
        let etag = header_string(&response, ETAG)?;
        if let Some(etag) = &etag {
            validate_strong_etag(etag)?;
        }
        Ok(ExactRangeResponse {
            response,
            total_file_bytes: parsed.total_file_bytes,
            expected_bytes,
            etag,
        })
    }

    fn read_small_response(&self, response: Response, max_bytes: u64) -> Result<RemoteFile> {
        let response = response
            .error_for_status()
            .context("download Hugging Face metadata file")?;
        let etag = header_string(&response, ETAG)?;
        if let Some(length) = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        {
            ensure!(
                length <= max_bytes,
                "metadata file is too large: {length} bytes"
            );
        }
        let limit = max_bytes
            .checked_add(1)
            .context("metadata byte limit overflow")?;
        let mut bytes = Vec::new();
        response
            .take(limit)
            .read_to_end(&mut bytes)
            .context("read metadata response")?;
        ensure!(
            bytes.len() as u64 <= max_bytes,
            "metadata response exceeded {max_bytes} bytes"
        );
        Ok(RemoteFile { bytes, etag })
    }

    fn authorized(&self, builder: RequestBuilder) -> RequestBuilder {
        match &self.token {
            Some(token) => builder.header(AUTHORIZATION, format!("Bearer {token}")),
            None => builder,
        }
    }
}

fn validate_strong_etag(etag: &str) -> Result<()> {
    ensure!(
        !etag.starts_with("W/") && etag.starts_with('"') && etag.ends_with('"'),
        "SafeTensors range response requires a strong quoted ETag"
    );
    Ok(())
}

impl ExactRangeResponse {
    pub fn etag(&self) -> Option<&str> {
        self.etag.as_deref()
    }

    pub fn into_bytes(mut self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(usize::try_from(self.expected_bytes)?);
        self.response
            .read_to_end(&mut bytes)
            .context("read HTTP range response")?;
        ensure!(
            bytes.len() as u64 == self.expected_bytes,
            "HTTP range returned {} bytes, expected {}",
            bytes.len(),
            self.expected_bytes
        );
        Ok(bytes)
    }

    pub fn copy_to(mut self, writer: &mut impl Write) -> Result<u64> {
        let written =
            std::io::copy(&mut self.response, writer).context("stream HTTP tensor range")?;
        ensure!(
            written == self.expected_bytes,
            "HTTP range returned {written} bytes, expected {}",
            self.expected_bytes
        );
        Ok(written)
    }
}

fn header_string(response: &Response, name: reqwest::header::HeaderName) -> Result<Option<String>> {
    response
        .headers()
        .get(name)
        .map(|value| {
            value
                .to_str()
                .context("HTTP identity header is not valid ASCII")
                .map(str::to_string)
        })
        .transpose()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParsedContentRange {
    start: u64,
    end_exclusive: u64,
    total_file_bytes: u64,
}

fn parse_content_range(value: &str) -> Result<ParsedContentRange> {
    let value = value
        .strip_prefix("bytes ")
        .context("Content-Range must use bytes")?;
    let (range, total) = value
        .split_once('/')
        .context("Content-Range omitted total size")?;
    let (start, end_inclusive) = range
        .split_once('-')
        .context("Content-Range omitted byte bounds")?;
    let start = start.parse::<u64>().context("parse Content-Range start")?;
    let end_inclusive = end_inclusive
        .parse::<u64>()
        .context("parse Content-Range end")?;
    let total_file_bytes = total.parse::<u64>().context("parse Content-Range total")?;
    let end_exclusive = end_inclusive
        .checked_add(1)
        .context("Content-Range end overflow")?;
    ensure!(start < end_exclusive, "Content-Range is empty");
    ensure!(
        end_exclusive <= total_file_bytes,
        "Content-Range exceeds total file size"
    );
    Ok(ParsedContentRange {
        start,
        end_exclusive,
        total_file_bytes,
    })
}

fn validate_relative_file(file: &str) -> Result<()> {
    ensure!(
        !file.is_empty()
            && !file
                .chars()
                .any(|character| matches!(character, '\\' | '?' | '#')),
        "unsafe Hugging Face repository file path {file:?}"
    );
    let path = Path::new(file);
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "unsafe Hugging Face repository file path {file:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use super::*;

    #[test]
    fn parses_exact_content_range() {
        assert_eq!(
            parse_content_range("bytes 8-15/100").unwrap(),
            ParsedContentRange {
                start: 8,
                end_exclusive: 16,
                total_file_bytes: 100,
            }
        );
    }

    #[test]
    fn rejects_unsafe_repository_paths() {
        assert!(validate_relative_file("../secret").is_err());
        assert!(validate_relative_file("/absolute").is_err());
        assert!(validate_relative_file("weights/model.safetensors").is_ok());
    }

    #[test]
    fn rejects_server_that_ignores_range() {
        let endpoint = serve_once(http_response("200 OK", &[], b"full"));
        let remote = RemoteSource::new(Some(&endpoint), None).unwrap();
        let url = remote
            .url("org/model", &"a".repeat(40), "model.safetensors")
            .unwrap();

        let error = remote.exact_range(url, 0..4).err().unwrap();

        assert!(error.to_string().contains("did not honor"));
    }

    #[test]
    fn rejects_mismatched_content_range() {
        let endpoint = serve_once(http_response(
            "206 Partial Content",
            &[("Content-Range", "bytes 1-4/10")],
            b"four",
        ));
        let remote = RemoteSource::new(Some(&endpoint), None).unwrap();
        let url = remote
            .url("org/model", &"a".repeat(40), "model.safetensors")
            .unwrap();

        let error = remote.exact_range(url, 0..4).err().unwrap();

        assert!(error.to_string().contains("did not match"));
    }

    #[test]
    fn rejects_truncated_range_body() {
        let response = b"HTTP/1.1 206 Partial Content\r\n\
            Content-Length: 4\r\n\
            Content-Range: bytes 0-3/10\r\n\
            Connection: close\r\n\r\nxx"
            .to_vec();
        let endpoint = serve_once(response);
        let remote = RemoteSource::new(Some(&endpoint), None).unwrap();
        let url = remote
            .url("org/model", &"a".repeat(40), "model.safetensors")
            .unwrap();

        let error = remote
            .exact_range(url, 0..4)
            .and_then(ExactRangeResponse::into_bytes)
            .unwrap_err();

        assert!(format!("{error:#}").contains("HTTP range"));
    }

    #[test]
    fn rejects_weak_etag_for_tensor_ranges() {
        let endpoint = serve_once(http_response(
            "206 Partial Content",
            &[("Content-Range", "bytes 0-3/10"), ("ETag", "W/\"weak\"")],
            b"four",
        ));
        let remote = RemoteSource::new(Some(&endpoint), None).unwrap();
        let url = remote
            .url("org/model", &"a".repeat(40), "model.safetensors")
            .unwrap();

        let error = remote.exact_range(url, 0..4).err().unwrap();

        assert!(error.to_string().contains("strong quoted ETag"));
    }

    fn serve_once(response: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request);
            let _ = stream.write_all(&response);
        });
        format!("http://{address}")
    }

    fn http_response(status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
        let headers = headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect()
    }
}
