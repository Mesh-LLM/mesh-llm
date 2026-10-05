use std::collections::VecDeque;

use futures::StreamExt;
use futures::stream::{self, Stream};
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;
use url::Url;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::retry;

struct PaginationState {
    buffer: VecDeque<serde_json::Value>,
    next_url: Option<Url>,
    is_first_page: bool,
    done: bool,
    items_yielded: usize,
    limit: Option<usize>,
}

impl HFClient {
    /// Create a paginated stream from an initial URL and query params.
    /// Query params are only sent on the first request; later pages
    /// use the full URL from the Link header.
    /// If `limit` is `Some(n)`, the stream stops after yielding `n` items.
    pub(crate) fn paginate<T: DeserializeOwned + 'static>(
        &self,
        initial_url: Url,
        params: Vec<(String, String)>,
        limit: Option<usize>,
    ) -> impl Stream<Item = HFResult<T>> + '_ {
        if limit == Some(0) {
            return futures::stream::empty().left_stream();
        }

        let state = PaginationState {
            buffer: VecDeque::new(),
            next_url: Some(initial_url),
            is_first_page: true,
            done: false,
            items_yielded: 0,
            limit,
        };

        stream::try_unfold(state, move |mut state| {
            let params = params.clone();
            async move {
                if state.limit.is_some_and(|max| state.items_yielded >= max) {
                    return Ok(None);
                }

                if let Some(raw) = state.buffer.pop_front() {
                    let item: T = serde_json::from_value(raw)?;
                    state.items_yielded += 1;
                    return Ok(Some((item, state)));
                }

                if state.done {
                    return Ok(None);
                }

                let url = match state.next_url.take() {
                    Some(u) => u,
                    None => return Ok(None),
                };

                let headers = self.auth_headers();
                let is_first_page = state.is_first_page;
                let response = retry::retry(self.retry_config(), || {
                    let mut req = self.http_client().get(url.clone()).headers(headers.clone());
                    if is_first_page {
                        req = req.query(&params);
                    }
                    req.send()
                })
                .await?;
                if state.is_first_page {
                    state.is_first_page = false;
                }

                if !response.status().is_success() {
                    let context =
                        Box::new(crate::error::HttpErrorContext::from_response(response).await);
                    return Err(HFError::Http { context });
                }

                state.next_url = parse_link_header_next(response.headers());
                if state.next_url.is_none() {
                    state.done = true;
                }

                let items: Vec<serde_json::Value> = response.json().await?;
                state.buffer = VecDeque::from(items);

                match state.buffer.pop_front() {
                    Some(raw) => {
                        let item: T = serde_json::from_value(raw)?;
                        state.items_yielded += 1;
                        Ok(Some((item, state)))
                    }
                    None => Ok(None),
                }
            }
        })
        .right_stream()
    }
}

/// Parse the `Link` header for a `rel="next"` URL.
/// Format: `<https://huggingface.co/api/models?p=1>; rel="next"`
fn parse_link_header_next(headers: &HeaderMap) -> Option<Url> {
    let link_header = headers.get("link")?.to_str().ok()?;

    for part in link_header.split(',') {
        let part = part.trim();
        if !part.contains("rel=\"next\"") {
            continue;
        }
        let start = part.find('<')? + 1;
        let end = part.find('>')?;
        let url_str = &part[start..end];
        return Url::parse(url_str).ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use reqwest::header::{HeaderMap, HeaderValue};

    use super::parse_link_header_next;

    #[test]
    fn test_parse_link_header_next() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "link",
            HeaderValue::from_static(r#"<https://huggingface.co/api/models?p=1>; rel="next""#),
        );
        let url = parse_link_header_next(&headers).unwrap();
        assert_eq!(url.as_str(), "https://huggingface.co/api/models?p=1");
    }

    #[test]
    fn test_parse_link_header_no_next() {
        let headers = HeaderMap::new();
        assert!(parse_link_header_next(&headers).is_none());
    }

    #[test]
    fn test_parse_link_header_multiple_rels() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "link",
            HeaderValue::from_static(
                r#"<https://example.com/prev>; rel="prev", <https://example.com/next>; rel="next""#,
            ),
        );
        let url = parse_link_header_next(&headers).unwrap();
        assert_eq!(url.as_str(), "https://example.com/next");
    }
}
