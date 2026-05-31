//! True async SSE streaming via a shared reqwest client (connection reuse + incremental TTFB).
//! Avoids the old blocking ureq path that called `into_string()` and buffered the entire
//! upstream response before the client saw the first byte.

use bytes::Bytes;
use futures::Stream;
use reqwest::Client;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tracing::{debug, error, trace};

pub struct SseLine {
    pub raw: String,
}

pub struct QwenSseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buf: String,
    ready: VecDeque<String>,
    finished: bool,
}

impl QwenSseStream {
    pub fn new(resp: reqwest::Response) -> Self {
        debug!(status = %resp.status(), "Qwen SSE stream opened (reqwest)");
        Self {
            inner: Box::pin(resp.bytes_stream()),
            buf: String::new(),
            ready: VecDeque::new(),
            finished: false,
        }
    }

    fn drain_lines(&mut self) {
        while let Some(pos) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=pos).collect();
            let line = line
                .trim_end_matches('\n')
                .trim_end_matches('\r')
                .to_string();
            if !line.is_empty() {
                self.ready.push_back(line);
            }
        }
    }
}

impl Stream for QwenSseStream {
    type Item = Result<SseLine, String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(line) = self.ready.pop_front() {
                return Poll::Ready(Some(Ok(SseLine { raw: line })));
            }
            if self.finished {
                if !self.buf.is_empty() {
                    let line = std::mem::take(&mut self.buf);
                    return Poll::Ready(Some(Ok(SseLine { raw: line })));
                }
                return Poll::Ready(None);
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.finished = true;
                }
                Poll::Ready(Some(Err(e))) => {
                    error!(error = %e, "Qwen SSE bytes_stream error");
                    return Poll::Ready(Some(Err(format!("Upstream stream error: {}", e))));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    trace!(len = bytes.len(), "SSE bytes chunk received");
                    self.buf.push_str(&String::from_utf8_lossy(&bytes));
                    self.drain_lines();
                }
            }
        }
    }
}

pub fn build_http_client() -> Result<Client, String> {
    // Build a custom rustls::ClientConfig using graviola (pure Rust) + webpki roots
    // This bypasses rustls-platform-verifier which panics on Android/Termux
    let mut root_store = rustls::RootCertStore::empty();
    for cert in webpki_root_certs::TLS_SERVER_ROOT_CERTS.iter() {
        root_store.add(cert.clone()).map_err(|e| format!("Failed to add root cert: {}", e))?;
    }
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(30))
        .timeout(Duration::from_secs(300))
        .tls_backend_preconfigured(tls_config)
        .build()
        .map_err(|e| format!("Failed to build reqwest client: {}", e))
}

/// Stream SSE lines from Qwen as they arrive (real streaming, not buffer-then-play).
/// Retries on 429 (rate limit) with exponential backoff.
pub async fn post_sse(
    client: Client,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
) -> Result<QwenSseStream, String> {
    let max_retries = 3;
    let mut retry_delay = Duration::from_secs(1);

    for attempt in 0..=max_retries {
        let mut req = client
            .post(&url)
            .header("accept", "text/event-stream")
            .body(body.clone());
        for (k, v) in &headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| format!("Qwen SSE request failed: {}", e))?;

        let status = resp.status();
        if status.is_success() {
            return Ok(QwenSseStream::new(resp));
        }

        let body_text = resp.text().await.unwrap_or_default();

        // Retry on 429 (rate limit) or "in progress" errors
        if (status.as_u16() == 429 || body_text.contains("in progress")) && attempt < max_retries {
            tracing::warn!(
                attempt = attempt + 1,
                max_retries = max_retries,
                delay_ms = retry_delay.as_millis(),
                "Rate limited, retrying..."
            );
            tokio::time::sleep(retry_delay).await;
            retry_delay *= 2;
            continue;
        }

        return Err(format!(
            "Qwen API returned {}: {}",
            status,
            body_text.chars().take(200).collect::<String>()
        ));
    }

    unreachable!()
}
