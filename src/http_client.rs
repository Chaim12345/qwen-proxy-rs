use once_cell::sync::Lazy;
use reqwest::Client;
use std::time::Duration;

/// Shared HTTP client reused across all upstream requests.
///
/// Benefits:
/// - Reuses TCP/TLS connections
/// - Enables HTTP/2 multiplexing
/// - Reduces allocation churn
/// - Prevents per-request client construction overhead
pub static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(32)
        .tcp_keepalive(Duration::from_secs(60))
        .http2_keep_alive_timeout(Duration::from_secs(30))
        .http2_keep_alive_interval(Duration::from_secs(15))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(300))
        .build()
        .expect("failed to build shared reqwest client")
});
