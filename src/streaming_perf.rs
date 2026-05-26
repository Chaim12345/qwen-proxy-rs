use bytes::Bytes;
use futures_util::Stream;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Optimized bounded streaming bridge.
///
/// Prevents:
/// - unbounded memory growth
/// - runaway producers
/// - stalled downstream consumers
///
/// Designed for SSE/OpenAI-compatible streaming.
pub fn bounded_sse_bridge(
    capacity: usize,
) -> (
    mpsc::Sender<Result<Bytes, std::io::Error>>,
    Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
) {
    let (tx, rx) = mpsc::channel(capacity);

    let stream = ReceiverStream::new(rx);

    (tx, Box::pin(stream))
}

/// Recommended default channel capacity for token streaming.
pub const DEFAULT_STREAM_BUFFER: usize = 32;

/// Small helper for avoiding oversized stream allocations.
#[inline]
pub fn should_flush(buffer_len: usize) -> bool {
    buffer_len >= 1024
}
