# Performance Optimization Notes

## Implemented

### Shared HTTP Client
- Added global reqwest client reuse
- Enabled HTTP/2 keepalive
- Reduced TCP/TLS handshake overhead
- Improved connection pooling

## Recommended Immediate Refactors

### Session State
Replace mutex-heavy session structures with:
- DashMap
- Atomic counters
- Smaller lock scopes

Avoid:
```rust
Arc<Mutex<HashMap<String, Session>>>
```

Prefer:
```rust
DashMap<String, Session>
```

### Streaming
Convert SSE forwarding to zero-copy passthrough.

Avoid:
- repeated serde_json conversions
- buffering entire streams
- unbounded channels

Prefer:
```rust
Body::wrap_stream(upstream.bytes_stream())
```

### Retry System
Add:
- exponential backoff
- randomized jitter
- account cooldown cache
- retry limits

### Build Optimizations
Recommended Cargo.toml release profile:

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
```

### Async Safety
Never hold mutexes across await points.

Bad:
```rust
let guard = state.lock().await;
network_call().await;
```

Good:
```rust
let value = {
    let guard = state.lock().await;
    guard.clone()
};
network_call(value).await;
```
