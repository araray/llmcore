//! Embedded-CPython async runtime.
//!
//! `llmcore` is async-first, so the binding owns a single asyncio event loop
//! running on a daemon thread for the life of the process. Rust drives Python
//! coroutines like this:
//!
//! 1. Under the GIL, build the coroutine and hand it to
//!    `asyncio.run_coroutine_threadsafe(coro, loop)`, which returns a
//!    `concurrent.futures.Future`.
//! 2. On a tokio blocking thread, call `future.result()`. That call blocks the
//!    thread but releases the GIL while waiting (CPython releases the GIL in the
//!    underlying condition wait), so the loop thread is free to actually run the
//!    coroutine. When it completes, `result()` re-acquires the GIL and returns.
//!
//! This keeps the async surface honest (`.await` in Rust) without reimplementing
//! asyncio↔tokio scheduling, and never holds the GIL while blocked.

use pyo3::prelude::*;

use crate::error::{EmbeddedError, Result};

/// Owns the embedded interpreter's event loop.
pub(crate) struct PyRuntime {
    /// The `asyncio` event loop object, already `run_forever()`-ing on a thread.
    loop_obj: Py<PyAny>,
}

const BOOTSTRAP: &str = r#"
import asyncio, threading

def _start_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(
        target=loop.run_forever,
        name="llmcore-embedded-loop",
        daemon=True,
    ).start()
    return loop
"#;

impl PyRuntime {
    /// Start the dedicated asyncio loop. Idempotent per instance.
    pub(crate) fn new() -> Result<Self> {
        Python::with_gil(|py| -> PyResult<Self> {
            let module = PyModule::from_code_bound(
                py,
                BOOTSTRAP,
                "llmcore_embedded_runtime.py",
                "llmcore_embedded_runtime",
            )?;
            let loop_obj = module.call_method0("_start_loop")?.unbind();
            Ok(Self { loop_obj })
        })
        .map_err(EmbeddedError::from)
    }

    /// Build a coroutine (under the GIL) and await its result from Rust.
    ///
    /// `make_coro` returns the Python coroutine object to schedule on the loop.
    pub(crate) async fn run_coro<F>(&self, make_coro: F) -> Result<Py<PyAny>>
    where
        F: FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
    {
        // Phase 1 — schedule the coroutine, get a concurrent.futures.Future.
        let loop_obj = Python::with_gil(|py| self.loop_obj.clone_ref(py));
        let conc_fut = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let coro = make_coro(py)?;
            let asyncio = py.import_bound("asyncio")?;
            let fut = asyncio
                .call_method1("run_coroutine_threadsafe", (coro, loop_obj.bind(py)))?;
            Ok(fut.unbind())
        })
        .map_err(EmbeddedError::from)?;

        // Phase 2 — block on .result() off the async worker (GIL released while
        // waiting), translating any Python exception via the bridge mapper.
        // `StopAsyncIteration` is detected by type (not message) so the streaming
        // pump can recognize a clean generator end.
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| match conc_fut.bind(py).call_method0("result") {
                Ok(v) => Ok(v.unbind()),
                Err(e)
                    if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) =>
                {
                    Err(EmbeddedError::stop_async())
                }
                Err(e) => Err(EmbeddedError::from_pyerr(py, &e)),
            })
        })
        .await
        .expect("llmcore-embedded: result task panicked")
    }
}
