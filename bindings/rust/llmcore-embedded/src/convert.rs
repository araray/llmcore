//! Marshal `llmcore` Python value objects into `llmcore.v1` prost types.
//!
//! These mirror the bridge's `_usage_to_proto` / `_cost_to_proto` /
//! `_model_details_to_proto` field-for-field, so the in-process binding returns
//! the same shapes as the gRPC path.

use llmcore_proto::v1 as pb;
use prost::Message;
use prost_types::{value::Kind, Struct, Value};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Marshal an `llmcore` value object to a prost message by delegating to a
/// bridge converter (e.g. `_chat_session_to_proto`) and decoding the serialized
/// proto it returns. This reuses the bridge's exact field mapping, so the
/// in-process result is byte-identical to the gRPC path.
pub(crate) fn via_converter<M: Message + Default>(
    py: Python<'_>,
    converter: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<M> {
    let core = py.import_bound("llmcore.bridge.core")?;
    let proto = core.call_method1(converter, (obj,))?;
    let bytes: Vec<u8> = proto.call_method0("SerializeToString")?.extract()?;
    M::decode(bytes.as_slice()).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// `google.protobuf.Value` → a native Python object.
fn value_to_py<'py>(py: Python<'py>, v: &Value) -> Bound<'py, PyAny> {
    match &v.kind {
        Some(Kind::NullValue(_)) | None => py.None().into_bound(py),
        Some(Kind::NumberValue(n)) => n.into_py(py).into_bound(py),
        Some(Kind::StringValue(s)) => s.into_py(py).into_bound(py),
        Some(Kind::BoolValue(b)) => b.into_py(py).into_bound(py),
        Some(Kind::StructValue(s)) => struct_to_pydict(py, s).into_any(),
        Some(Kind::ListValue(l)) => {
            let items = l.values.iter().map(|x| value_to_py(py, x));
            PyList::new_bound(py, items).into_any()
        }
    }
}

/// `google.protobuf.Struct` → a Python `dict` (used for `provider_kwargs`).
pub(crate) fn struct_to_pydict<'py>(py: Python<'py>, s: &Struct) -> Bound<'py, PyDict> {
    let d = PyDict::new_bound(py);
    for (k, v) in &s.fields {
        let _ = d.set_item(k, value_to_py(py, v));
    }
    d
}

/// Read an optional attribute, returning `None` when absent or Python-`None`.
fn opt<'py, T>(obj: &Bound<'py, PyAny>, name: &str) -> Option<T>
where
    T: FromPyObject<'py>,
{
    match obj.getattr(name) {
        Ok(v) if !v.is_none() => v.extract().ok(),
        _ => None,
    }
}

/// `ChatUsage` (dataclass) → `Usage` (all fields proto3-`optional`). Mirrors the
/// bridge: counts set when present, provider/model set only when truthy.
pub(crate) fn usage_to_proto(obj: &Bound<'_, PyAny>) -> pb::Usage {
    if obj.is_none() {
        return pb::Usage::default();
    }
    pb::Usage {
        prompt_tokens: opt::<i32>(obj, "prompt_tokens"),
        completion_tokens: opt::<i32>(obj, "completion_tokens"),
        total_tokens: opt::<i32>(obj, "total_tokens"),
        provider: opt::<String>(obj, "provider").filter(|s| !s.is_empty()),
        model: opt::<String>(obj, "model").filter(|s| !s.is_empty()),
    }
}

/// `CostEstimate` (pydantic) → `CostEstimate`.
pub(crate) fn cost_to_proto(o: &Bound<'_, PyAny>) -> pb::CostEstimate {
    pb::CostEstimate {
        input_cost: opt::<f64>(o, "input_cost").unwrap_or(0.0),
        output_cost: opt::<f64>(o, "output_cost").unwrap_or(0.0),
        cached_discount: opt::<f64>(o, "cached_discount").unwrap_or(0.0),
        reasoning_cost: opt::<f64>(o, "reasoning_cost").unwrap_or(0.0),
        total_cost: opt::<f64>(o, "total_cost").unwrap_or(0.0),
        currency: opt::<String>(o, "currency").unwrap_or_default(),
        pricing_source: opt::<String>(o, "pricing_source").unwrap_or_default(),
        prompt_tokens: opt::<i32>(o, "prompt_tokens").unwrap_or(0),
        completion_tokens: opt::<i32>(o, "completion_tokens").unwrap_or(0),
        cached_tokens: opt::<i32>(o, "cached_tokens").unwrap_or(0),
        reasoning_tokens: opt::<i32>(o, "reasoning_tokens").unwrap_or(0),
        input_price_per_million: opt::<f64>(o, "input_price_per_million"),
        output_price_per_million: opt::<f64>(o, "output_price_per_million"),
        cached_price_per_million: opt::<f64>(o, "cached_price_per_million"),
        model_id: opt::<String>(o, "model_id"),
        provider: opt::<String>(o, "provider"),
    }
}

/// `ModelDetails` (pydantic) → `ModelDetails`. (Metadata map is omitted: it maps
/// to a `Struct` and is not part of the Tier-0 parity surface.)
pub(crate) fn model_details_to_proto(m: &Bound<'_, PyAny>) -> pb::ModelDetails {
    pb::ModelDetails {
        id: opt::<String>(m, "id").unwrap_or_default(),
        provider_name: opt::<String>(m, "provider_name").unwrap_or_default(),
        context_length: opt::<i32>(m, "context_length").unwrap_or(0),
        supports_streaming: opt::<bool>(m, "supports_streaming").unwrap_or(false),
        supports_tools: opt::<bool>(m, "supports_tools").unwrap_or(false),
        supports_vision: opt::<bool>(m, "supports_vision").unwrap_or(false),
        supports_reasoning: opt::<bool>(m, "supports_reasoning").unwrap_or(false),
        display_name: opt::<String>(m, "display_name"),
        max_output_tokens: opt::<i32>(m, "max_output_tokens"),
        family: opt::<String>(m, "family"),
        parameter_count: opt::<String>(m, "parameter_count"),
        quantization_level: opt::<String>(m, "quantization_level"),
        file_size_bytes: opt::<i64>(m, "file_size_bytes"),
        model_type: opt::<String>(m, "model_type"),
        metadata: None,
    }
}
