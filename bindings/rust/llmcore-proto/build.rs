//! Compile the llmcore.v1 protos into Rust (messages + tonic gRPC clients).
//!
//! Requires `protoc` on PATH (e.g. `apt install protobuf-compiler`,
//! `brew install protobuf`). The proto sources live at `bindings/proto`,
//! i.e. `../../proto` relative to this crate.
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = PathBuf::from("../../proto");
    let protos = [
        "llmcore/v1/common.proto",
        "llmcore/v1/errors.proto",
        "llmcore/v1/inference.proto",
        "llmcore/v1/catalog.proto",
        "llmcore/v1/control.proto",
        "llmcore/v1/audio.proto",
    ];
    let paths: Vec<PathBuf> = protos.iter().map(|p| proto_root.join(p)).collect();

    for p in &paths {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-changed={}", proto_root.display());

    // build_client(true): we generate clients only (the bridge is the server).
    // tonic-build 0.12 API: `compile_protos`. For <0.11 use `.compile(...)`.
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&paths, &[proto_root])?;
    Ok(())
}
