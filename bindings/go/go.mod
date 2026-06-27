// NOTE: this module path is the single rename point. If you relocate the
// package, update (1) this line, (2) go_package_prefix in buf.gen.go.yaml, and
// (3) the imports in *.go. Then run `go mod tidy && make gen`.
module github.com/araray/llmcore-go

go 1.21

require (
	google.golang.org/grpc v1.65.0
	google.golang.org/protobuf v1.34.2
)
