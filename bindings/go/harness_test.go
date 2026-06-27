package llmcore_test

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"time"

	llmcore "github.com/araray/llmcore-go"
)

// freePort returns an unused localhost TCP port.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// bridge is a spawned llmcore-bridge process (FakeFacade) for e2e tests.
type bridge struct {
	grpcAddr string
	httpBase string
	cmd      *exec.Cmd
}

// startBridge launches the bridge over localhost (gRPC + HTTP) with the
// deterministic fake backend and waits until it answers Health. The Python
// interpreter is $LLMCORE_BRIDGE_PYTHON (default "python3") and must have
// llmcore[bridge] importable. (Unix-only: uses SIGTERM for graceful stop.)
func startBridge(t *testing.T) *bridge { return startBridgeEnv(t) }

// startBridgeAudio is startBridge with the Tier-2 fake audio surface enabled
// (advertises tier2.audio + audio.*).
func startBridgeAudio(t *testing.T) *bridge {
	return startBridgeEnv(t, "LLMCORE_BRIDGE_FAKE_AUDIO=1")
}

func startBridgeEnv(t *testing.T, extraEnv ...string) *bridge {
	t.Helper()
	grpcPort := freePort(t)
	httpPort := freePort(t)
	b := &bridge{
		grpcAddr: fmt.Sprintf("127.0.0.1:%d", grpcPort),
		httpBase: fmt.Sprintf("http://127.0.0.1:%d", httpPort),
	}
	py := os.Getenv("LLMCORE_BRIDGE_PYTHON")
	if py == "" {
		py = "python3"
	}
	b.cmd = exec.Command(py,
		"-m", "llmcore.bridge.cli", "serve",
		"--transport", "grpc,http",
		"--grpc-address", b.grpcAddr,
		"--http-address", fmt.Sprintf("127.0.0.1:%d", httpPort),
		"--insecure", "--log-level", "WARNING",
	)
	b.cmd.Env = append(append(os.Environ(), "LLMCORE_BRIDGE_FAKE=1"), extraEnv...)
	if err := b.cmd.Start(); err != nil {
		t.Fatalf("start bridge (%s): %v", py, err)
	}
	if err := b.waitReady(); err != nil {
		_ = b.cmd.Process.Kill()
		t.Fatalf("bridge not ready: %v", err)
	}
	return b
}

func (b *bridge) waitReady() error {
	deadline := time.Now().Add(25 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		c, err := llmcore.Dial(b.grpcAddr)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			st, herr := c.Health(ctx)
			cancel()
			_ = c.Close()
			if herr == nil && st.GetOk() {
				return nil
			}
			lastErr = herr
		} else {
			lastErr = err
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for bridge: %v", lastErr)
}

// stop gracefully terminates the bridge (SIGTERM, then SIGKILL after a grace).
func (b *bridge) stop() {
	if b.cmd == nil || b.cmd.Process == nil {
		return
	}
	_ = b.cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan struct{})
	go func() { _ = b.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(8 * time.Second):
		_ = b.cmd.Process.Kill()
	}
}
