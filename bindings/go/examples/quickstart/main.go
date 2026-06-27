// Command quickstart drives the llmcore bridge over gRPC.
//
//	LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
//	  --transport grpc --grpc-address 127.0.0.1:50151 --insecure
//	go run ./examples/quickstart   # or set LLMCORE_GRPC=host:port
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	llmcore "github.com/araray/llmcore-go"
	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
)

func main() {
	target := os.Getenv("LLMCORE_GRPC")
	if target == "" {
		target = "127.0.0.1:50151"
	}
	c, err := llmcore.Dial(target)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	info, err := c.EnsureCompatible(ctx, "tier0")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("contract=%s caps=%v\n", info.GetContractVersion(), info.GetCapabilities())

	res, err := c.Chat(ctx, &llmcorev1.ChatRequest{Message: "hello from go"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("chat -> %q tokens=%d\n", res.GetText(), res.GetUsage().GetTotalTokens())

	stream, err := c.ChatStream(ctx, &llmcorev1.ChatRequest{Message: "stream me"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print("stream -> ")
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		if !chunk.GetDone() {
			fmt.Print(chunk.GetText())
		}
	}
	fmt.Println()
}
