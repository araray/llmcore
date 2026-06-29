/**
 * Quickstart: drive the bridge over both transports.
 *
 *   LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
 *     --transport grpc,http --grpc-address 127.0.0.1:50151 \
 *     --http-address 127.0.0.1:50152 --insecure
 *
 *   npx tsx examples/quickstart.ts
 */
import { LlmcoreGrpcClient, LlmcoreHttpClient } from "../src/index";

const GRPC = process.env.LLMCORE_GRPC ?? "127.0.0.1:50151";
const HTTP = process.env.LLMCORE_HTTP ?? "http://127.0.0.1:50152";

async function main(): Promise<void> {
  // ---- gRPC ----
  const grpc = new LlmcoreGrpcClient(GRPC);
  const info = await grpc.ensureCompatible(["tier0"]);
  console.log(`[grpc] contract=${info.contractVersion} caps=${info.capabilities.join(",")}`);

  const chat = await grpc.chat({ message: "hello from grpc" });
  console.log(`[grpc] chat -> ${JSON.stringify(chat.text)} tokens=${chat.usage?.totalTokens}`);

  process.stdout.write("[grpc] stream -> ");
  for await (const chunk of grpc.chatStream({ message: "stream me" })) {
    if (!chunk.done) process.stdout.write(chunk.text);
  }
  process.stdout.write("\n");
  grpc.close();

  // ---- HTTP ----
  const http = new LlmcoreHttpClient(HTTP);
  await http.ensureCompatible();
  const hres = await http.chat({ message: "hello from http" });
  console.log(`[http] chat -> ${JSON.stringify(hres.text)}`);
  console.log(`[http] providers -> ${JSON.stringify((await http.listProviders()).providers)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
