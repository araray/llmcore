/**
 * Tier-1 demo: sessions, context items, vector store, and context presets.
 *
 * Start a bridge with the Tier-1 fake stores enabled, then run the example:
 *
 *   LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
 *     python -m llmcore.bridge.cli serve --transport grpc \
 *     --grpc-address 127.0.0.1:50151 --insecure
 *
 *   LLMCORE_GRPC=127.0.0.1:50151 npx tsx examples/sessions.ts
 */
import { LlmcoreGrpcClient } from "../src/index";

const GRPC = process.env.LLMCORE_GRPC ?? "127.0.0.1:50151";

async function main(): Promise<void> {
  const client = new LlmcoreGrpcClient(GRPC);
  try {
    const info = await client.getInfo();
    console.log(`contract=${info.contractVersion} tiers=${info.tiers.join(",")}`);

    // ---- sessions + context items ----
    const session = await client.createSession({
      name: "demo-session",
      systemMessage: "You are a terse assistant.",
    });
    console.log(`created session ${session.id} (${session.messages.length} message[s])`);

    const added = await client.addContextItem({
      sessionId: session.id,
      content: "Remember: the launch date is June 30.",
      type: "user_text",
    });
    const item = await client.getContextItem({ sessionId: session.id, itemId: added.itemId });
    console.log(`context item [${item.type}]: ${JSON.stringify(item.content)} (${item.tokens} tokens)`);

    // ---- vector store ----
    await client.addDocuments({
      documents: [{ content: "Paris is the capital of France.", metadata: { topic: "geography" } }],
    });
    const hits = await client.searchVectorStore({ query: "capital of France", k: 3 });
    for (const d of hits.documents) {
      console.log(`  hit (score=${d.score?.toFixed(3)}): ${JSON.stringify(d.content)}`);
    }

    // ---- context presets ----
    await client.saveContextPreset({
      preset: {
        name: "preamble",
        description: "Standard preamble",
        items: [{ type: "preset_text_content", content: "Always cite sources." }],
      },
    });
    const preset = await client.getContextPreset({ presetName: "preamble" });
    console.log(`preset ${JSON.stringify(preset.name)} has ${preset.items.length} item[s]`);

    // ---- cleanup ----
    await client.deleteSession({ sessionId: session.id });
    console.log("done.");
  } finally {
    client.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
