# LLMCore Examples

This directory contains runnable examples for the current LLMCore API.

Most examples use your configured default LLM provider. With the packaged
defaults that usually means Ollama, so local examples expect the Ollama service
and model to be available unless you override the provider in config.

## Offline Examples

These do not call a provider or external API:

- `model_cards_and_tokens.py` - bundled model cards, context windows, cost
  estimation, and token counting.

## Live Smoke Examples

These are the shortest examples to run against real credentials. They make
small calls and skip providers whose required environment is incomplete.

- `live_provider_smoke.py` - OpenAI/Gemini provider calls, plus Gemini streaming.
- `live_search_smoke.py` - Serper, SerpApi, Semantic Scholar, and Bright Data
  SERP when the required keys/zones are present.
- `live_audio_smoke.py` - Deepgram text-to-speech and speech-to-text.

## Core Chat Examples

- `simple_chat.py` - stateless chat plus an optional provider override.
- `streaming_chat.py` - streaming chat and saved-session verification.
- `session_chat.py` - multi-turn persistent sessions.
- `chat_with_usage.py` - per-call token usage with `ChatUsage`.

## RAG And External Context

- `rag_example.py` - internal vector-store RAG with direct similarity search.
- `search_to_chat_external_rag.py` - Semantic Scholar retrieval formatted as
  external context, then sent through `chat_with_usage`.

## Provider Examples

- `ollama_example.py` - local Ollama chat, streaming, sessions, and RAG.
- `gemini_example.py` - Gemini chat, streaming, sessions, and RAG.

## Search Provider Examples

These make live network calls. Some consume provider credits.

- `brightdata_search_example.py` - Bright Data SERP, scrape, discover, datasets.
- `serper_search_example.py` - Serper web search, verticals, batch, scrape.
- `serpapi_search_example.py` - SerpApi engines, async polling, batch, account.
- `semanticscholar_search_example.py` - keyless Semantic Scholar paper search,
  snippets, citation graph, recommendations, datasets.

## Deepgram Voice And Audio

These require:

```bash
pip install "llmcore[deepgram]"
export DEEPGRAM_API_KEY="dg_..."
```

- `deepgram_batch_stt.py`
- `deepgram_batch_tts.py`
- `deepgram_streaming_stt.py`
- `deepgram_streaming_tts.py`
- `deepgram_flux.py`
- `deepgram_text_intelligence.py`
- `deepgram_voice_agent.py`

## Useful Environment Variables

```bash
# Load a local env file without printing secrets
set -a
source /av/data/dbs/.env
set +a

# Local default provider
ollama pull llama3.2

# Hosted LLM providers
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Search providers
export BRIGHTDATA_API_TOKEN="..."
export BRIGHTDATA_SERP_ZONE="..."
export BRIGHTDATA_UNLOCKER_ZONE="..."
export SERPER_API_KEY="..."
export SERPAPI_API_KEY="..."
export SEMANTIC_SCHOLAR_API_KEY="..."  # optional
```

## Verification

Before running live examples, you can check that the scripts at least compile:

```bash
python -m compileall -q examples
```
