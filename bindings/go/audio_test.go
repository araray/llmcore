package llmcore_test

import (
	"bytes"
	"encoding/base64"
	"io"
	"testing"

	llmcore "github.com/araray/llmcore-go"
	llmcorev1 "github.com/araray/llmcore-go/gen/llmcore/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// dialAudioClient spins up an audio-enabled bridge + connected client.
func dialAudioClient(t *testing.T) (*llmcore.Client, func()) {
	t.Helper()
	b := startBridgeAudio(t)
	c, err := llmcore.Dial(b.grpcAddr)
	if err != nil {
		b.stop()
		t.Fatalf("dial: %v", err)
	}
	return c, func() { _ = c.Close(); b.stop() }
}

func mustSend(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("send: %v", err)
	}
}

func TestAudioCapabilitiesAdvertised(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	info, err := c.GetInfo(ctx)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"tier2.audio": true, "audio.transcribe_stream": true,
		"audio.synthesize_stream": true, "audio.voice_agent": true,
		"audio.synthesize": true, "audio.transcribe": true,
		"audio.generate_image": true, "audio.ocr": true, "audio.analyze_text": true,
	}
	have := map[string]bool{}
	for _, cap := range info.GetCapabilities() {
		have[cap] = true
	}
	for cap := range want {
		if !have[cap] {
			t.Fatalf("missing capability %q", cap)
		}
	}
}

// ---- live duplex ------------------------------------------------------ //

func TestTranscribeStream(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	stream, err := c.TranscribeStream(ctx)
	if err != nil {
		t.Fatal(err)
	}
	mustSend(t, stream.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Audio{Audio: []byte("hello")}}))
	mustSend(t, stream.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Audio{Audio: []byte("world")}}))
	mustSend(t, stream.Send(&llmcorev1.AudioIn{Frame: &llmcorev1.AudioIn_Control{Control: llmcorev1.SttControl_STT_CONTROL_CLOSE}}))
	mustSend(t, stream.CloseSend())

	var types []llmcorev1.StreamEventType
	var finalText string
	for {
		ev, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		types = append(types, ev.GetType())
		if ev.GetType() == llmcorev1.StreamEventType_STREAM_EVENT_TYPE_FINAL {
			finalText = ev.GetText()
		}
	}
	want := []llmcorev1.StreamEventType{
		llmcorev1.StreamEventType_STREAM_EVENT_TYPE_INTERIM,
		llmcorev1.StreamEventType_STREAM_EVENT_TYPE_INTERIM,
		llmcorev1.StreamEventType_STREAM_EVENT_TYPE_FINAL,
		llmcorev1.StreamEventType_STREAM_EVENT_TYPE_UTTERANCE_END,
	}
	if len(types) != len(want) {
		t.Fatalf("event types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("event[%d] = %v, want %v", i, types[i], want[i])
		}
	}
	if finalText != "hello world" {
		t.Fatalf("final text = %q", finalText)
	}
}

func TestSynthesizeStream(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	pieces := []string{"foo", "bar", "baz"}
	stream, err := c.SynthesizeStream(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pieces {
		mustSend(t, stream.Send(&llmcorev1.SynthControl{Frame: &llmcorev1.SynthControl_Text{Text: p}}))
	}
	mustSend(t, stream.Send(&llmcorev1.SynthControl{Frame: &llmcorev1.SynthControl_Control{Control: llmcorev1.TtsControl_TTS_CONTROL_CLOSE}}))
	mustSend(t, stream.CloseSend())

	var got []string
	var seqs []int64
	for {
		out, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, string(out.GetAudio()))
		seqs = append(seqs, out.GetSeq())
	}
	if len(got) != len(pieces) {
		t.Fatalf("chunks = %v, want %v", got, pieces)
	}
	for i := range pieces {
		if got[i] != pieces[i] {
			t.Fatalf("chunk[%d] = %q, want %q", i, got[i], pieces[i])
		}
		if seqs[i] != int64(i) {
			t.Fatalf("seq[%d] = %d, want %d", i, seqs[i], i)
		}
	}
}

func TestVoiceAgentDuplex(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	settings, err := structpb.NewStruct(map[string]interface{}{"provider_name": "fake"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := c.VoiceAgent(ctx)
	if err != nil {
		t.Fatal(err)
	}
	mustSend(t, stream.Send(&llmcorev1.VoiceAgentClientEvent{Event: &llmcorev1.VoiceAgentClientEvent_Settings{Settings: settings}}))
	mustSend(t, stream.Send(&llmcorev1.VoiceAgentClientEvent{Event: &llmcorev1.VoiceAgentClientEvent_InjectUserMessage{InjectUserMessage: "hi there"}}))
	mustSend(t, stream.Send(&llmcorev1.VoiceAgentClientEvent{Event: &llmcorev1.VoiceAgentClientEvent_Audio{Audio: []byte{1, 2}}}))
	mustSend(t, stream.CloseSend())

	var events []*llmcorev1.VoiceAgentEvent
	for {
		ev, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	if len(events) == 0 {
		t.Fatal("no events")
	}
	if events[0].GetType() != llmcorev1.VoiceAgentEventType_VOICE_AGENT_EVENT_TYPE_WELCOME {
		t.Fatalf("first event = %v, want WELCOME", events[0].GetType())
	}
	if last := events[len(events)-1]; last.GetType() != llmcorev1.VoiceAgentEventType_VOICE_AGENT_EVENT_TYPE_CLOSE {
		t.Fatalf("last event = %v, want CLOSE", last.GetType())
	}
	var sawConv, sawAudio bool
	for _, ev := range events {
		switch ev.GetType() {
		case llmcorev1.VoiceAgentEventType_VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT:
			if ev.GetRole() == "user" && ev.GetContent() == "hi there" {
				sawConv = true
			}
		case llmcorev1.VoiceAgentEventType_VOICE_AGENT_EVENT_TYPE_AUDIO:
			if bytes.Equal(ev.GetAudio(), append([]byte("agent:"), 1, 2)) {
				sawAudio = true
			}
		}
	}
	if !sawConv {
		t.Fatal("missing conversation-text (user, hi there)")
	}
	if !sawAudio {
		t.Fatal("missing agent audio")
	}
}

// ---- one-shot --------------------------------------------------------- //

func TestSynthesizeUnary(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	r, err := c.Synthesize(ctx, &llmcorev1.SynthesizeRequest{Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if string(r.GetAudioData()) != "tts:hello" {
		t.Fatalf("audio = %q", string(r.GetAudioData()))
	}
	if r.GetModel() != "fake-tts" {
		t.Fatalf("model = %q", r.GetModel())
	}
}

func TestTranscribeUnary(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	r, err := c.Transcribe(ctx, &llmcorev1.TranscribeRequest{AudioData: []byte("hello world")})
	if err != nil {
		t.Fatal(err)
	}
	if r.GetText() != "hello world" {
		t.Fatalf("text = %q", r.GetText())
	}
	if r.GetLanguage() != "en" {
		t.Fatalf("language = %q", r.GetLanguage())
	}
	if segs := r.GetSegments(); len(segs) != 1 || segs[0].GetSpeaker() != "spk_0" {
		t.Fatalf("segments = %v", segs)
	}
}

func TestGenerateImageUnary(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	r, err := c.GenerateImage(ctx, &llmcorev1.GenerateImageRequest{Prompt: "a cat", N: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.GetImages()) != 2 {
		t.Fatalf("images = %d, want 2", len(r.GetImages()))
	}
	data, err := base64.StdEncoding.DecodeString(r.GetImages()[0].GetData())
	if err != nil {
		t.Fatalf("image data not base64: %v", err)
	}
	if string(data) != "img:a cat" {
		t.Fatalf("image[0] = %q", string(data))
	}
}

func TestOCRUnary(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	r, err := c.OCR(ctx, &llmcorev1.OcrRequest{Source: &llmcorev1.OcrRequest_Data{Data: []byte("PDFBYTES")}})
	if err != nil {
		t.Fatal(err)
	}
	if r.GetModel() != "fake-ocr" {
		t.Fatalf("model = %q", r.GetModel())
	}
	if r.GetPagesProcessed() != 1 {
		t.Fatalf("pagesProcessed = %d", r.GetPagesProcessed())
	}
	if r.GetDocSizeBytes() != int64(len("PDFBYTES")) {
		t.Fatalf("docSizeBytes = %d", r.GetDocSizeBytes())
	}
	if len(r.GetPages()) != 1 {
		t.Fatalf("pages = %d, want 1", len(r.GetPages()))
	}
}

func TestAnalyzeTextUnary(t *testing.T) {
	c, cleanup := dialAudioClient(t)
	defer cleanup()
	ctx, cancel := testCtx()
	defer cancel()

	feat, err := structpb.NewStruct(map[string]interface{}{
		"summarize": true, "topics": true, "sentiment": true, "intents": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.AnalyzeText(ctx, &llmcorev1.AnalyzeTextRequest{Text: "some text", Features: feat})
	if err != nil {
		t.Fatal(err)
	}
	if r.GetSummary() != "summary:some text" {
		t.Fatalf("summary = %q", r.GetSummary())
	}
	if len(r.GetTopics()) == 0 {
		t.Fatal("expected topics")
	}
	if r.GetModel() != "fake-analyze" {
		t.Fatalf("model = %q", r.GetModel())
	}
}
