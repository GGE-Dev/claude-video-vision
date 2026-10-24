import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Capture the OpenAI client construction and the transcription request so we
// can assert the Mistral backend points the SDK at Mistral's base URL and
// sends a Voxtral-compatible payload (no `language`, no `response_format`).
const ctorArgs: any[] = [];
const createArgs: any[] = [];
let fakeResponse: any = {};

vi.mock("openai", () => ({
  default: class FakeOpenAI {
    audio = {
      transcriptions: {
        create: async (params: any) => {
          createArgs.push(params);
          return fakeResponse;
        },
      },
    };
    constructor(opts: any) {
      ctorArgs.push(opts);
    }
  },
}));

vi.mock("fs", async () => {
  const actual = await vi.importActual<typeof import("fs")>("fs");
  return { ...actual, readFileSync: () => Buffer.from("RIFF") };
});

import { transcribeWithMistral, MISTRAL_BASE_URL, MISTRAL_DEFAULT_MODEL } from "../../src/backends/mistral.js";

describe("transcribeWithMistral", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    ctorArgs.length = 0;
    createArgs.length = 0;
    process.env.MISTRAL_API_KEY = "test-key";
    delete process.env.MISTRAL_TRANSCRIPTION_MODEL;
    fakeResponse = {
      text: "bonjour le monde",
      segments: [
        { start: 0, end: 1.5, text: " bonjour " },
        { start: 1.5, end: 3.2, text: "le monde" },
      ],
    };
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("throws a clear error when MISTRAL_API_KEY is missing", async () => {
    delete process.env.MISTRAL_API_KEY;
    await expect(transcribeWithMistral("/tmp/x.wav")).rejects.toThrow(/MISTRAL_API_KEY/);
  });

  it("points the OpenAI SDK at Mistral's base URL", async () => {
    await transcribeWithMistral("/tmp/x.wav");
    expect(ctorArgs[0]).toMatchObject({ apiKey: "test-key", baseURL: MISTRAL_BASE_URL });
  });

  it("uses voxtral-mini-latest by default and honours the model override", async () => {
    await transcribeWithMistral("/tmp/x.wav");
    expect(createArgs[0].model).toBe(MISTRAL_DEFAULT_MODEL);

    process.env.MISTRAL_TRANSCRIPTION_MODEL = "voxtral-custom";
    await transcribeWithMistral("/tmp/x.wav");
    expect(createArgs[1].model).toBe("voxtral-custom");
  });

  it("requests segment timestamps and omits params Mistral rejects", async () => {
    await transcribeWithMistral("/tmp/x.wav");
    const params = createArgs[0];
    expect(params.timestamp_granularities).toEqual(["segment"]);
    expect(params).not.toHaveProperty("language");
    expect(params).not.toHaveProperty("response_format");
  });

  it("maps segments to HH:MM:SS transcription entries", async () => {
    const result = await transcribeWithMistral("/tmp/x.wav");
    expect(result.backend).toBe("mistral");
    expect(result.transcription).toEqual([
      { start: "00:00:00", end: "00:00:01", text: "bonjour" },
      { start: "00:00:01", end: "00:00:03", text: "le monde" },
    ]);
    expect(result.audio_tags).toEqual([]);
    expect(result.full_analysis).toBeNull();
  });

  it("falls back to the full text when no segments are returned", async () => {
    fakeResponse = { text: "  juste du texte " };
    const result = await transcribeWithMistral("/tmp/x.wav");
    expect(result.transcription).toHaveLength(1);
    expect(result.transcription[0].text).toBe("juste du texte");
  });
});
