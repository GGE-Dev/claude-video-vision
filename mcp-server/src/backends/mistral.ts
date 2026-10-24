import { readFileSync } from "fs";
import type { AudioResult, TranscriptionSegment } from "../types.js";
import { formatHMS } from "../utils/timestamps.js";

export const MISTRAL_BASE_URL = "https://api.mistral.ai/v1";
export const MISTRAL_DEFAULT_MODEL = "voxtral-mini-latest";

/**
 * Transcribes audio with Mistral's Voxtral models.
 *
 * Mistral exposes an OpenAI-compatible `audio/transcriptions` endpoint, so we
 * reuse the OpenAI SDK pointed at Mistral's base URL. Differences from the
 * OpenAI backend:
 *   - auth via MISTRAL_API_KEY
 *   - model is `voxtral-mini-latest` by default (override with
 *     MISTRAL_TRANSCRIPTION_MODEL)
 *   - `language` must NOT be sent alongside `timestamp_granularities`
 *     (Mistral rejects the combination), so we rely on auto-detection
 *   - `response_format` is not part of Mistral's contract, so it is omitted;
 *     segments are returned by default when `timestamp_granularities` is set
 */
export async function transcribeWithMistral(wavPath: string): Promise<AudioResult> {
  const apiKey = process.env.MISTRAL_API_KEY;
  if (!apiKey) {
    throw new Error("MISTRAL_API_KEY environment variable is not set. Run video_setup to configure.");
  }

  const model = process.env.MISTRAL_TRANSCRIPTION_MODEL || MISTRAL_DEFAULT_MODEL;

  const OpenAI = (await import("openai")).default;
  const client = new OpenAI({ apiKey, baseURL: MISTRAL_BASE_URL });

  const audioFile = new File(
    [readFileSync(wavPath)],
    "audio.wav",
    { type: "audio/wav" },
  );

  const response = await client.audio.transcriptions.create({
    model,
    file: audioFile,
    timestamp_granularities: ["segment"],
  } as any);

  const segments: any[] = (response as any).segments || [];
  const transcription: TranscriptionSegment[] = segments.map((seg: any) => ({
    start: formatHMS(seg.start),
    end: formatHMS(seg.end),
    text: String(seg.text ?? "").trim(),
  }));

  // Some responses may omit segments (very short clips); fall back to the
  // full text as a single segment so Claude still sees the transcript.
  if (transcription.length === 0 && typeof (response as any).text === "string" && (response as any).text.trim()) {
    transcription.push({
      start: formatHMS(0),
      end: formatHMS(0),
      text: (response as any).text.trim(),
    });
  }

  return {
    backend: "mistral",
    transcription,
    audio_tags: [],
    full_analysis: null,
  };
}
