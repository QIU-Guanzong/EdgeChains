import {
  ComprehendClient,
  DetectPiiEntitiesCommand,
  type DetectPiiEntitiesCommandOutput,
} from "@aws-sdk/client-comprehend";
import {
  Observable,
  concatMap,
  type ObservableInput,
  type OperatorFunction,
} from "rxjs";

export interface PiiClient {
  send(
    command: DetectPiiEntitiesCommand,
    options: { abortSignal: AbortSignal },
  ): Promise<DetectPiiEntitiesCommandOutput>;
}

export interface ComprehendRedactorOptions {
  client?: PiiClient;
  region?: string;
  languageCode?: "en" | "es";
  timeoutMs?: number;
}

export interface TextPrompt {
  prompt?: string;
  messages?: ReadonlyArray<{ content: string }>;
}

/** Only detected spans are removed; Comprehend can still miss sensitive data. */
export class ComprehendRedactor {
  private readonly client: PiiClient;
  private readonly languageCode: "en" | "es";
  private readonly timeoutMs: number;

  constructor(options: ComprehendRedactorOptions = {}) {
    this.languageCode = options.languageCode ?? "en";
    this.timeoutMs = options.timeoutMs ?? 30_000;
    if (!["en", "es"].includes(this.languageCode)) {
      throw new Error("PII detection supports English or Spanish input");
    }
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs < 1) {
      throw new Error("timeoutMs must be a positive safe integer");
    }
    // No automatic retries: repeated detection requests may incur charges.
    this.client =
      options.client ??
      new ComprehendClient({ region: options.region, maxAttempts: 1 });
  }

  private validateText(text: string): void {
    if (typeof text !== "string")
      throw new Error("PII input must be plain text");
    if (Buffer.byteLength(text, "utf8") > 100_000) {
      throw new Error("PII input exceeds the 100 KB UTF-8 limit");
    }
    // Do not let JSON/UTF-8 replacement of lone surrogates change offsets.
    if (
      /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(
        text,
      )
    ) {
      throw new Error("PII input contains an unpaired Unicode surrogate");
    }
  }

  async redact(text: string, signal?: AbortSignal): Promise<string> {
    this.validateText(text);
    if (signal?.aborted) throw new Error("PII detection cancelled");
    if (!text) return "";
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(abort, this.timeoutMs);
    let stopWaiting: () => void = () => {};
    try {
      // Also bound a custom client's lifetime if it ignores abortSignal.
      const cancelled = new Promise<never>((_, reject) => {
        stopWaiting = () =>
          reject(new Error("PII detection cancelled or timed out"));
        controller.signal.addEventListener("abort", stopWaiting, {
          once: true,
        });
      });
      let response: DetectPiiEntitiesCommandOutput;
      try {
        response = await Promise.race([
          this.client.send(
            new DetectPiiEntitiesCommand({
              Text: text,
              LanguageCode: this.languageCode,
            }),
            { abortSignal: controller.signal },
          ),
          cancelled,
        ]);
      } catch {
        // SDK errors may contain request text. Never forward or log them.
        throw new Error("PII detection failed; no text was forwarded");
      }
      if (controller.signal.aborted)
        throw new Error("PII detection cancelled or timed out");
      if (!response || !Array.isArray(response.Entities)) {
        throw new Error("PII detection returned an invalid entity list");
      }
      const characters = Array.from(text);
      const spans = response.Entities.map((entity) => {
        const begin = entity?.BeginOffset;
        const end = entity?.EndOffset;
        if (
          !Number.isSafeInteger(begin) ||
          !Number.isSafeInteger(end) ||
          begin! < 0 ||
          end! <= begin! ||
          end! > characters.length
        ) {
          throw new Error("PII detection returned an invalid character range");
        }
        return { begin: begin!, end: end! };
      }).sort((a, b) => a.begin - b.begin || a.end - b.end);

      const merged: Array<{ begin: number; end: number }> = [];
      for (const span of spans) {
        const previous = merged[merged.length - 1];
        if (previous && span.begin <= previous.end)
          previous.end = Math.max(previous.end, span.end);
        else merged.push({ ...span });
      }
      const pieces: string[] = [];
      let cursor = 0;
      for (const span of merged) {
        pieces.push(
          characters.slice(cursor, span.begin).join(""),
          "[REDACTED]",
        );
        cursor = span.end;
      }
      pieces.push(characters.slice(cursor).join(""));
      return pieces.join("");
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      controller.signal.removeEventListener("abort", stopWaiting);
    }
  }

  /** A cold Observable; unsubscribing aborts the in-flight detection request. */
  redact$(text: string): Observable<string> {
    return this.abortable((signal) => this.redact(text, signal));
  }

  /** Sequential redaction preserves source order and avoids an unbounded burst. */
  redactOperator(): OperatorFunction<string, string> {
    return concatMap((text) => this.redact$(text));
  }

  /** Redact all supported text before invoking an existing chat endpoint. */
  protect<T extends TextPrompt, R>(
    endpoint: (options: T) => ObservableInput<R>,
  ): OperatorFunction<T, R> {
    return concatMap((options) =>
      this.abortable(async (signal) => {
        if (!options || typeof options !== "object")
          throw new Error("A text prompt is required");
        if (options.prompt === undefined && options.messages === undefined) {
          throw new Error("A prompt or messages array is required");
        }
        if (options.prompt !== undefined) this.validateText(options.prompt);
        if (options.messages !== undefined) {
          if (!Array.isArray(options.messages))
            throw new Error("messages must be an array");
          for (const message of options.messages)
            this.validateText(message?.content);
        }
        const result = { ...options };
        if (options.prompt !== undefined)
          result.prompt = await this.redact(options.prompt, signal);
        if (options.messages !== undefined) {
          const messages: Array<{ content: string }> = [];
          for (const message of options.messages) {
            messages.push({
              ...message,
              content: await this.redact(message.content, signal),
            });
          }
          result.messages = messages;
        }
        return result;
      }).pipe(concatMap(endpoint)),
    );
  }

  private abortable<T>(
    work: (signal: AbortSignal) => Promise<T>,
  ): Observable<T> {
    return new Observable((subscriber) => {
      const controller = new AbortController();
      work(controller.signal).then(
        (value) => {
          subscriber.next(value);
          subscriber.complete();
        },
        (error) => subscriber.error(error),
      );
      return () => controller.abort();
    });
  }
}
