/**
 * Typed client for the FlyMind backend.
 *
 * The Next.js rewrite proxy (`next.config.ts`) makes the API reachable at
 * `/api/*` same-origin in dev so the browser never needs to know where the
 * backend lives. `NEXT_PUBLIC_API_URL` overrides that (full absolute URL)
 * when a client must talk to a remote API directly.
 */

import type {
  HealthStatus,
  ModelMetadata,
  PipelineStatus,
  EvaluationResponse,
  Neuron,
  NeuronSearchResponse,
  PredictionResult,
  CandidatesResponse,
  ResearchSummary,
} from "./types";

const DIRECT_URL = process.env.NEXT_PUBLIC_API_URL;
export const API_BASE = DIRECT_URL ? DIRECT_URL.replace(/\/+$/, "") : "/api";

export const DEFAULT_TIMEOUT_MS = 45_000;
export const MAX_AUTOMATIC_RETRIES = 2; // idempotent (GET) requests only

interface RetryableOptions {
  timeoutMs?: number;
  retries?: number;
}

export class FlyMindApiError extends Error {
  readonly status: number | null;
  readonly code: string | null;
  readonly requestId: string | null;
  readonly payload: unknown;

  constructor(message: string, opts: { status?: number | null; code?: string | null; requestId?: string | null; payload?: unknown } = {}) {
    super(message);
    this.name = "FlyMindApiError";
    this.status = opts.status ?? null;
    this.code = opts.code ?? null;
    this.requestId = opts.requestId ?? null;
    this.payload = opts.payload ?? null;
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function request<T>(path: string, init: RequestInit, options: RetryableOptions = {}): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const retries = options.retries ?? MAX_AUTOMATIC_RETRIES;
  const idempotent = !init.method || init.method === "GET" || init.method === "HEAD" || init.method === "OPTIONS";

  let attempt = 0;
  while (true) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });

      if (!res.ok) {
        let error: FlyMindApiError;
        try {
          const body = (await res.json()) as unknown;
          const { code, message, request_id } = (isPlainObject(body) &&
            isPlainObject(body.error) &&
            body.error) as Record<string, unknown>;
          error = new FlyMindApiError(
            typeof message === "string" ? message : `API error ${res.status} ${res.statusText}`,
            {
              status: res.status,
              code: typeof code === "string" ? code : null,
              requestId: typeof request_id === "string" ? request_id : null,
              payload: body,
            },
          );
        } catch {
          error = new FlyMindApiError(`API error ${res.status} ${res.statusText}`, { status: res.status });
        }
        throw error;
      }
      return (await res.json()) as T;
    } catch (err) {
      if (idempotent && attempt < retries && err instanceof DOMException && err.name === "AbortError") {
        attempt += 1;
        continue;
      }
      if (idempotent && attempt < retries && err instanceof TypeError /* network failure */) {
        attempt += 1;
        continue;
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
}

function get<T>(path: string, options?: RetryableOptions): Promise<T> {
  return request<T>(path, { method: "GET", headers: { Accept: "application/json" } }, options);
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
}

/* ---- backend endpoints ------------------------------------------------- */

export async function getHealth(): Promise<HealthStatus> {
  return get<HealthStatus>("/health");
}

export async function getReady(): Promise<{ status: string }> {
  return get<{ status: string }>("/ready");
}

export async function getModel(): Promise<ModelMetadata> {
  return get<ModelMetadata>("/model");
}

export async function getPipeline(): Promise<PipelineStatus> {
  return get<PipelineStatus>("/pipeline");
}

export async function getEvaluation(): Promise<EvaluationResponse> {
  return get<EvaluationResponse>("/evaluation");
}

export async function predictConnection(
  sourceRootId: string,
  targetRootId: string,
): Promise<PredictionResult> {
  return post<PredictionResult>("/predict", {
    source_root_id: sourceRootId,
    target_root_id: targetRootId,
  });
}

export async function getCandidates(
  sourceRootId: string,
  k?: number,
  candidatePoolSize?: number,
): Promise<CandidatesResponse> {
  return post<CandidatesResponse>("/candidates", {
    source_root_id: sourceRootId,
    k,
    candidate_pool_size: candidatePoolSize,
  });
}

export async function getNeuron(rootId: string): Promise<Neuron> {
  return get<Neuron>("/neurons/" + rootId);
}

export async function searchNeurons(query: string, limit = 10): Promise<NeuronSearchResponse> {
  return get<NeuronSearchResponse>(`/neurons/search?q=${encodeURIComponent(query)}&limit=${limit}`);
}

export async function getResearchSummary(): Promise<ResearchSummary> {
  return get<ResearchSummary>("/research/summary");
}

export type { HealthStatus, ModelMetadata, PipelineStatus, EvaluationResponse, Neuron, NeuronSearchResponse, PredictionResult, CandidatesResponse, ResearchSummary };
