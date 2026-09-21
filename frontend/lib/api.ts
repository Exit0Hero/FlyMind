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

const BASE_URL = "/api";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function getHealth(): Promise<HealthStatus> {
  return fetchJSON<HealthStatus>("/health");
}

export async function getModel(): Promise<ModelMetadata> {
  return fetchJSON<ModelMetadata>("/model");
}

export async function getPipeline(): Promise<PipelineStatus> {
  return fetchJSON<PipelineStatus>("/pipeline");
}

export async function getEvaluation(): Promise<EvaluationResponse> {
  return fetchJSON<EvaluationResponse>("/evaluation");
}

export async function getNeuron(rootId: number): Promise<Neuron> {
  return fetchJSON<Neuron>(`/neurons/${rootId}`);
}

export async function searchNeurons(q: string, limit = 20): Promise<NeuronSearchResponse> {
  return fetchJSON<NeuronSearchResponse>(`/neurons/search?q=${encodeURIComponent(q)}&limit=${limit}`);
}

export async function predictConnection(
  source_root_id: number,
  target_root_id: number
): Promise<PredictionResult> {
  return postJSON<PredictionResult>("/predict", { source_root_id, target_root_id });
}

export async function getCandidates(
  source_root_id: number,
  k = 10,
  candidate_pool_size = 1000
): Promise<CandidatesResponse> {
  return postJSON<CandidatesResponse>("/candidates", { source_root_id, k, candidate_pool_size });
}

export async function getResearchSummary(): Promise<ResearchSummary> {
  return fetchJSON<ResearchSummary>("/research/summary");
}
