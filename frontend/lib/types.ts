export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  n_neurons: number | null;
  n_edges: number | null;
  version: string;
}

export interface ModelMetadata {
  model_type: string;
  n_estimators: number | null;
  feature_dim: number | null;
  node_feature_dim: number | null;
  n_features: number | null;
  feature_names: string[];
  n_classes: number | null;
  class_labels: number[];
  n_neurons: number | null;
  n_edges: number | null;
}

export interface PipelineStage {
  name: string;
  status: string;
  detail: string;
}

export interface PipelineStatus {
  stages: PipelineStage[];
}

export interface ExperimentResult {
  experiment: string;
  data: Record<string, unknown>;
}

export interface EvaluationResponse {
  experiments: ExperimentResult[];
}

export interface Neuron {
  root_id: number;
  nt_type: string | null;
  nt_type_score: number | null;
  primary_type: string | null;
  super_class: string | null;
  flow: string | null;
  coord_x: number | null;
  coord_y: number | null;
  coord_z: number | null;
  length_nm: number | null;
  area_nm: number | null;
  size_nm: number | null;
  side: string | null;
  name: string | null;
  outgoing_count: number | null;
  incoming_count: number | null;
}

export interface NeuronSearchItem {
  root_id: number;
  name: string | null;
  nt_type: string | null;
  super_class: string | null;
  primary_type?: string | null;
}

export interface NeuronSearchResponse {
  query: string;
  results: NeuronSearchItem[];
  total: number;
}

export interface PredictionResult {
  source_root_id: number;
  target_root_id: number;
  score: number;
  is_known_edge: boolean;
  is_self_loop: boolean;
}

export interface CandidateItem {
  rank: number;
  source_root_id: number;
  target_root_id: number;
  score: number;
  target_nt_type: string | null;
  target_super_class: string | null;
  target_primary_type: string | null;
}

export interface CandidatesResponse {
  source_root_id: number;
  candidates: CandidateItem[];
  total_sampled: number;
}

export interface ResearchSummary {
  title: string;
  description: string;
  dataset: Record<string, unknown>;
  experiments: Record<string, string>;
  key_findings: string[];
}
