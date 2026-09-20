"""FlyMind Production Inference — Structured exceptions."""


class FlyMindError(Exception):
    """Base exception for all FlyMind inference errors."""
    pass


class ModelLoadError(FlyMindError):
    """Failed to load or validate the model artifact."""
    pass


class ModelArtifactMismatchError(FlyMindError):
    """Model artifact does not match expected contract."""
    pass


class FeatureContractError(FlyMindError):
    """Feature vector does not match the expected contract."""
    pass


class InvalidNeuronIDError(FlyMindError):
    """Neuron ID is invalid or not found in the dataset."""
    pass


class InferenceError(FlyMindError):
    """Model prediction failed."""
    pass


class CandidateLimitError(FlyMindError):
    """Requested candidate count exceeds allowed limit."""
    pass
