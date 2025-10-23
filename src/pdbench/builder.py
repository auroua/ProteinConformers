from src.pdbench.benchmark_methods.bioemu_sampler import BioemuSampler
from src.pdbench.benchmark_methods.esmdiff_sampler import ESMDiffSampler
from src.pdbench.benchmark_methods.alphaflow_sampler import AlphaFlowSampler


def get_sampler(
    sampler_type: str
):
    assert sampler_type in ["esmdiff", "alphaflow", 'bioemu'], \
        f"{sampler_type} is not a valid sampler type"

    if sampler_type == "esmdiff":
        return ESMDiffSampler
    elif sampler_type == "alphaflow":
        return AlphaFlowSampler
    elif sampler_type == "bioemu":
        return BioemuSampler