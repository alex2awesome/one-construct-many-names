"""Stratified Codability Profile.

Keep the public convenience API, but load it lazily. Experiment entry points import submodules
directly; eagerly importing every legacy analysis module here made unrelated code part of each
scoring process and defeated implementation-closure audits.
"""
from importlib import import_module


_EXPORTS = {
    "articulation_gaps": (".decompose", "articulation_gaps"),
    "attenuation_correct": (".decompose", "attenuation_correct"),
    "delta_context": (".decompose", "delta_context"),
    "mixed_model": (".decompose", "mixed_model"),
    "GATES": (".levels", "GATES"),
    "LEVELS": (".levels", "LEVELS"),
    "codability_map": (".levels", "codability_map"),
    "profile_level": (".levels", "profile_level"),
    "make_stratum_judge": (".strata", "make_stratum_judge"),
    "normalize_strata": (".strata", "normalize_strata"),
    "probe_balance_guard": (".strata", "probe_balance_guard"),
    "stratified_split": (".strata", "stratified_split"),
    "block_structure": (".transfer", "block_structure"),
    "kappa": (".transfer", "kappa"),
    "transfer_matrix": (".transfer", "transfer_matrix"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
