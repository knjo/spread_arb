"""Admission forecast is separate from the unchanged actual capacity ledger."""
from ..ev_lookup_cost.capacity_policy import activate_parked, admission_limit as legacy_limit


def admission_limit(actor, ns):
    method = getattr(actor, 'target_admission_limit', None)
    return method(ns) if method else legacy_limit(actor, ns)
