"""Ticket-size sensitivity under the same observable aggregate capacity gate.

Added after the shadow supply diagnostic revealed that the discretionary 2M
ticket filter removed material eligible supply. This is exploratory policy
selection; every runtime quote still uses strictly prior Q/release tables.
"""
from pathlib import Path

from .. import study


study.CONFIGS=[
    dict(name='q_net_shared20_anyticket',cap_twd=20_000_000,q_policy='net',reserve_quotes=False,max_ticket_twd=None),
    dict(name='q_net_release25_anyticket',cap_twd=25_000_000,q_policy='net',reserve_quotes=False,
         release_credit=True,max_ticket_twd=None),
    dict(name='q_target_release25_anyticket',cap_twd=25_000_000,q_policy='target',reserve_quotes=False,
         release_credit=True,max_ticket_twd=None),
]
base_source_paths=study.source_paths


def source_paths():
    return base_source_paths()+[Path(__file__).resolve()]


study.source_paths=source_paths


if __name__=='__main__':
    study.main()
