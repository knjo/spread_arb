"""Current frozen Q, full paired exits/carry, event S2 and no artificial CD."""
from pathlib import Path

from .. import study
from .event_s2_policy import EventS2Portfolio


study.CONFIGS=[
    dict(name='q_event20_deep',cap_twd=20_000_000,q_policy='net',reserve_quotes=False,
         release_credit=False,max_ticket_twd=None,deep_shared=True,event_s2=True),
    dict(name='q_event_release25_deep',cap_twd=25_000_000,q_policy='net',reserve_quotes=False,
         release_credit=True,max_ticket_twd=None,deep_shared=True,event_s2=True),
]
study.TargetPortfolio=EventS2Portfolio
base_source_paths=study.source_paths
base_write_json=study.write_json


def source_paths():
    folder=Path(__file__).resolve().parent
    return base_source_paths()+[folder/name for name in ('event_s2_study.py','event_s2_policy.py','deep_policy.py')]


def write_json(path,value):
    if path.name=='manifest.json' and 'configurations' in value:
        value['entry_execution_exception']=(
            'S2 evaluates idle symbols on every observable book event, effective entry cancellation, '
            'and immediately after the previous actual stock hedge completes. No fixed fill cooldown. '
            'A pending stock hedge blocks another S2 entry on that symbol, including after overnight restore. '
            'Every attempt uses current first-priority A1-minus-one-tick, depth5, residual25bp, prior Q and '
            'actual shared capacity. S1 below B1 can retain its queue without room; approach triggers '
            'a delayed cancellation. All race fills remain. Existing shadow and frozen Q/release inputs '
            'retain their original policy to isolate the currently requested Q-table execution change.')
    return base_write_json(path,value)


study.source_paths=source_paths
study.write_json=write_json


if __name__=='__main__':
    study.main()
