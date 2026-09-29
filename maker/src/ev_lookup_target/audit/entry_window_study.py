"""Fixed entry-window target hurdle with deep S1 retention and first-priority S2."""
from pathlib import Path

from .. import study
from .entry_window import WindowPortfolio


study.CONFIGS=[dict(name='q_window_target_release25_deep',cap_twd=25_000_000,q_policy='net',reserve_quotes=False,
                   release_credit=True,max_ticket_twd=None,deep_shared=True,target_clock='entry_window')]
study.TargetPortfolio=WindowPortfolio
base_source_paths=study.source_paths
base_write_json=study.write_json


def source_paths():
    folder=Path(__file__).resolve().parent
    return base_source_paths()+[folder/name for name in ('entry_window_study.py','entry_window.py','deep_policy.py')]


def write_json(path,value):
    if path.name=='manifest.json' and 'configurations' in value:
        value['allocation_exceptions']={'deep_shared': 'S1 below observable B1 can retain its original queue '
            'without room; approach to B1 triggers a 50ms cancellation if room remains absent. '
            'S2 still requires room. Actual capacity above25M cancels all other entries and pauses new '
            'deep requests; direct-jump and cancellation-race fills remain booked.'}
        value['q_gate_exception']=('Net Q must cover 12bp per expected full 14000-second entry-window session '
            'occupied. Funding remains2% per calendar day. Prior-only competing risks predict this duration. '
            'The original calendar hurdle remains a diagnostic field, not this policy admission threshold. '
            'Actual utilization and entry supply are still needed to achieve the30% budget.')
    return base_write_json(path,value)


study.source_paths=source_paths
study.write_json=write_json


if __name__=='__main__':
    study.main()
