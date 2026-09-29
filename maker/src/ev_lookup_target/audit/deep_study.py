"""S1 deep queue retention with observed-release admission and first-priority S2."""
from pathlib import Path

from .. import study
from .deep_policy import DeepSpotPortfolio


study.CONFIGS=[dict(name='q_net_release25_deep',cap_twd=25_000_000,q_policy='net',reserve_quotes=False,
                   release_credit=True,max_ticket_twd=None,deep_shared=True)]
study.TargetPortfolio=DeepSpotPortfolio
base_source_paths=study.source_paths
base_write_json=study.write_json


def source_paths():
    return base_source_paths()+[Path(__file__).resolve(),Path(__file__).with_name('deep_policy.py').resolve()]


study.source_paths=source_paths


def write_json(path,value):
    if path.name=='manifest.json' and 'configurations' in value:
        value['allocation_exceptions']={
            'deep_shared': 'The base allocation description has this explicit S1 exception: '
                'a buy strictly below the observable stock B1 may be sent and retained without individual room. '
                'On approach to B1, capacity shortage requests a 50ms cancellation. Existing queue priority '
                'survives natural capital releases. Irrevocable jump and cancellation-race fills remain booked.',
            'actual_overrun': 'Once committed capacity exceeds the configured 25M ceiling, request cancellation '
                'of all other unreserved entries immediately and pause new deep quotes until natural releases. '
                'This cannot undo same-print or pre-cancel race fills.',
            'S2': 'No exception: first-priority future quotes require current individual room.'}
    return base_write_json(path,value)


study.write_json=write_json


if __name__=='__main__':
    study.main()
