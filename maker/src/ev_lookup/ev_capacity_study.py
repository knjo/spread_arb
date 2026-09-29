"""EV-only capacity controls, separating bpday filtering from utilization."""
from . import full_study


def main() -> None:
    full_study.CONFIGS = [
        dict(name="ev_park_20M", cap_twd=20_000_000, use_ev=True, use_bpday=False, park_spot=True),
        dict(name="ev_adaptive_reserved_30M", cap_twd=30_000_000, use_ev=True, use_bpday=False,
             park_spot=False, overnight_target_twd=20_000_000),
        dict(name="ev_adaptive_30M", cap_twd=30_000_000, use_ev=True, use_bpday=False,
             park_spot=True, overnight_target_twd=20_000_000),
    ]
    full_study.main()


if __name__ == "__main__":
    main()
