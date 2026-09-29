"""Isolate adaptive capacity from unreserved parking, retaining hard limits."""
from . import full_study


def main() -> None:
    full_study.CONFIGS = [
        dict(name=f"ev_bpday_adaptive_reserved_{cap}M", cap_twd=cap*1_000_000,
             use_ev=True, use_bpday=True, park_spot=False, overnight_target_twd=20_000_000)
        for cap in (25,30)
    ]
    full_study.main()


if __name__ == "__main__":
    main()
