"""Independently reconstruct P_sd and lambda from prior completed-day outputs."""
from collections import defaultdict
import json

import polars as pl

from ..verify_full_study import read_rows


def verify_lookup_inputs(root):
    manifest=json.loads((root/"manifest.json").read_text())
    snapshots={(s["day"],s["portfolio"]):s for s in json.loads((root/"snapshots.json").read_text())}
    configs={c["name"]:c for c in manifest["configurations"]}
    configs["shadow"]={"cap_twd":10**12}
    psd=defaultdict(lambda:[0,0])
    rejected=defaultdict(list)
    observed_entries=set()
    counts=dict(psd_snapshots=0,lambda_snapshots=0,shadow_entries=0,cap_denials=0)
    for day in manifest["days"]:
        for name,config in configs.items():
            snap=snapshots[day,name]
            if dict(psd)!=snap["psd"]:
                raise AssertionError("P_sd does not equal the preceding completed entry-day outcomes")
            window=rejected[name][-5:]
            lam=min(15.,max(0.,sum(window)/len(window)/config["cap_twd"]*10_000)) if window else 0.
            if abs(lam-snap["lam_bp"])>1e-7:
                raise AssertionError("lambda does not equal preceding five sessions' rejected shadow-filled EV")
            counts["psd_snapshots"]+=1;counts["lambda_snapshots"]+=1
        positions=read_rows(root/f"Date={day}"/"shadow/positions.parquet")
        filled=set()
        for p in positions:
            if p["entry_day"]!=day or p["entry_fill_ns"] is None:
                continue
            if p["id"] in observed_entries:
                raise AssertionError("entry outcome counted twice")
            filled.add(p["id"]);observed_entries.add(p["id"])
            second=p["quote_second"]
            key=f'{p["stream"]}_{0 if second<3600 else 1 if second<9000 else 2}'
            psd[key][0]+=1;psd[key][1]+=int(p["close_day"]==day)
        for name in configs:
            rejected_ev=0.
            if name!="shadow":
                path=root/f"Date={day}"/name/"decisions.parquet"
                if path.exists():
                    cap=(pl.scan_parquet(path).filter(pl.col("reason")=="cap")
                         .select("intent_id","est_bp","reservation_cents").collect())
                    counts["cap_denials"]+=cap.height
                    last=cap.unique("intent_id",keep="last",maintain_order=True)
                    rejected_ev=sum(max(d["est_bp"],0.)*d["reservation_cents"]/1_000_000
                                    for d in last.iter_rows(named=True) if d["intent_id"] in filled)
            rejected[name].append(rejected_ev)
    state=json.loads((root/"checkpoint.json").read_text())
    for actor in state["actors"]:
        for day,total in zip(manifest["days"],rejected[actor["name"]],strict=True):
            if abs(actor["rejected_by_day"][day]-total)>max(1e-6,abs(total)*1e-12):
                raise AssertionError("checkpoint lambda history differs from saved quote decisions")
    if len(state["entry_days"])!=len(observed_entries):
        raise AssertionError("checkpoint entry history has missing or extra outcomes")
    counts["shadow_entries"]=len(observed_entries)
    return counts
