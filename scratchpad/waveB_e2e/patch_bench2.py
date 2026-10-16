import sys
from pathlib import Path

p = Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/e2e/bench_bridge.py"
s = p.read_text()
start = s.index("def transport(work: Path, repeats: int = 3)")
end = s.index("def startup_overlap(")
new = '''def _ping_row(nodelay: bool, rep: int, order: int, size: int, stats: dict, ok: bool) -> dict:
    return {
        "tcp_nodelay": nodelay,
        "payload_bytes": size,
        "repeat": rep,
        "order_in_pair": order,
        "rtt_ms": stats["rtt_s"] * 1e3,
        "client_encode_ms": stats["encode_s"] * 1e3,
        "client_decode_ms": stats["decode_s"] * 1e3,
        "server_ms": (stats["server"].get("compute_s") or 0.0) * 1e3,
        "request_bytes": stats["request_bytes"],
        "response_bytes": stats["response_bytes"],
        "round_trip_MBps": 2 * size / stats["rtt_s"] / 1e6,
        "bit_exact": ok,
    }


def transport(work: Path, repeats: int = 5) -> dict[str, Any]:
    """Echo RTTs with TCP_NODELAY on and off. Both servers stay up and every repeat pings both,
    alternating which goes first, so a busy shared node perturbs both variants alike."""

    sizes = [1 << 10, 16 << 10, 256 << 10, 4 << 20, 32 << 20, 128 << 20, 256 << 20]
    remotes = {nd: _remote(work / f"transport_nodelay_{int(nd)}", tcp_nodelay=nd)
               for nd in (True, False)}
    rows: list[dict[str, Any]] = []
    lora_rows: list[dict[str, Any]] = []
    load_before = os.getloadavg()
    try:
        for nd, remote in remotes.items():
            assert remote.wait_ready()["tcp_nodelay"] is nd
        gen = torch.Generator().manual_seed(0)
        for size in sizes:
            payload = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen)
            for rep in range(repeats):
                order = (True, False) if rep % 2 == 0 else (False, True)
                for k, nd in enumerate(order):
                    echoed, stats = remotes[nd].ping(payload)
                    rows.append(_ping_row(nd, rep, k, size, stats,
                                          bool(torch.equal(echoed, payload))))
        lora = _real_shape_lora()
        lora_bytes = sum(t.numel() * t.element_size() for t in lora.values())
        t0 = time.perf_counter()
        sha_before = lora_sha256(lora)
        sha_ms = (time.perf_counter() - t0) * 1e3
        for rep in range(repeats):
            order = (True, False) if rep % 2 == 0 else (False, True)
            for k, nd in enumerate(order):
                echoed, stats = remotes[nd].ping(lora)  # type: ignore[arg-type]
                t0 = time.perf_counter()
                sha_after = lora_sha256(echoed)
                sha_after_s = time.perf_counter() - t0
                lora_rows.append(
                    {
                        "tcp_nodelay": nd,
                        "repeat": rep,
                        "order_in_pair": k,
                        "num_tensors": len(lora),
                        "lora_bytes": lora_bytes,
                        "rtt_ms": stats["rtt_s"] * 1e3,
                        "client_encode_ms": stats["encode_s"] * 1e3,
                        "client_decode_ms": stats["decode_s"] * 1e3,
                        "server_ms": (stats["server"].get("compute_s") or 0.0) * 1e3,
                        "one_way_MBps": lora_bytes / (stats["rtt_s"] / 2) / 1e6,
                        "sha256_equal": sha_after == sha_before,
                        "lora_sha256_ms": sha_after_s * 1e3,
                    }
                )
    finally:
        for remote in remotes.values():
            remote.close()
    return {
        "note": "echo through the trainer server (megatron env, process launcher, CPU); one "
        "round trip = request + reply, each a torch.save blob over a localhost socket. "
        "tcp_nodelay=True is the production setting (bridge.set_nodelay on both ends); "
        "tcp_nodelay=False keeps Nagle on both ends (RemoteTrainer(tcp_nodelay=False)). Both "
        "servers run at once and every repeat pings both (order alternates per repeat). The "
        "node is shared with other agents: see load_average_1_5_15.",
        "repeats": repeats,
        "load_average_1_5_15": {"before": list(load_before), "after": list(os.getloadavg())},
        "payloads": rows,
        "real_lora_rank16": {
            "shape_source": "museglimmer_manifest dims, 52 layers x 8 projections x (A, B), bf16",
            "lora_sha256_ms_before_send": sha_ms,
            "rows": lora_rows,
        },
    }


'''
s = s[:start] + new + s[end:]

old = '''    if args.cpu:
        data["surrogate_exactness"] = {"cpu": surrogate_exactness("cpu")}'''
new2 = '''    if args.transport_only:
        data["transport"] = transport(work)
        data["transport"]["commit"] = data["commit"]
    if args.cpu:
        data["surrogate_exactness"] = {"cpu": surrogate_exactness("cpu")}'''
assert s.count(old) == 1
s = s.replace(old, new2)
old = '''        data["transport"] = transport(work)
        data["failure_handling"] = failure_handling(work / "faults")'''
new3 = '''        data["transport"] = transport(work)
        data["transport"]["commit"] = data["commit"]
        data["failure_handling"] = failure_handling(work / "faults")'''
assert s.count(old) == 1
s = s.replace(old, new3)
old = '''    parser.add_argument("--startup-overlap", action="store_true",'''
new4 = '''    parser.add_argument("--transport-only", action="store_true",
                        help="re-measure only the transport section (CPU)")
    parser.add_argument("--startup-overlap", action="store_true",'''
assert s.count(old) == 1
s = s.replace(old, new4)
old = '''    data.setdefault("env", {})[f"client_{mode}_run"] = _client_env()'''
new5 = '''    data.setdefault("env", {})[f"client_{mode}_run"] = _client_env()
    data["env"][f"client_{mode}_run"]["load_average_1_5_15_at_start"] = list(os.getloadavg())'''
assert s.count(old) == 1
s = s.replace(old, new5)
p.write_text(s)
print("ok")
