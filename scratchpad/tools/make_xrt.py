"""Build the cross-runtime token file: GSM8K chat prompts + gold answers (teacher forcing), plus one
long sequence that crosses the 2048 sliding window. Output: muse_glimmer_xrt_v1."""
import hashlib, json, sys
from pathlib import Path
from rl_engine.models.muse_glimmer.e2e.data import HFChatTokenizer, build_messages

out = Path(sys.argv[1])
rows = [json.loads(l) for l in open(Path("~/data/gsm8k/test.jsonl").expanduser()) if l.strip()]
tok = HFChatTokenizer()
seqs = []
for i, r in enumerate(rows[:8]):
    p = tok.encode_chat(build_messages(r["question"]))
    c = tok._tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
    seqs.append({"id": f"gsm8k_test_{i}", "token_ids": p + [int(t) for t in c], "completion_start": len(p)})
# long sequence: many problems+answers in one user turn, completion = one more gold answer; > 2048 tokens total
body, j = [], 8
while True:
    body.append("Q: " + rows[j]["question"] + "\nA: " + rows[j]["answer"])
    j += 1
    p = tok.encode_chat(build_messages("\n\n".join(body) + "\n\nQ: " + rows[j]["question"]))
    if len(p) >= 2100:
        break
c = tok._tokenizer(rows[j]["answer"], add_special_tokens=False)["input_ids"]
seqs.append({"id": "gsm8k_long_window", "token_ids": p + [int(t) for t in c], "completion_start": len(p)})
tj = Path("~/models/Muse-Glimmer-30B/tokenizer.json").expanduser().read_bytes()
doc = {
    "schema": "muse_glimmer_xrt_v1",
    "note": "Score log p(token_ids[i] | token_ids[:i]) for i in [completion_start, len). Write results as "
            "muse_glimmer_xrt_logprobs_v1: {schema, runtime, mode, tp, backend, sequences:[{id, logprobs_hex}]} where "
            "logprobs_hex[k] = '%08x' % fp32 bit pattern (uint32) of the logprob of token completion_start+k.",
    "checkpoint_revision": Path("~/models/Muse-Glimmer-30B/REVISION").expanduser().read_text().strip(),
    "tokenizer_json_sha256": hashlib.sha256(tj).hexdigest(),
    "sequences": seqs,
}
out.write_text(json.dumps(doc) + "\n")
print({s["id"]: (len(s["token_ids"]), s["completion_start"]) for s in seqs})
