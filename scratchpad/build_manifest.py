import json, sys, hashlib
sys.path.insert(0, sys.argv[2])
from rl_engine.models.muse_glimmer import workload as w
D = sys.argv[1]
cfg = json.load(open(f"{D}/config.json"))
idx = json.load(open(f"{D}/model.safetensors.index.json"))
shards = [
    {"filename": "model-00001-of-00002.safetensors", "sha256": "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694", "size_bytes": 49950112952},
    {"filename": "model-00002-of-00002.safetensors", "sha256": "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381", "size_bytes": 9603322320},
]
vc = cfg["vision_config"]
raw = {
  "version": "museglimmer-arch-fingerprint-v1",
  "workload_id": "museglimmer-30b-dense-text-v1",
  "model_identity": {
    "model_id": "meta-models/Muse-Glimmer-30B",
    "hf_repo": "meta-models/Muse-Glimmer-30B",
    "revision": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
    "architecture": cfg["architectures"][0],
    "model_type": cfg["model_type"],
    "density": "dense",
    "dtype": cfg["dtype"],
    "exit_forbids_architecture_shrink": True,
    "semantics_source": {
      "package": "transformers",
      "version": "5.17.0",
      "git_commit": "856157a2f3e9594954310df18fdccc31ffddebe9",
      "modeling_file": "src/transformers/models/muse_glimmer/modeling_muse_glimmer.py",
      "modeling_file_sha256": hashlib.sha256(open(f"{D}/modeling_muse_glimmer.py","rb").read()).hexdigest(),
      "checkpoint_transformers_version": cfg["transformers_version"],
    },
    "config_sha256": hashlib.sha256(open(f"{D}/config.json","rb").read()).hexdigest(),
    "config_fingerprint": {
      "text_config": cfg["text_config"],
      "multimodal": {
        "image_token_id": cfg["image_token_id"], "video_token_id": cfg["video_token_id"],
        "out_hidden_size": cfg["out_hidden_size"], "projector_hidden_act": cfg["projector_hidden_act"],
        "projector_hidden_size": cfg["projector_hidden_size"],
      },
      "vision_config_phase_b": {k: vc[k] for k in ("model_type","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","patch_size","patch_temporal","merge_size","hidden_act")},
    },
    "weight_snapshot": {
      "pin_method": "hf_revision_plus_index_sha256_plus_all_lfs_shard_sha256",
      "index_file": "model.safetensors.index.json",
      "index_sha256": hashlib.sha256(open(f"{D}/model.safetensors.index.json","rb").read()).hexdigest(),
      "total_parameters": idx["metadata"]["total_parameters"],
      "tensor_total_size_bytes": idx["metadata"]["total_size"],
      "weight_files_total_size_bytes": sum(s["size_bytes"] for s in shards),
      "text_tensor_count": len(w.text_weight_schema()),
      "text_tensor_dtype": "BF16",
      "content_hash_algorithm": "sha256-of-sorted-shard-records-v1",
      "content_hash": w.weight_snapshot_hash(shards),
      "shards": shards,
      "source": "HF paths-info LFS oid at the pinned revision; each value is the shard content SHA-256",
    },
  },
  "frozen_semantics": w.OFFICIAL_FROZEN_SEMANTICS,
  "projection_grouping": w.OFFICIAL_PROJECTION_GROUPING,
  "parallel_layout": w.OFFICIAL_PARALLEL_LAYOUT,
}
raw["fixture_identity_sha256"] = w.manifest_identity_hash(raw)
out = sys.argv[3]
with open(out, "w") as fh:
    json.dump(raw, fh, indent=2); fh.write("\n")
w.load_manifest(out); print("ok", raw["fixture_identity_sha256"])
