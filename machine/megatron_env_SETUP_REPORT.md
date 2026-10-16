# `~/envs/megatron` setup report — Megatron Bridge (main) for Muse-Glimmer-30B

Date: 2026-10-15 (UTC). Wall-clock: **13:15:27 → 13:37:12, ~22 minutes** (incl. ~1 min TE torch-binding compile, ~3.5 min fast-hadamard-transform compile; everything else was wheel installs from a warm uv cache).

Host: 8x A100-SXM4-80GB, driver 580.105, system nvcc 12.8 (`/usr/local/cuda`), gcc 11.4, Ubuntu 22.04 (glibc 2.35), uv 0.13.1, managed CPython 3.12.15.
Nothing under `/home/ubuntu/RL-Kernel` or `~/envs/vllm` was touched. GPU use: `CUDA_VISIBLE_DEVICES=7` only, for the smoke test below.

## Result: everything imports

```
torch 2.13.0+cu129 | cuda 12.9 | triton 3.7.1 | transformers 5.17.0
megatron.core      0.20.0+913159367  (editable: ~/src/Megatron-Bridge/3rdparty/Megatron-LM)
megatron.training  OK               (same editable install; needed by muse_glimmer)
transformer_engine 2.20.2           (transformer_engine.pytorch.Linear OK)
megatron.bridge    0.7.0+d86a5c09b  (editable: ~/src/Megatron-Bridge/src/megatron/bridge)
peft               0.21.0
flashinfer         0.6.18.post1     (flashinfer-python + flashinfer-cubin, from https://flashinfer.ai/whl/)
```

`uv pip check` → `Checked 240 packages ... All installed packages are compatible`.

### TE smoke test (GPU 7) — PASSED
`CUDA_VISIBLE_DEVICES=7`: `torch.cuda.is_available()` = True on `NVIDIA A100-SXM4-80GB`; `te.pytorch.Linear(64, 64, bias=True, params_dtype=bf16)` forward on a 64x64 bf16 tensor → `(64, 64) torch.bfloat16`, max abs diff vs `F.linear` = 0.0. `triton.__version__ == "3.7.1"` confirmed in the same process.

## Torch / Triton CUDA-build decision (read this)

- Requirement: `torch==2.13.0`, `triton==3.7.1` exactly (sibling `~/envs/vllm` has `torch 2.13.0+cu130`, `triton 3.7.1`, `transformers 5.17.0`, `flashinfer 0.7.0.post1`).
- PyPI default `torch==2.13.0` is a **CUDA 13.0** build (deps `nvidia-cuda-runtime==13.0.96`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`). TE's torch bindings are compiled locally with nvcc, and `torch.utils.cpp_extension` refuses a CUDA *major* mismatch (12.8 toolkit vs 13.0 torch), so a CUDA-12 torch was needed.
- The suggested `https://download.pytorch.org/whl/cu128` index has **no torch 2.13.0** (`No solution found`). 2.13.0 exists on cu126, cu129, cu130, cu132 only.
- **Chosen: `torch==2.13.0+cu129`** from `https://download.pytorch.org/whl/cu129` (CUDA 12.9, cuDNN 9.20.0.48 — same cuDNN as the cu130 build, NCCL 2.29.7, **triton 3.7.1**, identical to the vLLM env's triton). nvcc 12.8 vs torch's 12.9 is a minor-version mismatch, which cpp_extension only warns about; the TE bindings built and run fine. `torchvision 0.28.0+cu129` was installed from the same index so its CUDA major matches torch (PyPI torchvision is a cu130 build and would refuse to import).
- Note: Megatron-Bridge's `uv.lock` locks `torch 2.14.1 / triton 3.8.0`; its declared floor is `torch>=2.6.0`, so 2.13.0 is within spec.

## Megatron-Bridge / Megatron-LM commits

- `~/src/Megatron-Bridge` — branch `main`, commit **`d86a5c09bbff297132326812289693122240b12b`** ("build: bump MCore main to 91315936 (#6479)", 2026-10-15). Installed editable (`megatron-bridge 0.7.0+d86a5c09b`).
- Submodule `3rdparty/Megatron-LM` — commit **`9131593673d037d360d252904f9198d36bc0c246`** (describes as `25.04-alpha.rc1-4062-g913159367`; package version `megatron-core 0.20.0+913159367`). Installed editable. This is what Bridge main's `[tool.uv.sources] megatron-core = { path = "3rdparty/Megatron-LM", editable = true }` requires.
- TE: Bridge main pins `transformer-engine` to git rev `131157b5750b8578cb8efb78554f6fe6a671cc53` (metadata version `2.20.2+44f31bbe`). The **PyPI 2.20.2 release** was used instead (prebuilt `transformer-engine-cu12` core + locally compiled `transformer-engine-torch`), avoiding a ~30+ min core build. Same minor version; if the strict-mode hooks later need a TE source build, use that rev with `NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=80 MAX_JOBS=64`.

## `megatron.bridge.models.muse_glimmer` exports

`__all__`: `MuseGlimmerBridge, MuseGlimmerModel, MuseGlimmerModelBuilder, MuseGlimmerModelConfig, MuseGlimmerTransformerConfig, MuseGlimmerVisionModelConfig, MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerVisionConfig` (the last three re-exported from transformers 5.17.0, which has them).

`dir()` (public): `MuseGlimmerBridge, MuseGlimmerConfig, MuseGlimmerModel, MuseGlimmerModelBuilder, MuseGlimmerModelConfig, MuseGlimmerTextConfig, MuseGlimmerTransformerConfig, MuseGlimmerVisionConfig, MuseGlimmerVisionModelConfig, logger, logging, modeling_muse_glimmer, muse_glimmer_bridge, muse_glimmer_builder, muse_glimmer_config, transformers`

Classes defined per module:
- `muse_glimmer_bridge.py`: `MuseGlimmerBridge(MegatronModelBridge)`, `MuseGlimmerQKVGMapping(MegatronParamMapping[dict[str, torch.Tensor]])`
- `muse_glimmer_builder.py`: `MuseGlimmerModelBuilder(HybridModelBuilder)` (from `megatron.training.models.hybrid`)
- `muse_glimmer_config.py`: `MuseGlimmerTransformerConfig(TransformerConfig)`, `MuseGlimmerVisionModelConfig`, `MuseGlimmerModelConfig(ModelConfigOverrideMixin, HybridModelConfig)`
- `modeling_muse_glimmer.py`: `MuseGlimmerModel`

Also importable: `megatron.bridge.recipes.muse_glimmer` → `muse_glimmer_30b_peft_8gpu_h100_bf16_config`, `muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config`, `muse_glimmer_30b_sft_32gpu_h100_bf16_config`, `muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config` (+ `h100` subpackage). `megatron.bridge.training.{pretrain,train,nvrx_straggler}` import OK.

## Exact commands that worked (in order)

```bash
UV=~/.local/bin/uv; P=~/envs/megatron/bin/python
$UV venv ~/envs/megatron --python 3.12
git clone --branch main https://github.com/NVIDIA-NeMo/Megatron-Bridge ~/src/Megatron-Bridge
cd ~/src/Megatron-Bridge && git submodule update --init --recursive 3rdparty/Megatron-LM

# 1. torch (CUDA 12.9 build; see decision above)
$UV pip install --python $P torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129

# 2. build prereqs + TransformerEngine (prebuilt cu12 core, torch bindings compiled locally, ~70 s)
$UV pip install --python $P pip "setuptools<80" wheel pybind11 ninja cmake packaging numpy "pydantic<2.14"
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH MAX_JOBS=64 NVTE_BUILD_MAX_JOBS=64 \
NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=80 TORCH_CUDA_ARCH_LIST=8.0 \
  $UV pip install --python $P --no-build-isolation "transformer-engine[pytorch,core_cu12]==2.20.2"
# cleanup of the stray cu13 core (see failure #3) -- do NOT just `uv pip uninstall transformer-engine-cu13`:
$UV pip install --python $P --reinstall --no-deps "transformer-engine-cu12==2.20.2"

# constraints used for every later install (/tmp/.../constraints.txt):
#   torch==2.13.0+cu129  triton==3.7.1  transformers==5.17.0  transformer-engine==2.20.2
#   transformer-engine-torch==2.20.2  transformer-engine-cu12==2.20.2  nvidia-nccl-cu12==2.29.7
#   nvidia-cudnn-cu12==9.20.0.48  nvidia-cudnn-frontend==1.29.0  peft==0.21.0
C=constraints.txt

# 3. torchvision from the same CUDA index
$UV pip install --python $P -c $C torchvision --index-url https://download.pytorch.org/whl/cu129

# 4. megatron-core editable from the submodule ([mlm] extra only; [te] would pull the cu13 core, [dev] is uninstallable here)
$UV pip install --python $P -c $C -e "~/src/Megatron-Bridge/3rdparty/Megatron-LM[mlm]"

# 5. Megatron-Bridge runtime deps, explicitly (resolver would otherwise re-pull megatron-core[te] -> transformer-engine-cu13)
$UV pip install --python $P -c $C "transformers==5.17.0" "peft==0.21.0" "mistral-common>=1.10.0" "datasets>=2.20.0" \
  accelerate "diffusers>=0.36.0" einops "omegaconf>=2.3.0" "tensorboard>=2.19.0" typing-extensions rich "wandb>=0.29.0" \
  "six>=1.17.0" "regex>=2024.11.6" "pyyaml>=6.0.2" "tqdm>=4.67.1" "hydra-core>=1.3.4,<1.4" qwen-vl-utils \
  nvidia-resiliency-ext timm "open-clip-torch>=3.2.0" "mlflow>=3.16.0" "comet-ml>=3.50.0" safetensors huggingface_hub \
  webdataset pillow filelock "nvidia-modelopt[torch]==0.47.0" flash-linear-attention

# 6. Megatron-Bridge editable, no deps (deps handled above)
$UV pip install --python $P -c $C --no-deps --no-build-isolation -e ~/src/Megatron-Bridge

# 7. flashinfer at the Bridge pin, honoring its extra index
$UV pip install --python $P -c $C "flashinfer-python==0.6.18.post1" "flashinfer-cubin==0.6.18.post1" \
  --extra-index-url https://flashinfer.ai/whl/

# 8. leftovers found by `uv pip check` / recipe imports
CUDA_HOME=/usr/local/cuda MAX_JOBS=64 TORCH_CUDA_ARCH_LIST=8.0 $UV pip install --python $P -c $C --no-build-isolation \
  "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0"   # ~3.5 min
$UV pip install --python $P -c $C "megatron-energon~=7.0"
$UV pip install --python $P -c $C "s3fs<=2026.7.0" "fsspec[http]<=2026.7.0"
```

## Failures and workarounds (in the order hit)

1. **cu128 index has no torch 2.13.0** → used `cu129` (`torch 2.13.0+cu129`, triton 3.7.1). PyPI default is cu130 and the system toolkit is 12.8, so TE bindings could not have been compiled against it.
2. **`uv pip install -e` does not honor Megatron-Bridge's `[tool.uv]` sources/overrides** (megatron-core from the submodule, TE from git, fast-hadamard-transform from a git tag, `nvidia-cudnn-frontend==1.29.0`, `nvidia-modelopt==0.47.0`). Those were installed explicitly (steps 4, 5, 8) and the Bridge was installed `--no-deps` (said so, as requested). Also, megatron-core's `[te]` extra is `transformer-engine[pytorch,core_cu13]` — wrong for a CUDA-12 torch — so `[te]` was never used; TE was installed directly with `core_cu12`.
3. **TE install pulled `transformer-engine-cu13` in addition to `-cu12`**: uv's metadata-preparation pass of the `transformer-engine-torch` sdist declared `transformer-engine-cu13` while the actual build declared `-cu12`, so both cores landed. Removing cu13 with `uv pip uninstall transformer-engine-cu13` **deleted files shared with the cu12 wheel** (both write `transformer_engine/common/__init__.py`, `wheel_lib/libtransformer_engine.so`, ...), breaking `import transformer_engine`. Fix: `uv pip install --reinstall --no-deps transformer-engine-cu12==2.20.2`.
4. **Import-order gotcha (documented, not a bug in the env):** `import transformer_engine` *before* `import torch` fails with `libtorch_cuda.so: undefined symbol: ncclCommResume`. `libtransformer_engine.so` has `NEEDED libnccl.so.2`; loaded first, the dynamic linker binds it to the system `/usr/lib/x86_64-linux-gnu/libnccl.so.2.26.2`, and torch (which needs NCCL ≥2.28 symbols) then fails. Importing torch first binds everything to the pip `nvidia-nccl-cu12 2.29.7` and all is well; Megatron/Bridge always import torch first. If a stand-alone TE import is ever needed: `LD_LIBRARY_PATH=~/envs/megatron/lib/python3.12/site-packages/nvidia/nccl/lib`.
5. **`--no-build` batch failed**: `antlr4-python3-runtime` (hydra-core/omegaconf dep) is sdist-only → re-ran allowing builds (pure-python build, instant).
6. **`nvidia-resiliency-ext` resolves to 0.4.1, not 0.7.0**: 0.5.0+ ship only `manylinux_2_39` wheels; this host is glibc 2.35. 0.4.1 is the newest installable. `nvidia_resiliency_ext.inprocess.CallWrapper`, async-ckpt and straggler imports used by the Bridge all work with 0.4.1. This also makes megatron-core's `[dev]` extra uninstallable here (it pins `nvidia-resiliency-ext==0.7.0`), so `[dev]`/`[inference]` extras were skipped; the pieces the Bridge actually imports from them (`nvidia-modelopt[torch]==0.47.0`, `megatron-energon`, `flashinfer`, `fast-hadamard-transform`) were installed individually.
7. **`megatron.bridge.recipes` eagerly imports every recipe**; the Bagel recipe needs `megatron.energon` → installed `megatron-energon 7.4.1` (without the `[av_decode]` extra, matching the Bridge's `av` exclusion).
8. **`uv pip check` nits**: `fast-hadamard-transform>=1.1.0` missing (installed from the git tag the Bridge itself overrides to, since the PyPI sdist is incomplete); energon pulled `fsspec 2026.9.0` while `datasets 5.1.0` caps `<=2026.7.0`, and `s3fs` tracks fsspec release-for-release → pinned both to 2026.7.0. Check is now clean.
9. Side effects worth knowing: `omegaconf` 2.4.0 → 2.3.x (hydra-core<1.4 requirement); `nvidia-cudnn-frontend` 1.31.0 → 1.29.0 (Bridge override); `setuptools<80` installed because the Bridge's build-system says `setuptools<80` and it is built without isolation (Megatron-LM, which wants `setuptools>=80`, was built in isolation so this does not conflict).

## `uv pip list --python ~/envs/megatron/bin/python | grep -i -E "torch|triton|transformers|megatron|transformer.engine|peft|flashinfer|nvidia"`

```
flashinfer-cubin                         0.6.18.post1
flashinfer-python                        0.6.18.post1
megatron-bridge                          0.7.0+d86a5c09b  /home/ubuntu/src/Megatron-Bridge
megatron-core                            0.20.0+913159367 /home/ubuntu/src/Megatron-Bridge/3rdparty/Megatron-LM
megatron-energon                         7.4.1
nvidia-cublas-cu12                       12.9.1.4
nvidia-cuda-cupti-cu12                   12.9.79
nvidia-cuda-nvrtc-cu12                   12.9.86
nvidia-cuda-runtime-cu12                 12.9.79
nvidia-cudnn-cu12                        9.20.0.48
nvidia-cudnn-frontend                    1.29.0
nvidia-cufft-cu12                        11.4.1.4
nvidia-cufile-cu12                       1.14.1.1
nvidia-curand-cu12                       10.3.10.19
nvidia-cusolver-cu12                     11.7.5.82
nvidia-cusparse-cu12                     12.5.10.65
nvidia-cusparselt-cu12                   0.8.1
nvidia-cutlass-dsl                       4.8.0
nvidia-cutlass-dsl-libs-base             4.8.0
nvidia-cutlass-dsl-libs-core             4.8.0
nvidia-cutlass-dsl-libs-cu12             4.8.0
nvidia-cutlass-dsl-libs-cu13             4.8.0
nvidia-ml-py                             13.615.71
nvidia-modelopt                          0.47.0
nvidia-nccl-cu12                         2.29.7
nvidia-nvjitlink-cu12                    12.9.86
nvidia-nvshmem-cu12                      3.4.5
nvidia-nvtx-cu12                         12.9.79
nvidia-resiliency-ext                    0.4.1
open-clip-torch                          3.3.0
peft                                     0.21.0
torch                                    2.13.0+cu129
torchvision                              0.28.0+cu129
transformer-engine                       2.20.2
transformer-engine-cu12                  2.20.2
transformer-engine-torch                 2.20.2
transformers                             5.17.0
triton                                   3.7.1
```
(also: `fast-hadamard-transform 1.1.0`, `fsspec 2026.7.0`, `s3fs 2026.7.0`, `datasets 5.1.0`; 240 packages total, venv ≈17 GB.)

## Verification one-liner

```bash
CUDA_VISIBLE_DEVICES=7 ~/envs/megatron/bin/python -c "
import torch, triton, transformers, megatron.core, megatron.training, transformer_engine.pytorch as te, peft, flashinfer, megatron.bridge
import megatron.bridge.models.muse_glimmer as mg, megatron.bridge.recipes.muse_glimmer
assert torch.__version__.startswith('2.13.0') and triton.__version__=='3.7.1' and transformers.__version__=='5.17.0'
print(mg.__all__)"
```
