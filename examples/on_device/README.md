# On-device Deployment (AMD Ryzen AI PC)

Concrete recipes for getting a fine-tuned model running on the laptop you're assigned for the live demo.

The AMD fleet ships with: **FastFlowLM** (XDNA 2 NPU offload for LFM2-1.2B text), **llama.cpp + Vulkan**
(integrated Radeon GPU), and **ROCm** (the iGPU as a PyTorch device, required for LFM2.5-Audio). Pick the
path that matches your track:

| Track | Recommended runtime | Why |
|---|---|---|
| Text | llama.cpp + Vulkan (§2a) | Zero driver setup, GGUF quantization, runs on the iGPU |
| Text (LFM2-1.2B) | FastFlowLM NPU (§2b) | Offloads to the XDNA 2 NPU, frees the iGPU |
| Audio | ROCm + `liquid-audio` (§2c) | `liquid_audio`'s decoder requires a GPU torch backend; ROCm provides it on the iGPU |

## 1. Pull your fine-tune to the laptop

```powershell
# Windows + WSL2 or native Linux
pip install huggingface_hub
hf auth login
hf download your-username/your-finetune --local-dir ./model
```

Use the full merged checkpoint (LoRA-merged) so the on-device runtime doesn't need PEFT.

## 2a. Text via llama.cpp + Vulkan iGPU

Convert HF format → GGUF Q4_K_M once on the training machine (faster than the laptop):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
python convert_hf_to_gguf.py /path/to/model --outfile your-finetune.gguf
./build/bin/llama-quantize your-finetune.gguf your-finetune.Q4_K_M.gguf Q4_K_M
```

On the laptop, run with canonical LFM2 sampling:

```bash
./llama-cli \
  --model ./your-finetune.Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --temp 0.3 --min-p 0.15 --repeat-penalty 1.05 \
  --chat-template lfm2 \
  --interactive --color
```

`--n-gpu-layers 99` offloads everything to the iGPU via Vulkan. Drop to `0` for CPU-only.
(llama.cpp also has a ROCm/HIP backend via `-DGGML_HIP=ON`, but Vulkan needs no driver stack and
performs comparably on these iGPUs; stick with Vulkan for text unless you've already set up ROCm
for the audio path.)

## 2b. Text via FastFlowLM NPU (LFM2-1.2B only, XDNA 2 silicon)

FastFlowLM has a partner-built `.q4nx` quant of LFM2-1.2B with NPU offload. Strix Point / Strix Halo only;
Hawk Point (XDNA 1) does not qualify. Convert + run is FastFlowLM-specific; follow their CLI docs once you
have a 1.2B fine-tune.

## 2c. Audio via ROCm + `liquid-audio` (iGPU as the torch GPU)

`liquid_audio`'s audio decoder requires a GPU torch backend (its detokenizer calls `.cuda()`
unconditionally), so **plain CPU execution does not work**. On the AMD PCs the route is **ROCm**:
ROCm builds of PyTorch expose the iGPU through `torch.cuda` (the HIP backend masquerades as CUDA),
so `liquid_audio` (and the kit's CUDA gates) work unchanged.

**Hardware/OS prerequisites** (per [AMD's Strix Halo system guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)):

- Ryzen AI Max (Strix Halo, `gfx1151`) or Ryzen AI 300 (Strix Point, `gfx1150`). ROCm 7.2 detects both
  natively; on older ROCm stacks export `HSA_OVERRIDE_GFX_VERSION=11.0.0` as a workaround.
- Linux (Ubuntu 24.04 under WSL2 or native). Ryzen AI Max needs a recent kernel
  (Ubuntu 24.04 HWE `6.17.0-19.19~24.04.2`+, other distros `6.18.4`+) for the KFD driver fixes.
- BIOS: keep dedicated VRAM small (~0.5 GB) and let the model live in shared GTT memory.
  Use `pipx install amd-debug-tools && amd-ttm --set <GiB>` if the 3.6 GB checkpoint doesn't fit.

**Setup:** use a fresh venv, NOT the kit's uv-managed env (the kit pins CPU/CUDA torch wheels,
which would shadow the ROCm build):

```bash
# 1. ROCm 7.2 driver + runtime (skip if the fleet machine is pre-provisioned;
#    check with: rocminfo | grep gfx)
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb && sudo apt update
sudo amdgpu-install -y --usecase=graphics,rocm

# 2. Fresh Python 3.12 venv with ROCm torch (browse
#    https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ for the matching
#    torchaudio + triton wheels; install torch first so pip treats it as satisfied)
python3.12 -m venv ~/rocm-venv && source ~/rocm-venv/bin/activate
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
pip install <matching torchaudio wheel from the same index>
pip install liquid-audio

# 3. Verify the iGPU is the torch device ("CUDA" here is the HIP backend)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True AMD Radeon Graphics
```

**Synthesize a sample** with the programmatic TTS inference from
[`examples/audio/README.md` §5a](../audio/README.md) (point `CHECKPOINT` at `./model`); run it with
plain `python` from the ROCm venv, NOT `uv run`, which would re-sync the kit's CPU torch over your
ROCm install. It writes a WAV you can play with the OS audio player (`aplay` / `paplay`); pre-record
one (§3) to complement the live judge demo.

**Verify before demo day.** `gfx1151` is not yet on ROCm's official support matrix (it works in
practice; community guides report `torch.cuda.is_available() == True` and real workloads on
ROCm 7.2); wheel pairings (torch/torchaudio) move fast. Run the smoke above on the
actual fleet machine the day before, and have a pre-recorded WAV (§3) ready to complement the live demo.

## 3. Hardware tiers and demo recording

On-device latency on these AMD parts is **untested** for this kit, so don't trust any specific number
("real-time", TTFT, tokens/sec) quoted here or upstream, and measure on your assigned machine before
relying on live inference. Performance varies across devices, so **pre-record your demo to complement
your live one** regardless of tier.

| Hardware tier | NPU |
|---|---|
| **Strix Halo / Strix Point** | XDNA 2, 50 TOPS |
| **Krackan Point** | XDNA 2, 50 TOPS (smaller iGPU) |
| **Hawk Point** | XDNA 1, 16 TOPS |

To make the recording, run [`scripts/run_eval_audio.py`](../../scripts/run_eval_audio.py) on HF Jobs
(`l4x1`, ~$0.15) to generate the WAVs, download them, and play them back alongside the live demo.

## 4. LEAP SDK (for mobile / structured demos)

[LEAP](https://leap.liquid.ai) is Liquid AI's on-device SDK with bundled LFM2 GGUFs for iOS / Android /
desktop. If your demo benefits from a polished mobile shell rather than a CLI / Gradio app, build against
LEAP and load your fine-tune via its Hub-pull mechanism.
