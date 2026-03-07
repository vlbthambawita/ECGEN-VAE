## How Stage 1 Passes the Checkpoint to Stage 2

This project trains the ECG VQ-VAE in **two stages**:

- **Stage 1**: Train the VQ-VAE (encoder + vector quantizer + decoder).
- **Stage 2**: Train the PixelCNN prior over the discrete VQ-VAE codes.

This document explains how the **best Stage 1 checkpoint** is produced and then **automatically passed** into Stage 2 when you use `run_train_vqvae_g002_full_v2.sh`.

---

## 1. Where Stage 1 Saves Its Best Checkpoint

Stage 1 is implemented in `train_vqvae_standalone.py` via `train_stage1_vqvae(args)`.

- A `ModelCheckpoint` callback writes checkpoints into:
  - `runs/<EXP_NAME_STAGE1>/seed_<SEED>/checkpoints/`
- A `BestCheckpointCopyCallback` maintains a convenient copy named:
  - `runs/<EXP_NAME_STAGE1>/seed_<SEED>/checkpoints/best.ckpt`

At the end of Stage 1, the script prints something like:

```text
VQ-VAE training (Stage 1) finished.
Best checkpoint: runs/<EXP_NAME_STAGE1>/seed_<SEED>/checkpoints/epochXXX-stepYYYYYY.ckpt
Best checkpoint (copy): runs/<EXP_NAME_STAGE1>/seed_<SEED>/checkpoints/best.ckpt
```

So after Stage 1, the **canonical checkpoint path** is:

```text
runs/<EXP_NAME_STAGE1>/seed_<SEED>/checkpoints/best.ckpt
```

---

## 2. How the Shell Script Finds the Best Stage 1 Checkpoint

When you run:

```bash
./run_train_vqvae_g002_full_v2.sh both
```

the script first executes Stage 1 and, after it finishes, runs this logic:

1. Construct the checkpoint directory:

```bash
CHECKPOINT_DIR="${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints"
```

2. Prefer `best.ckpt` if it exists:

```bash
if [ -f "${CHECKPOINT_DIR}/best.ckpt" ]; then
    BEST_CHECKPOINT="${CHECKPOINT_DIR}/best.ckpt"
else
    BEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "epoch*.ckpt" -type f | sort | tail -n 1)
    if [ -z "${BEST_CHECKPOINT}" ]; then
        BEST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
    fi
fi
```

3. Print and **export** the selected checkpoint:

```bash
print_info "Best checkpoint: ${BEST_CHECKPOINT}"
export VQVAE_CHECKPOINT="${BEST_CHECKPOINT}"
```

Key point: **`VQVAE_CHECKPOINT` is an environment variable** that now holds the full path to the chosen Stage 1 checkpoint.

---

## 3. How Stage 2 Receives the Stage 1 Checkpoint

In the same script, the Stage 2 block runs whenever:

- `STAGE == "2"` or
- `STAGE == "both"`.

Before launching Python, it validates that the environment variable is set and the file exists:

```bash
if [ -z "${VQVAE_CHECKPOINT:-}" ]; then
    # print error and exit with instructions
fi

if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
    # print error and exit
fi
```

Then Stage 2 is launched with `--vqvae-checkpoint` pointing to that file:

```bash
python train_vqvae_standalone.py \
    --stage 2 \
    --exp-name ${EXP_NAME_STAGE2} \
    ... \
    --vqvae-checkpoint ${VQVAE_CHECKPOINT}
```

So when you run `./run_train_vqvae_g002_full_v2.sh both`, Stage 1 finishes, the script **exports `VQVAE_CHECKPOINT`**, and Stage 2 is immediately started using that same path.

---

## 4. How the Python Stage 2 Code Uses the Checkpoint

Inside `train_vqvae_standalone.py`, Stage 2 is implemented in `train_stage2_prior(args)`:

- The CLI argument `--vqvae-checkpoint` becomes `args.vqvae_checkpoint`.
- A `PriorConfig` instance is created with `vqvae_checkpoint=args.vqvae_checkpoint`.
- `PriorLightning` receives this config and, if `config.vqvae_checkpoint` is non-empty, calls:

```python
VQVAELightning.load_from_checkpoint(checkpoint_path)
```

The loaded VQ-VAE is set to `eval()` and `freeze()`, and is then used to:

- Encode ECG inputs to discrete code indices.
- Provide those indices as training targets for the PixelCNN prior.

In summary:

- **Stage 1**: writes checkpoints under `runs/<EXP_NAME_STAGE1>/seed_<SEED>/checkpoints/` and maintains `best.ckpt`.
- **Shell script**: finds the best checkpoint, exports it as `VQVAE_CHECKPOINT`, and passes it to Stage 2 via `--vqvae-checkpoint`.
- **Stage 2 Python code**: reads that path, loads the frozen VQ-VAE from checkpoint, and uses it during prior training.

