# Full run — evaluates 20 batches + generates at 4 temperatures
python check_prior_quality.py \
    --prior-checkpoint /work/vajira/DL2026/ECGEN-VAE/vqvae/runs/prior_mimic_standalone/seed_42/checkpoints/epoch098-step486981.ckpt \
    --data-dir /work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --n-batches 20 \
    --output-dir prior_diagnostics

# Quick run
#python check_prior_quality.py \
#    --prior-checkpoint ... --data-dir ... --n-batches 5 --quick