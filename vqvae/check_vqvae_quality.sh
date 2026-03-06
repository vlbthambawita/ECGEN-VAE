# Standard run (evaluates 20 batches × 16 samples = 320 ECGs)
python check_vqvae_quality.py \
    --vqvae-checkpoint /work/vajira/DL2026/ECGEN-VAE/vqvae/runs/vqvae_mimic_standalone/seed_42/checkpoints/epoch098-step486981.ckpt \
    --data-dir /work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --n-batches 20 \
    --output-dir diagnostics

# Quick run (fewer batches)
#python check_vqvae_quality.py \
#    --vqvae-checkpoint runs/vqvae_mimic_standalone/seed_42/checkpoints/best.ckpt \
#    --data-dir /work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
#    --n-batches 5 --quick