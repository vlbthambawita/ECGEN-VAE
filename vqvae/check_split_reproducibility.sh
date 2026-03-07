# Standard check — tests seed=42 twice (must match) + seed=123 (must differ)
python check_split_reproducibility.py \
    --data-dir /work/vajira/data/mimic_iv_original/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --seeds 42 42 123 \
    --n-repeats 3 \
    --output-dir split_diagnostics

# Also save a fixed index file for guaranteed future reproducibility
#python check_split_reproducibility.py \
#    --data-dir /path/to/mimic-iv-ecg \
#    --seeds 42 42 123 \
#    --save-fixed-splits fixed_splits_seed42.json