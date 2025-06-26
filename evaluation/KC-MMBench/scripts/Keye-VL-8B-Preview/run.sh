export OPENAI_API_KEY=""

torchrun --master-port 1241 --nproc-per-node=8 run.py \
    --config cfg.json \
    --mode all \
    --verbose