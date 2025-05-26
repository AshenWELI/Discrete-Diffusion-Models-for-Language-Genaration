import torch
print(torch.__version__)          # Check PyTorch version
print(torch.cuda.is_available())  # Should output "True"
print(torch.version.cuda) 

!python mdlm/main.py \
  model=small-ar \
  data=lm1b \
  wandb.name=ar-lm1b \
  parameterization=ar \
  backbone=ar \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface

  python main.py loader.batch_size=16 loader.eval_batch_size=16 model=small-ar backbone=ar data=lm1b parameterization=ar model.length=128 eval.compute_generative_perplexity=True sampling.steps=1000

b2a997ee3845603198dc79a599d2f6208ec7f7d4

python main.py loader.batch_size=16 eval.compute_generative_perplexity=True sampling.steps=1000 data=wikitext103 parameterization=ar model.length=128 eval.compute_generative_perplexity=True data.cache_dir=$HOME/.cache/textdiffusion


python mdlm/main.py model=small-ar data=wikitext103 parameterization=ar model.length=1024 eval.compute_generative_perplexity=True data.cache_dir=/local/data2/home/thawe276/.cache/textdiffusion loader.batch_size=4 loader.eval_batch_size=4

/local/data2/home/thawe276/Documents/mdlm/outputs/wikitext103/2025.03.21/071148/checkpoints/best.ckpt

# evaluate

python main.py mode=ppl_eval loader.batch_size=4 loader.eval_batch_size=4 data=wikitext103 model=small-ar parameterization=subs backbone=ar model.length=1024 eval.checkpoint_path=/local/data2/home/thawe276/Documents/mdlm/outputs/wikitext103/2025.03.25/010611/checkpoints/best.ckpt +wandb.offline=true

chmod +x run.sh

python mdlm/main.py \
  mode=ppl_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=4 \
  data=wikitext103 \
  model=small-ar \
  parameterization=ar \
  backbone=ar \
  model.length=512 \
  model.hidden_size=768 \
  eval.checkpoint_path=/local/data2/home/thawe276/Documents/outputs/wikitext103/2025.04.04/010336/checkpoints/best.ckpt \
  eval.disable_ema=true \
  data.cache_dir=/local/data2/home/thawe276/.cache/textdiffusion

# sample eval


python mdlm/main.py \
  mode=sample_eval \
  eval.checkpoint_path=/local/data2/home/thawe276/Documents/outputs/wikitext103/2025.04.04/010336/checkpoints/best.ckpt \
  data=wikitext103  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=10000 \
  loader.eval_batch_size=2 \
  sampling.num_sample_batches=2 \
  backbone=ar \
  data.cache_dir=/local/data2/home/thawe276/.cache/textdiffusion \
  +model.causal=True \
  sampling.semi_ar=True


# D3PM training

python ./mdlm/main.py model=small data=wikitext103 parameterization=d3pm model.length=512 eval.compute_generative_perplexity=True sampling.steps=1000 sampling.predictor=ddpm loader.batch_size=8 loader.eval_batch_size=8 time_conditioning=True data.cache_dir=/local/data2/home/thawe276/.cache/textdiffusion seed=1000 T=1000