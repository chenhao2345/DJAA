export CUDA_VISIBLE_DEVICES=2,3,4,5
python examples/unsupervised_memory.py \
--rho 0.55 \
--batch-size 32 \
--epochs 30 \
--iters 400 \
--k1 30 \
--tau-c 0.5 \
--tau-v 0.1 \
--height 256 \
--width 128 \
--scale-kl 0.2 \
--num-instances 4 \
--mem-samples 1 \
--mem-num 512 \
--lambda-kl 20
