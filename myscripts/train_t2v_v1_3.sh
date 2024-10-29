export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
# NCCL setting
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=162
# export NCCL_IB_TIMEOUT=25
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_ALGO=Ring
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NCCL_IB_RETRY_CNT=32
# export NCCL_ALGO=Tree

accelerate launch \
           --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
           opensora/train/train_t2v_diffusers.py \
           --pretrained "/workspace/public/models/Open-Sora-Plan-v1.3.0/any93x640x640" \
           --output_dir="./runs/test/" \
           --cache_dir "../../cache_dir/" \
           --data "mydata/test.txt" \
           --dataset t2v \
           --sample_rate 1 \
           --max_hxw 262144 \
           --min_hxw 65536 \
           --num_frames 93 \
           --drop_short_ratio 0.0 \
           --group_data \
           --interpolation_scale_t 1.0 \
           --interpolation_scale_h 1.0 \
           --interpolation_scale_w 1.0 \
           --gradient_checkpointing \
           --train_batch_size=4 \
           --dataloader_num_workers 8 \
           --gradient_accumulation_steps=1 \
           --max_train_steps=1000000 \
           --learning_rate=1e-5 \
           --lr_scheduler="constant" \
           --lr_warmup_steps=0 \
           --mixed_precision="bf16" \
           --report_to="wandb" \
           --checkpointing_steps=500 \
           --allow_tf32 \
           --model_max_length 512 \
           --use_ema \
           --ema_start_step 0 \
           --cfg 0.1 \
           --speed_factor 1.0 \
           --ema_decay 0.9999 \
           --pretrained "" \
           --hw_stride 32 \
           --sparse1d --sparse_n 4 \
           --train_fps 16 \
           --seed 1234 \
           --trained_data_global_step 0 \
           --use_decord \
           --prediction_type "v_prediction" \
           --snr_gamma 5.0 \
           --model OpenSoraT2V_v1_3-2B/122 \
           --ae WFVAEModel_D8_4x8x8 \
           --ae_path "/workspace/public/models/Open-Sora-Plan-v1.3.0/vae" \
           --text_encoder_name_1 "/workspace/host_folder/Open-Sora-Plan/google-mt5-xxl" \
           --rescale_betas_zero_snr
           # --resume_from_checkpoint="latest" \
           # --max_height 352 \
           # --max_width 640 \
           # --force_resolution \
