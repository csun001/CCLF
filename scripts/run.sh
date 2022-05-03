python -u train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp \
    --agent CCLF_sac --frame_stack 3 \
    --seed 360 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 1250 --batch_size 512 --num_train_steps 63000 --K_num 5 --M_num 5
