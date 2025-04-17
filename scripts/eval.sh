#!/bin/bash
export HF_HOME=/root/huggingface
export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail

infer_eval_image_reward() {
    # ${pip_ext} install image-reward pytorch_lightning
    # ${pip_ext} install -U timm diffusers
    # ${pip_ext} install openai==1.34.0 
    # ${pip_ext} install httpx==0.20.0 

    # step 1, infer images
    ${python_ext} evaluation/image_reward/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --use_flex_attn ${use_flex_attn} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir  ${out_dir}

    # step 2, compute image reward
    # ${pip_ext} install diffusers==0.16.0
    # ${pip_ext} install git+ssh://git@github.com/openai/CLIP.git ftfy
    ${python_ext} evaluation/image_reward/cal_imagereward.py \
    --meta_file ${out_dir}/metadata.jsonl
}

infer_eval_hpsv21() {
    # ${pip_ext} install hpsv2
    # ${pip_ext} install -U diffusers
    # apt install python3-tk wget -y
    # wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    # PYSITE="/usr/local/lib/python3.10/dist-packages"
    # mv bpe_simple_vocab_16e6.txt.gz $PYSITE/hpsv2/src/open_clip

    mkdir -p ${out_dir}
    ${python_ext} evaluation/hpsv2/eval_hpsv2.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --use_flex_attn ${use_flex_attn} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images | tee ${out_dir}/log.txt
}

test_gen_eval() {
    # ${pip_ext} install -U openmim
    # mim install mmengine mmcv-full==1.7.2
    # ${pip_ext} install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
    # ${pip_ext} install -U diffusers
    # sudo apt install libgl1
    # ${pip_ext} install openai
    # ${pip_ext} install httpx==0.20.0

    # run inference
    ${python_ext} evaluation/gen_eval/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --use_flex_attn ${use_flex_attn} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --rewrite_prompt ${rewrite_prompt}\
    --si_para ${si_para}\
    --ratio_list ${ratio_list}\
    --kv_opt ${kv_opt}

    # detect objects
    ${python_ext} evaluation/gen_eval/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path weights/mask2former

    # accumulate results
    ${python_ext} evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}

test_fid() {
    # ${pip_ext} install pytorch_fid

    # step 1, infer images
    ${python_ext} tools/comprehensive_infer.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --coco30k_prompts 0 \
    --save4fid_eval 1 \
    --use_flex_attn ${use_flex_attn} \
    --jsonl_filepath ${jsonl_filepath} \
    --long_caption_fid ${long_caption_fid} \
    --out_dir  ${out_dir} \
    --si_para ${si_para}\
    --ratio_list ${ratio_list}\
    --kv_opt ${kv_opt}

    # step 2, compute fid
    ${python_ext} tools/fid_score.py \
    ${out_dir}/pred \
    ${out_dir}/gt | tee ${out_dir}/log.txt
}

test_val_loss() {
    ${python_ext} evaluation/validation_loss/validation_loss.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --save_dir ${out_dir} \
    --reweight_loss_by_scale ${reweight_loss_by_scale} \
    --meta_folder ${jsonl_folder} \
    --noise_apply_strength ${noise_apply_strength} \
    --bf16 0 \
    --log_freq 10
}


python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_2b_reg.pth
out_dir_root=output/infinity_2b_evaluation
vae_type=32
vae_path=weights/infinity_vae_d32reg.pth
cfg=4
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}
use_flex_attn=0
si_para=12
ratio_list='[100]'
kv_opt=0
prefix=1497


# 参数校验
if [ $# -eq 0 ]; then
    echo "Usage: $0 [task_name]"
    echo "Available tasks:"
    echo "  image_reward, hpsv21, gen_eval, mjhq30k_fid, val_loss"
    exit 1
fi

task=$1

case $task in
    image_reward)
        out_dir="${out_dir_root}/image_reward_${sub_fix}_flex_attn${use_flex_attn}_prefix${prefix}"
        infer_eval_image_reward
        break
        ;;
    hpsv21)
        out_dir="${out_dir_root}/hpsv21_${sub_fix}_flex_attn${use_flex_attn}_prefix${prefix}"
        infer_eval_hpsv21
        break
        ;;
    gen_eval)
        rewrite_prompt=2
        out_dir="${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_flex_attn${use_flex_attn}_round2_real_rewrite_prefix${prefix}_mtp1"
        test_gen_eval
        break
        ;;
    mjhq30k_fid)
        long_caption_fid=1
        jsonl_filepath='data/mjhq30k/meta_data.json'
        out_dir="${out_dir_root}/val_mjhq30k_fid_${sub_fix}_flex_attn${use_flex_attn}_si_para${si_para}_ratio${ratio_list}_kvopt${kv_opt}"
        test_fid
        break
        ;;
    val_loss)
        out_dir="${out_dir_root}/val_loss_${sub_fix}_rewrite_prompt${rewrite_prompt}"
        reweight_loss_by_scale=0
        jsonl_folder='[YOUR VAL JSONL FILEPATH]'
        noise_apply_strength=0.2
        test_val_loss
        break
        ;;
    *)
        echo "Error: Unknown task '$task'"
        echo "Available tasks:"
        echo "  image_reward, hpsv21, gen_eval, long_caption_fid, val_loss"
        exit 1
        ;;
esac

echo "Task [$task] executed successfully"
