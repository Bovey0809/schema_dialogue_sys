#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train_intents.py \
  --bert_model bert-base-cased \
  --load_model_dir '../pre_trained_models_base_cased/' \
  --history_model_file 'output_base/2019-09-24_12_54_24_pytorch_model.bin' \
  --do_eval \
  --data_dir '/diskb/houbw/dstc8-schema-guided-dialogue' \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --num_train_epochs 1 \
  --max_seq_length 196 \
  --output_dir output_base/ \
  --gradient_accumulation_steps 1 #> log_intent_base/log_0p9sampleAllDev_onTest 2>&1 & #> log_intent_base/log_bsz32_warm0p1_epoch3_maxUttrLen64_withSYSTurn_MixedSlotDespTrue_lastIntentTag_frameChangeTag_0p99SampleAllDev.txt 2>&1 &


# python train_cross_copy.py \
#   --bert_model bert-base-cased \
#   --load_model_dir '../pre_trained_models_base_cased/' \
#   --history_model_file 'output_base/2019-09-27_17_03_58_pytorch_model.bin' \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.3 \
#   --num_train_epochs 1 \
#   --max_seq_length 160 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1 > log_slotCross_base/log_0p99sampleAllDev_onTest_1011 2>&1 & # log_slotCross_base/log_0p01PriorNegSampleMultiRunTrue_bsz32_warm0p3_epoch3_maxUttrLen72_withUserIntentInfo_expand_mixSlotDesp_0p99SampleAllDev.txt 2>&1 & #>log_slotCross_base/log_data_check.txt 2>&1 & #

# --history_model_file 'output_base/2019-09-06_23_16_35_pytorch_model.bin' \
# python train_copy_slot.py \
#   --bert_model bert-base-cased \
#   --load_model_dir '../pre_trained_models_base_cased/' \
#   --history_model_file 'output_base/2019-09-28_23_19_36_pytorch_model.bin' \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.7 \
#   --num_train_epochs 1 \
#   --max_seq_length 160 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1 > log_slotCopy_base/log_0p99sampleAllDev_onTest_1011 2>&1 & #log_slotCopy_base/log_0p1NegSampleMultiRunTrue_bsz32_warm0p7_epoch3_maxUttrLen72_filtINFORM_withUserIntentInfo_0p99SampleAllDev.txt 2>&1 &


# python train_slotNotCare.py \
#   --bert_model bert-base-cased \
#   --load_model_dir '../pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.1 \
#   --num_train_epochs 3 \
#   --max_seq_length 128 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1 > log_slotNotCare_base/log_0p01NegSampleMultiRunTrue_bsz32_warm0p1_epoch5_maxUttrLen72_0p9SampleDev.txt 2>&1 &

# '''
# joint-pretrained model '../scripts_single_domain/output_base/2019-09-02_22_54_01_pytorch_model.bin'
# pretrained bert-base models
# trained model for non-categorical slots: output_base/2019-09-03_07_21_34_pytorch_model.bin
# trained model for categorical slots: output_base/2019-09-04_14_11_04_pytorch_model.bin
# trained model for requested slots: output_base/2019-09-02_22_54_01_pytorch_model.bin
# '''


# --history_model_file 'output_base/2019-10-12_13_15_30_pytorch_model.bin' \
# python train_combine.py \
#   --bert_model bert-base-cased \
#   --load_model_dir '../pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.1 \
#   --num_train_epochs 3 \
#   --max_seq_length 144 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1 > log_joint_base/log_0p02NegsampleMultiRunTrue_bsz32_warm0p1_epoch3_maxUttrLen72_0p0SampleAllDev_joint.txt 2>&1 &

#> log_joint_base/log_0p0sampleAllDev_finalTestOnDev_onlyT4_seqL160_uttrL96 2>&1 & #
