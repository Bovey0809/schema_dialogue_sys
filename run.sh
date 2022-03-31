export CUDA_VISIBLE_DEVICES=0
# python -m torch.distributed.launch --nproc_per_node=8 train_intents.py \
#   --bert_model 'pre_trained_models_base_cased/' \
#   --load_model_dir 'pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '/dstc8-schema-guided-dialogue' \
#   --train_batch_size 128 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.1 \
#   --num_train_epochs 3 \
#   --max_seq_length 196 \
#   --fp16 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1 > log_intent_base/original_params.txt 2>&1


# python train_cross_copy.py \
#   --bert_model bert-base-cased \
#   --load_model_dir 'pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '/dstc8-schema-guided-dialogue' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.3 \
#   --num_train_epochs 1 \
#   --max_seq_length 160 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1

# --history_model_file 'output_base/2019-09-06_23_16_35_pytorch_model.bin' \
# python train_copy_slot.py \
#   --bert_model bert-base-cased \
#   --load_model_dir 'pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.7 \
#   --num_train_epochs 1 \
#   --max_seq_length 160 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1


# python train_slotNotCare.py \
#   --bert_model bert-base-cased \
#   --load_model_dir 'pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.1 \
#   --num_train_epochs 3 \
#   --max_seq_length 128 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1 

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
#   --load_model_dir 'pre_trained_models_base_cased/' \
#   --do_train \
#   --do_eval \
#   --data_dir '' \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --warmup_proportion 0.1 \
#   --num_train_epochs 3 \
#   --max_seq_length 144 \
#   --output_dir output_base/ \
#   --gradient_accumulation_steps 1