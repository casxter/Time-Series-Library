#!/usr/bin/env bash


EXP_TIME=`date +'%m%d-%H%M'`
export EXP_TIME
LOG_FILE="ae_${EXP_TIME}.log"
model_id='origin'

start=$(date +%s)

for ev_id in $(seq 0 5); do
  echo "ev_id: $ev_id"

  root_path='/home/qc/twj/ml_data/data2/'
  data_path="#${ev_id}_ae.csv"
  c_len=2
  enc_in=$c_len
  dec_in=$c_len
  c_out=$c_len
  target="available_energy"
  des="ae-#${ev_id}"
  train_epochs=6
  # 文件名
  filename="./scripts/seq_label_pred.txt"

  # 逐行读取文件
  while IFS=' ' read -r seq_len label_len pred_len
  do
    # 输出读取的值
    echo "seq_len: $seq_len, label_len: $label_len, pred_len: $pred_len"

    # Transformer
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id\
        --model Transformer \
        --data custom \
        --features MS \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --target $target \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --train_epochs $train_epochs \
        --batch_size 64 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --des $des \
        --itr 1 \
        2>&1 | tee -a logs/$LOG_FILE

    # Informer
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id\
      --model Informer \
      --data custom \
      --features MS \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --target $target \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --train_epochs $train_epochs \
      --batch_size 64 \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des $des \
      --itr 1 \
      2>&1 | tee -a  logs/$LOG_FILE

    # CNN-LSTM
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id\
      --model CNNLSTM \
      --data custom \
      --features MS \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --target $target \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --train_epochs $train_epochs \
      --batch_size 128 \
      --d_model 16 \
      --d_ff 32 \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des $des \
      --itr 1 \
      --cnnlstm_hidden 128 \
      --cnnlstm_nl 3 \
      2>&1 | tee -a  logs/$LOG_FILE

  done < "$filename"
done

end=$(date +%s)

#cp result_long_term_forecast.txt "result_long_term_forecast_${EXP_TIME}_.txt"

echo "Execution time: $((end-start)) seconds" | tee -a  logs/$LOG_FILE
