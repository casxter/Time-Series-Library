#!/usr/bin/env bash


EXP_TIME=`date +'%m%d-%H%M'`
export EXP_TIME
LOG_FILE="r_ae_${EXP_TIME}.log"
model_id='propose'

start=$(date +%s)

for ev_id in $(seq 0 5); do
  echo "ev_id: $ev_id"

  root_path='/home/qc/twj/ml_data/data2/'
  data_path="#${ev_id}_r_ae_imf.csv"
  c_len=20
  enc_in=$c_len
  dec_in=$c_len
  c_out=$c_len
  target="available_energy"
  des="r-ae-imf-#${ev_id}"
  train_epochs=10

  # 文件名
  filename="./scripts/seq_label_pred.txt"

  # 逐行读取文件
  while IFS=' ' read -r seq_len label_len pred_len
  do
    # 输出读取的值
    echo "seq_len: $seq_len, label_len: $label_len, pred_len: $pred_len"

    # FEATimeMixer

    d_model=16
    d_ff=32
    patience=10
    down_sampling_layers=3
    down_sampling_window=2

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id\
      --model FEATimeMixer \
      --data custom \
      --features MS \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --target $target \
      --e_layers 2 \
      --d_layers 1 \
      --train_epochs 10 \
      --batch_size 32 \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des $des \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate 0.001 \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window \
      2>&1 | tee -a  logs/$LOG_FILE

  done < "$filename"
done

end=$(date +%s)

echo "Execution time: $((end-start)) seconds" | tee -a  logs/$LOG_FILE
