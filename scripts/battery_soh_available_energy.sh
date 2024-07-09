#!/usr/bin/env bash


LOG_FILE="ae_`date +'%Y%m%d-%H%M%S'`.log"

for ev_id in $(seq 0 5); do
  echo "ev_id: $ev_id"

  root_path='/home/qc/twj/ml_data/data2/'
  data_path="#${ev_id}_hour_available_energy_2.csv"
  c_len=2
  enc_in=$c_len
  dec_in=$c_len
  c_out=$c_len
  target="available_energy"
  des="available-energy-2-#${ev_id}"

  # 文件名
  filename="./scripts/seq_label_pred.txt"

  # 逐行读取文件
  while IFS=' ' read -r seq_len label_len pred_len
  do
    # 输出读取的值
    echo "seq_len: $seq_len, label_len: $label_len, pred_len: $pred_len"

    # Transformer
    model_name=Transformer
    model_id=`date +'%Y%m%d-%H%M%S'`

    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id\
        --model $model_name \
        --data custom \
        --features MS \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --target $target \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --batch_size 16 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --des $des \
        --itr 1 \
        2>&1 | tee logs/$LOG_FILE

    # Informer
    model_name=Informer
    model_id=`date +'%Y%m%d-%H%M%S'`

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id\
      --model $model_name \
      --data custom \
      --features MS \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --target $target \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --batch_size 16 \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des $des \
      --itr 1 \
      2>&1 | tee logs/$LOG_FILE

      # FEDformer
      # 爆内存
      model_name=FEDformer
      model_id=`date +'%Y%m%d-%H%M%S'`

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id\
        --model $model_name \
        --data custom \
        --features MS \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --target $target \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --des $des \
        --itr 1 \
        2>&1 | tee logs/$LOG_FILE

      #CNN-LSTM
      model_name=CNNLSTM
      model_id=`date +'%Y%m%d-%H%M%S'`

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id\
        --model $model_name \
        --data custom \
        --features MS \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --target $target \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
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
        2>&1 | tee logs/$LOG_FILE

    # TimeMixer
    model_name=TimeMixer
    model_id=`date +'%Y%m%d-%H%M%S'`

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
      --model $model_name \
      --data custom \
      --features MS \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --target $target \
      --e_layers 2 \
      --batch_size 64 \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des $des \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate 0.001 \
      --patience $patience \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window \
      2>&1 | tee logs/$LOG_FILE

  done < "$filename"
done