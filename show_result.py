import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="ticks")

model_id_dict = {'Transformer': 1, 'Informer': 2, 'FEDformer': 3, 'CNNLSTM': 4,'TimeMixer': 5}

def read_result():
    # 显示跑完的结果
    metrics_df = pd.DataFrame(columns=['ev_id', 'model', 'target', 'sl', 'pl', 'mae', 'mse', 'mape'])
    with open('./result_long_term_forecast.txt', 'rt') as f:
        for model_str in f.readlines():
            model_str = model_str.rstrip()
            if 'long_term_forecast' in model_str:
                #    ['long', 'term', 'forecast', '20240702-165158', 'Informer', 'custom', 'ftMS', 'sl144', 'll72', 'pl72', 'dm512',
                #    'nh8', 'el2', 'dl1', 'df2048', 'expand2', 'dc4', 'fc3', 'ebtimeF', 'dtTrue', 'soh-residual', 'ae-#0', '0']
                model_str_split = model_str.split('_')
                # mae, mse, rmse, mape, mspe
                metrics = np.load(f"./results/{model_str.rstrip()}/metrics.npy")
                metrics_df = metrics_df.append(
                    {'ev_id': int(model_str_split[20][-1]),
                     'model': model_str_split[4],
                     'model_id': model_id_dict[model_str_split[4]],
                     'target': model_str_split[20][0:-3],
                     'sl': int(model_str_split[7][2:]),
                     'pl': int(model_str_split[9][2:]),
                     'sl_pl': f"{model_str_split[7][2:]}-{model_str_split[9][2:]}",
                     'mae': float(metrics[0]),
                     'mse': float(metrics[1]),
                     'mape': float(metrics[2])
                     },ignore_index=True)
    print(metrics_df.describe())
    metrics_df.to_excel("./logs/exp_metrics_7_10.xlsx", index=False)
    return metrics_df
    # metrics_df = metrics_df[(metrics_df['mse'] < 100) & (metrics_df['target'] == 'soh-residual-ae-imf')]

def show_all_model():
    metrics_df = read_result()

    plt.figure(figsize=(12,9))

    lm = sns.lmplot(
        data=metrics_df, x="model_id", y="mse", col="ev_id", hue='sl_pl', col_wrap=3, palette="muted", ci=None,
        scatter_kws={"s": 50, "alpha": 1},fit_reg=False,legend_out=True
    )
    axs = lm.axes.flatten()
    for i in  range(3,6):
        axs[i].set_xticks(np.linspace(1, 5, 5))
        axs[i].set_xticklabels(list(model_id_dict.keys()), rotation=30)

    # plt.tight_layout()
    plt.show()

def show_fed_timemixer():
    metrics_df = read_result()
    plt.figure(figsize=(12,9))
    metrics_df = metrics_df[(metrics_df['model'] == 'FEDformer') | (metrics_df['model'] == 'TimeMixer')]

    lm = sns.lmplot(
        data=metrics_df, x="model_id", y="mse", col="ev_id", hue='sl_pl', col_wrap=3, palette="muted", ci=None,
        scatter_kws={"s": 50, "alpha": 1}, fit_reg=False, legend_out=True
    )
    # axs = lm.axes.flatten()
    # for i in range(3, 6):
    #     axs[i].set_xticks(np.linspace(1, 5, 5))
    #     axs[i].set_xticklabels(list(model_id_dict.keys()), rotation=30)

    plt.tight_layout()
    plt.show()

show_all_model()
# show_fed_timemixer()