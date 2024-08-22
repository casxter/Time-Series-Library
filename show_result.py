import os.path
import platform

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 400

FIG_PATH = r'C:\Users\twj\Dropbox\我的文档\研究生\项目\论文2\图表' if platform.system() == 'Windows' else r"/home/qc/twj/paper2_fig"

model_id_dict = {'Transformer': 1, 'Informer': 2, 'CNNLSTM': 3, 'FEDformer': 4, 'TimeMixer': 5, 'FEATimeMixer': 6}

EXP_TIME = "0727"


def read_result():
    # 显示跑完的结果
    metrics_list = []
    with open(f'./result_long_term_forecast_{EXP_TIME}.txt', 'rt') as f:
        for model_str in f.readlines():
            model_str = model_str.rstrip()
            if 'long_term_forecast' in model_str:
                #    ['long', 'term', 'forecast', '20240702-165158', 'Informer', 'custom', 'ftMS', 'sl144', 'll72', 'pl72', 'dm512',
                #    'nh8', 'el2', 'dl1', 'df2048', 'expand2', 'dc4', 'fc3', 'ebtimeF', 'dtTrue', 'soh-residual', 'ae-#0', '0']
                model_str_split = model_str.split('_')
                # mae, mse, rmse, mape, mspe
                metrics = np.load(f"./results/{model_str.rstrip()}/metrics.npy")
                metrics_list.append(
                    {'ev_id': int(model_str_split[20][-1]),
                     'model_name': model_str_split[4],
                     'Model': model_id_dict[model_str_split[4]],
                     'target': model_str_split[20][0:-3],
                     'sl': int(model_str_split[7][2:]),
                     'pl': int(model_str_split[9][2:]),
                     'sl_pl': f"{model_str_split[7][2:]}-{model_str_split[9][2:]}",
                     'MAE': float(metrics[0]),
                     'MSE': float(metrics[1]),
                     'MAPE': float(metrics[2])
                     })
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.describe())
    metrics_df.to_excel(f"./logs/exp_metrics_{EXP_TIME}.xlsx", index=False)
    return metrics_df


def show_all_model(metrics_df):
    sns.set_theme(style="ticks", font_scale=1.8)

    lm = sns.lmplot(
        data=metrics_df, x="Model", y="MSE", col="ev_id", hue='sl_pl', col_wrap=3, palette="muted", ci=None,
        fit_reg=False, legend_out=True
    )
    axs = lm.axes.flatten()
    for i in range(3, 6):
        axs[i].set_xticks(np.linspace(1, 6, 6))
        axs[i].set_xticklabels(list(model_id_dict.keys()), rotation=25)

    for ax in lm.axes.flat:
        ax.grid(True)  # 启用网格线

    plt.subplots_adjust(bottom=0.11)
    # plt.tight_layout()

    plt.savefig(os.path.join(FIG_PATH, 'model_6_mse_comparison.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_6_mse_comparison.pdf'))
    plt.show()

    lm = sns.lmplot(
        data=metrics_df, x="Model", y="MAE", col="ev_id", hue='sl_pl', col_wrap=3, palette="muted", ci=None,
        fit_reg=False, legend_out=True
    )
    axs = lm.axes.flatten()
    for i in range(3, 6):
        axs[i].set_xticks(np.linspace(1, 6, 6))
        axs[i].set_xticklabels(list(model_id_dict.keys()), rotation=25)

    for ax in lm.axes.flat:
        ax.grid(True)  # 启用网格线

    plt.subplots_adjust(bottom=0.11)
    # plt.tight_layout()

    plt.savefig(os.path.join(FIG_PATH, 'model_6_mae_comparison.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_6_mae_comparison.pdf'))
    plt.show()

def show_all_model_violinplot(metrics_df):
    sns.set_theme(style="ticks")
    g = sns.FacetGrid(metrics_df, col="ev_id",col_wrap=3)
    g.map(sns.violinplot, "Model", "MSE", fill=False)

    plt.show()

def show_timemixer_comparison(metrics_df):
    sns.set_theme(style="ticks", font_scale=1.8)

    models = ['FEDformer', 'TimeMixer', 'FEATimeMixer']
    # plt.figure(figsize=(12,9))
    metrics_df = metrics_df[(metrics_df['model_name'] == 'FEDformer') |
                            (metrics_df['model_name'] == 'TimeMixer') |
                            (metrics_df['model_name'] == 'FEATimeMixer')]

    lm = sns.lmplot(
        data=metrics_df, x="Model", y="MSE", col="ev_id", hue='sl_pl', col_wrap=3, palette="muted", ci=None,
        fit_reg=False, legend_out=True
    )

    axs = lm.axes.flatten()
    for i in range(3, 6):
        axs[i].set_xticks(np.linspace(4, 6, 3))
        axs[i].set_xticklabels(models, rotation=25)

    for ax in lm.axes.flat:
        ax.grid(True)  # 启用网格线

    plt.subplots_adjust(bottom=0.11)
    # plt.tight_layout()

    plt.savefig(os.path.join(FIG_PATH, 'model_3_comparison.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_3_comparison.pdf'))
    plt.show()


def show_multistep(metrics_df):
    """
    3.5
    """
    sns.set_theme(style="whitegrid")

    # plt.figure(figsize=(12,9))
    metrics_df = metrics_df[(metrics_df['model_name'] == 'FEATimeMixer')]

    metrics_df_96 = metrics_df[metrics_df['sl'] == 96]
    metrics_df_168 = metrics_df[metrics_df['sl'] == 168]

    # 96 MSE
    axs = sns.lineplot(x="pl", y="MSE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_96)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('prediction step')
    # axs.set_xticklabels(list(model_id_dict.keys()), rotation=30)
    axs.set_title('sequence length 96')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mse_96.png'))
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mse_96.pdf'))
    plt.show()

    # 168 MSE
    axs = sns.lineplot(x="pl", y="MSE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_168)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('prediction step')
    axs.set_title('sequence length 168')
    # axs.set_xticklabels(list(model_id_dict.keys()), rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mse_168.png'))
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mse_168.pdf'))
    plt.show()

    # 96 MAE
    axs = sns.lineplot(x="pl", y="MAE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_96)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('prediction step')
    # axs.set_xticklabels(list(model_id_dict.keys()), rotation=30)
    axs.set_title('sequence length 96')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mae_96.png'))
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mae_96.pdf'))
    plt.show()

    # 168 MSE
    axs = sns.lineplot(x="pl", y="MAE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_168)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('prediction step')
    axs.set_title('sequence length 168')
    # axs.set_xticklabels(list(model_id_dict.keys()), rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mae_168.png'))
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mae_168.pdf'))
    plt.show()


def show_all_model_table(metrics_df):
    """
    附录 3.6
    """
    for ev_id in metrics_df.ev_id.unique():
        for model_name in model_id_dict.keys():
            for sl in [96, 168]:
                print(f"{ev_id}\t{model_name}\t{sl}\t", end='')

                for pl in [96, 192, 336, 720]:
                    r = metrics_df[
                        (metrics_df['ev_id'] == ev_id) &
                        (metrics_df['model_name'] == model_name) &
                        (metrics_df['sl'] == sl) &
                        (metrics_df['pl'] == pl)]
                    print(f"{r['MSE'].iloc[0]}\t{r['MAE'].iloc[0]}\t", end='')
                print("")


def show_rader(metrics_df):
    """
    96-720
    168-720
    """
    metrics_df = metrics_df[(metrics_df['model_name'] == 'FEDformer') |
                            (metrics_df['model_name'] == 'TimeMixer') |
                            (metrics_df['model_name'] == 'FEATimeMixer')]
    # 将数据框分为两个子数据框（根据 'sl_pl' 列的不同值）
    df_96_96 = metrics_df[metrics_df['sl_pl'] == '96-96']
    df_168_96 = metrics_df[metrics_df['sl_pl'] == '168-96']
    df_96_720 = metrics_df[metrics_df['sl_pl'] == '96-720']
    df_168_720 = metrics_df[metrics_df['sl_pl'] == '168-720']

    # 定义绘制雷达图的函数
    def plot_radar(df, title):
        plt.figure(figsize=(5, 5))

        # 获取类别数量
        categories = list(df['ev_id'].unique())
        N = len(categories)

        # 创建雷达图
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # 初始化雷达图
        ax = plt.subplot(111, polar=True)

        # 绘制一个大圆
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # 绘制每一个小类
        plt.xticks(angles[:-1], [f"#{ev_id}" for ev_id in categories])

        # 绘制 y 轴标签
        ax.set_rlabel_position(0)
        # plt.yticks(np.linspace(df['MSE'].min(), df['MSE'].max(), 5), np.linspace(df['MSE'].min(), df['MSE'].max(), 5),
        #            color="grey", size=7)
        # plt.ylim(df['MSE'].min(), df['MSE'].max())

        models = ['FEDformer', 'TimeMixer', 'FEATimeMixer']

        for model in models:
            # 添加数据
            values = df.loc[df['model_name'] == model, 'MSE'].tolist()
            values += values[:1]
            ax.plot(angles, values, label=model)
            ax.fill(angles, values, 'b', alpha=0.1)
        # max min 差值大于2个数量级
        if title in ['sl_pl=96-96', 'sl_pl=168-96']:
            ax.set_yscale('log')

        # 添加标题
        plt.title(title)
        plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.15))

    plot_radar(df_96_96, 'sl_pl=96-96')
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_96_96.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_96_96.pdf'))
    plt.show()

    plot_radar(df_168_96, 'sl_pl=168-96')
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_168_96.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_168_96.pdf'))
    plt.show()

    plot_radar(df_96_720, 'sl_pl=96-720')
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_96_720.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_96_720.pdf'))
    plt.show()

    plot_radar(df_168_720, 'sl_pl=168-720')
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_168_720.png'))
    plt.savefig(os.path.join(FIG_PATH, 'model_comparison_radar_168_720.pdf'))
    plt.show()


metrics_df = read_result()

# show_multistep(metrics_df)
# show_all_model(metrics_df)
# show_timemixer_comparison(metrics_df)
show_rader(metrics_df)
# show_all_model_table(metrics_df)
# show_all_model_violinplot(metrics_df)
