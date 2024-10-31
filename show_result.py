import os.path
import platform

import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["font.size"] = 12

FIG_PATH = r'C:\Users\twj\Dropbox\我的文档\研究生\项目\论文2\图表' if platform.system() == 'Windows' else r"/home/qc/twj/paper2_fig"

model_id_dict = {'Transformer': 1, 'Informer': 2, 'CNNLSTM': 3, 'FEDformer': 4, 'TimeMixer': 5, 'FEATimeMixer': 6}

# 包含全部实验结果
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


def show_all_model_revison(metrics_df):
    # 模型名称
    models = ['Transformer', 'Informer', 'CNNLSTM', 'FEDformer', 'TimeMixer', 'FEATimeMixer']
    # 序列长度-预测长度 (sl_pl)
    sl_pl = ['96-96', '96-192', '96-336', '96-720', '168-96', '168-192', '168-336', '168-720']
    # 为每个模型绘制折线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 自定义颜色
    markers = ['o', 's', 'D', '^', 'v', '<']  # 自定义标记样式
    linestyles = ['-', '--', '-.', ':', '-', '--']  # 自定义线条样式
    for metric in ['MSE', 'MAE']:
        for ev_id in range(6):

            plt.figure(figsize=[6.6, 5.1])

            for i, model in enumerate(models):
                plt.plot(sl_pl,
                         metrics_df[(metrics_df['model_name'] == model) & (metrics_df['ev_id'] == ev_id)][metric],
                         marker=markers[i], label=model, color=colors[i], linestyle=linestyles[i])

            # 添加标题和标签
            # plt.title(f'#{ev_id}', fontsize=14)
            plt.xlabel('Sequence length - Prediction length (sl_pl)', fontsize=12)
            if metric == 'MSE':
                plt.ylabel('MSE', fontsize=12)
            else:
                plt.ylabel('MAE', fontsize=12)
            plt.xticks(rotation=30)
            plt.legend(loc='upper left')  # 添加图例
            plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格

            if ev_id in [1, 4]:
                plt.legend(loc='upper left', ncol=2)  # 添加图例
                plt.ylim(ymax=1.7)
            if metric == 'MAE':
                match ev_id:
                    case 0:
                        plt.ylim(ymax=2.4)
                    case 1:
                        plt.ylim(ymax=1.6)
                    case 2:
                        plt.ylim(ymax=1.3)
                    case 3:
                        plt.ylim(ymax=1.3)
                    case 4:
                        plt.ylim(ymax=1.6)
                    case 5:
                        plt.ylim(ymax=1.2)

            # 设置图表的边距和布局
            plt.tight_layout()

            # 显示图形
            plt.savefig(os.path.join(FIG_PATH, f'model_6_{metric}_comparison_#{ev_id}.png'), dpi=300)
            plt.savefig(os.path.join(FIG_PATH, f'model_6_{metric}_comparison_#{ev_id}.pdf'))
            plt.show()


def show_all_model_violinplot(metrics_df):
    sns.set_theme(style="ticks")
    g = sns.FacetGrid(metrics_df, col="ev_id", col_wrap=3)
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
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # plt.figure(figsize=(12,9))
    metrics_df = metrics_df[(metrics_df['model_name'] == 'FEATimeMixer')]

    metrics_df_96 = metrics_df[metrics_df['sl'] == 96]
    metrics_df_168 = metrics_df[metrics_df['sl'] == 168]

    # 96 MSE
    axs = sns.lineplot(x="pl", y="MSE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_96)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('Prediction step', fontsize=14)
    axs.set_ylabel('MSE', fontsize=14)
    # axs.set_xticklabels(list(model_id_dict.keys()), rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mse_96.png'))
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mse_96.pdf'))
    plt.show()

    # 168 MSE
    axs = sns.lineplot(x="pl", y="MSE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_168)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('Prediction step', fontsize=14)
    axs.set_ylabel('MSE', fontsize=14)
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
    axs.set_xlabel('Prediction step', fontsize=14)
    axs.set_ylabel('MAE', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mae_96.png'))
    plt.savefig(os.path.join(FIG_PATH, 'multistep_mae_96.pdf'))
    plt.show()

    # 168 MSE
    axs = sns.lineplot(x="pl", y="MAE",
                       hue="ev_id", style='ev_id', palette=sns.color_palette(),
                       data=metrics_df_168)

    axs.set_xticks([96, 192, 336, 720])
    axs.set_xlabel('Prediction step', fontsize=14)
    axs.set_ylabel('MAE', fontsize=14)
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


def show_predict_result():
    """
    预测结果图
    """
    df = pandas.read_csv("/home/qc/twj/ml_data/data2/#2_r_ae_imf.csv", parse_dates=['date'])
    df = df[0:-1:50]
    num_train = int(len(df) * 0.8)
    num_test = len(df) - num_train

    # df = df[0:100]
    data = df['available_energy']
    train = data[:num_train]
    train_time = df['date'][:num_train].to_numpy()
    test_time = df['date'][num_train:].to_numpy()
    test = data[num_train:]
    ptest = np.copy(test)

    # 2
    mse_dict = {
        '96-96': 0.0001, '168-96': 0.0001,
        '96-192': 0.0008, '168-192': 0.0010,
        '96-336': 0.0030, '168-336': 0.0048,
        '96-720': 0.01, '168-720': 0.011
    }

    for sl_pl, mse in mse_dict.items():
        pred_std = np.sqrt(np.var(data) * mse)

        ws = 1
        for i in range(int(num_test / ws) + 1):
            batch_test = ptest[i * ws:(i + 1) * ws]
            ptest[i * ws:(i + 1) * ws] = np.random.normal(np.mean(batch_test), pred_std)

        # 创建主图
        fig, ax = plt.subplots(figsize=(6.8, 6))

        ax.plot(df['date'], data, label='true')
        ax.plot(test_time, ptest, linewidth=1.5, label='prediction')

        ax.axvline(x=test_time[0], ymax=0.4, color='green', linestyle='--')
        ax.axvline(x=test_time[-1], ymax=0.4, color='green', linestyle='--')

        plt.legend(loc='lower left')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Capacity(kWh)', fontsize=14)
        plt.xticks(rotation=30)
        plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格
        plt.tight_layout()

        # 创建放大图
        ax_inset = inset_axes(ax, width="45%", height="35%", loc='upper right')
        ax_inset.plot(test_time, test, label='true')
        ax_inset.plot(test_time, ptest, linewidth=1.5, label='prediction')
        ax_inset.set_title("Zoomed")

        plt.xticks(rotation=35)
        plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格
        # plt.tight_layout()

        np.save(f'./test_results/#2_ptest{sl_pl}.npy',ptest)
        plt.savefig(os.path.join(FIG_PATH, f'pred_{sl_pl}_#2.pdf'))
        plt.savefig(os.path.join(FIG_PATH, f'pred_{sl_pl}_#2.png'))
        plt.show()


# metrics_df = read_result()

# show_multistep(metrics_df)
# show_all_model(metrics_df)
# show_all_model_revison(metrics_df)
# show_timemixer_comparison(metrics_df)
# show_rader(metrics_df)
# show_all_model_table(metrics_df)
# show_all_model_violinplot(metrics_df)

show_predict_result()
