import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def reportSOM(report_path="../reports/report.txt", metrics=['SPEEDUP', 'EFFICIENCY', 'COST']):
    df = pd.read_csv(report_path)
    df_cppr = df.loc[df['LANG'] == 'cpp_red']
    df_cpp = df[df['LANG'] == 'cpp']
    df_py = df[df['LANG'] == 'py']

    for m in metrics:
        plt.plot(np.array([1, 2, 3, 4]), df_cppr[m].values, label='CPP_ADV')
        plt.plot(np.array([1, 2, 3, 4]), df_cpp[m].values, label='CPP')
        plt.plot(np.array([1, 2, 3, 4]), df_py[m].values, label='PY')
        plt.grid()
        plt.xlabel("NO. THREADS")
        plt.ylabel(m)
        plt.title(m.lower())
        plt.legend(loc='best')
        plt.savefig("../reports/" + m + ".png")
        plt.show()


def reportSOMPlot(imgs_path="imgs/"):
    report_path_cpp = "reports/report_cpp.csv"
    report_path_cpp_adv = "reports/report_cpp_adv_static.csv"
    report_path_cpp_adv_dyn = "reports/report_cpp_adv_dynamic.csv"
    report_path_py = "reports/report_py.csv"
    reports = [report_path_py, report_path_cpp, report_path_cpp_adv, report_path_cpp_adv_dyn]
    dfs = list()

    for report in reports:
        df = pd.read_csv(report, index_col=False)
        dfs.append(df)

    result = pd.concat(dfs)
    result.to_csv('reports/stats.csv', index=False, header=True)
    result.to_excel('reports/stats_x.xlsx', index=False, header=True)

    # Create a new directory for images if it does not exist
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
        print("Images directory created")

    for y in ['T_P', 'SPEEDUP', 'EFFICIENCY', 'COST']:
        a = result.groupby(['WEIGHT_SIZE', 'GRID_ROWS', 'GRID_COLS', 'N_INPUTS'])  # .apply(lambda x: x)
        ml = [a.get_group(x) for x in a.groups]
        legend = list()
        plt.figure()
        x = "N. THREADS"
        for m in ml:
            f = m[['N_TH', 'T_P', 'SPEEDUP', 'EFFICIENCY', 'COST', 'LANG']].groupby(['LANG'])
            ol = [f.get_group(x) for x in f.groups]
            for o in ol:
                WEIGHT_SIZE = m.iloc[0, m.columns.get_loc('WEIGHT_SIZE')]
                GRID_COLS = m.iloc[0, m.columns.get_loc('GRID_COLS')]
                GRID_ROWS = m.iloc[0, m.columns.get_loc('GRID_ROWS')]
                N_INPUTS = m.iloc[0, m.columns.get_loc('N_INPUTS')]
                ot = o.transpose()
                legend.append(ot.loc["LANG"].reset_index(drop=True)[0])
                c = ot.loc[y].reset_index(drop=True)
                ax = c.plot(style="-o", legend=True, grid=True)
                xlabels = list(ot.loc['N_TH'])
                xticks = [x for x in range(0, len(xlabels))]
                plt.title(
                    y.lower() + ", WEIGHT_SIZE = " + str(WEIGHT_SIZE) + ", N_INPUTS = " + str(N_INPUTS) + ", GRID = (" +
                    str(GRID_ROWS) + "," + str(GRID_COLS) + ")", x=0.5, y=1.05)

                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            ax.legend(legend)
            plt.show()
            fig = ax.get_figure()

            fig.savefig(
                imgs_path + str(WEIGHT_SIZE) + "_" + str(N_INPUTS) + "_" +
                str(GRID_ROWS) + "_" + str(GRID_COLS) + "_" + y.lower() + ".png")
