import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def reportSOMPlot():
    report_path_cpp = "C:\\Users\\barba\\CLionProjects\\SOM_OMP\\reports\\report_cpp.csv"
    report_path_cpp_adv = "C:\\Users\\barba\\CLionProjects\\SOM_OMP\\reports\\report_cpp_adv.csv"
    report_path_py = "C:\\Users\\barba\\PycharmProjects\\SOM\\reports\\report_py_b.csv"
    reports = [report_path_py, report_path_cpp, report_path_cpp_adv]
    dfs = list()

    for report in reports:
        df = pd.read_csv(report)
        dfs.append(df)
    result = pd.concat(dfs)
    result.to_csv('reports/stats.csv')

    for y in ['T_P', 'SPEEDUP', 'EFFICIENCY', 'COST']:
        # a = result[(result['WEIGHT_SIZE'] == i) & (result['GRID_COLS'] == j) & (result['GRID_ROWS'] == j)]
        a = result.groupby(['WEIGHT_SIZE', 'GRID_ROWS', 'GRID_COLS', 'N_INPUTS'])  # .apply(lambda x: x)
        ml = [a.get_group(x) for x in a.groups]
        legend = list()
        plt.figure()
        x = "N. THREADS"
        for m in ml:
            for e in range(0, len(m), 4):
                b = m[['N_TH', 'T_P', 'SPEEDUP', 'EFFICIENCY', 'COST', 'LANG']].iloc[e:e + 4].transpose()
                LANG = m.iloc[0, m.columns.get_loc('LANG')]
                WEIGHT_SIZE = m.iloc[0, m.columns.get_loc('WEIGHT_SIZE')]
                GRID_COLS = m.iloc[0, m.columns.get_loc('GRID_COLS')]
                GRID_ROWS = m.iloc[0, m.columns.get_loc('GRID_ROWS')]
                N_INPUTS = m.iloc[0, m.columns.get_loc('N_INPUTS')]
                # b = m[['N_TH', 'SPEEDUP', 'EFFICIENCY', 'COST', 'LANG']].transpose()
                legend.append(b.loc["LANG"].reset_index(drop=True)[0])
                c = b.loc[y].reset_index(drop=True)
                ax = c.plot(style="-o", legend=True, grid=True)
                xlabels = list(b.loc['N_TH'])
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
                "imgs/" + y.lower() + "_WEIGHT_SIZE_" + str(WEIGHT_SIZE) + "_N_INPUTS_" + str(N_INPUTS) + "_GRID_" +
                str(GRID_ROWS) + "_" + str(GRID_COLS))
