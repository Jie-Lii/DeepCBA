import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def draw_heat_map(saliency_map, name, location):
    plot_to = saliency_map
    col_sum = []
    nrow = len(plot_to)
    ncol = len(plot_to[0])
    for i in range(ncol):
        col_sum.append(sum(plot_to[x][i] for x in range(nrow)))
    # plot_to, sum_col = keshihua_eqtl(ret_)
    mb = np.array(col_sum)
    plt.figure(figsize=(20, 3))
    plt.title(name)
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(16))
    fig = sns.heatmap(plot_to,
                      center=0,
                      # cmap="OrRd",
                      vmin=np.quantile(plot_to, 0.0001),
                      vmax=np.quantile(plot_to, 0.9999),
                      # linewidths=3,
                      xticklabels=500,  # ['0','500','1000','1500','2000','2500','3000'],
                      yticklabels=['A', 'C', 'G', 'T'],
                      cbar_kws=dict(use_gridspec=False, location="top")
                      )

    fig.get_figure().savefig(location + name + '_ACGT.png', bbox_inches='tight', transparent=True, dpi=1200)
    plt.close()

    plt.figure(figsize=(15, 2))

    plt.plot(mb.T, color='m')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(500))
    plt.title(name)
    plt.subplots_adjust(top=None, bottom=None, right=None, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.savefig(location + name + '_TH.png', dpi=1200)
    plt.close()