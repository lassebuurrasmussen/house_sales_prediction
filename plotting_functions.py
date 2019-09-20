import matplotlib.pyplot as plt
from pandas import DataFrame


def histogram(df: DataFrame):
    plt.close('all')
    df.hist(figsize=[12, 8])
    fig: plt.Figure = plt.gcf()
    fig.set_facecolor('white')
    fig.tight_layout()
