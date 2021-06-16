import torchaudio
import matplotlib.pyplot as plt
import numpy as np

def plot_audio_signal(data, path, lw=0.1, stem=False):
    fig, ax = plt.subplots()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if stem:
        ax.stem(np.arange(len(data)), data, 'black', basefmt=" ", markerfmt="k,")
    else:
        ax.plot(data, 'black', lw=lw)
    ax.set_ylim((-0.8, 0.8))
    fig.savefig(path, bbox='tight', bbox_inches='tight', pad_inches=0.)


wav = torchaudio.load("stylegan2.wav")
wav = wav[0].squeeze()
plot_audio_signal(wav, "stylegan2.png", lw=1.0, stem=False)
