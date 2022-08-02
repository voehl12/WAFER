import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import cm



def plot_powerspectrum(wl,spectrum,scales,wlscales,signal):
    figF,axF = plt.subplots(figsize=(8,6))
    normalizedsensorspec = spectrum#[spectrum[i]/scales[i]**0.5 for i in range(len(spectrum))]
    normalizedsensorspec = np.array(normalizedsensorspec)
    plotspec = normalizedsensorspec
    pF = axF.pcolor(wl,scales,plotspec,cmap='RdBu',shading='auto',vmin=-10,vmax=10)
    colF = figF.colorbar(pF,ax=axF,orientation='horizontal')
    ax2 = axF.twinx()
    #ax2.pcolor(wl,wlscales,plotspec,cmap='RdBu',shading='auto',vmin=-1,vmax=1)
    ax2.set_yticks(axF.get_yticks())
    ax2.set_ylim(axF.get_ylim())
    figF.canvas.draw()
    #axF.plot(wl,signal/np.max(signal)*30,color='black',linewidth=3)
    labels = [int(i.get_position()[1]) for i in ax2.get_yticklabels()]
    print(labels)
    labels_new = np.zeros(len(labels))
    """ for j,i in enumerate(labels):
        if i >= 0:
            labels_new[j] = np.round(wlscales[i],2)

    ax2.set_yticklabels(labels_new) """
    ax2.set_ylabel(r'peak width [nm]')
    #rect = patches.Rectangle((wl[1], 29), wl[-2]-wl[1], 1, linewidth=1, edgecolor='r', facecolor='none')
    #axF.add_patch(rect)
    axF.set_xlabel(r'$\lambda$ [nm]')
    axF.set_ylabel(r'level')
    plt.show()

def get_colormap(N):
    my_cmap = cm.get_cmap('viridis', N)
    return my_cmap