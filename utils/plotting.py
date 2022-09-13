import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import cm
import tikzplotlib
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


def plot_powerspectrum(wl,spectrum,scales,wlscales,filename):
    figF,axF = plt.subplots() #figsize=(8,6)
    normalizedsensorspec = [spectrum[i]/scales[i]**0.5 for i in range(len(spectrum))]
    normalizedsensorspec = np.array(normalizedsensorspec)
    plotspec = normalizedsensorspec
    pF = axF.pcolor(wl,np.arange(len(spectrum)),plotspec,cmap='RdBu',shading='auto',vmin=-5,vmax=5)
    colF = figF.colorbar(pF,ax=axF,orientation='horizontal')
    ax2 = axF.twinx()
    ax2.pcolor(wl,wlscales,plotspec,cmap='RdBu',shading='auto',vmin=-5,vmax=5)
    #ax2.set_yticks(axF.get_yticks())
    #ax2.set_ylim(axF.get_ylim())
    figF.canvas.draw()
    #axF.plot(wl,signal/np.max(signal)*30,color='black',linewidth=3)
    #labels = [int(i.get_position()[1]) for i in ax2.get_yticklabels()]
    
    """ labels_new = np.zeros(len(labels))
    for j,i in enumerate(labels):
        if i >= 0:
            labels_new[j] = np.round(wlscales[i],2)

    ax2.set_yticklabels(labels_new) """
    
    ax2.set_ylabel(r'peak width [nm]')
    #rect = patches.Rectangle((wl[1], 29), wl[-2]-wl[1], 1, linewidth=1, edgecolor='r', facecolor='none')
    #axF.add_patch(rect)
    axF.set_xlabel(r'$\lambda$ [nm]')
    axF.set_ylabel(r'level')
    figF.savefig(filename+'.pdf')
    tikzplotlib.save(figure=figF,filepath=filename+'.tex')
    
def get_colormap(N,m='viridis'):
    my_cmap = cm.get_cmap(m, N)
    return my_cmap


def plot_3d(x,y,data,name,text):
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig, ax = plt.subplots()
    ax.text(x[0]-0.2,y[0]-1,text,color='white')

    # Make data.
    
    x, y = np.meshgrid(x, y)
    pl = ax.pcolormesh(x, y, data,shading='auto',cmap='magma_r',vmin=0,vmax=1.0)
    #surf = ax.plot_surface(x, y, data, cmap=cm.magma,linewidth=0, antialiased=False)

    

    # Add a color bar which maps values to colors.
    cb = fig.colorbar(pl)
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label(r'RMSE [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]', rotation=270)
    ax.set_xlabel(r'LAI [$\textrm{m}^2/\textrm{m}^2$]')
    ax.set_ylabel(r'$C_{ab}$ [$\mu \textrm{g} / \textrm{cm}^2$]')
    fig.savefig(name)
    plt.show()


