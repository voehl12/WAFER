from ssqueezepy import ssq_cwt, ssq_stft,cwt,icwt,issq_cwt
import numpy as np
import pywt

class decomp:
    """ class that manages all decomposition characteristics and can hold the decomposition but also the masking for absorption lines
        it can also be used to determine the best decomposition levels from weighting
    """
    def __init__(self,jmin,jmax,nlevel) -> None:
        
        self.jmin = jmin
        self.jmax = jmax
        self.nlevel = nlevel
        self.scales = np.logspace(jmin,jmax,num=nlevel,base=2)
        self.optlevel = int


    def calc_scales(self):
        self.scales =  np.logspace(self.jmin,self.jmax,num=self.nlevel,base=2)

    def adjust_levels(self,testsignal):

        weights = determine_weights(testsignal,self.scales)
        
        weightthres = np.max(weights)*0.9
        newscaleinds = []
        for i in range(len(weights)):
            
            if weights[i] >= weightthres:
                newscaleinds.append(i)
        newjmin = np.log2(self.scales[np.min(newscaleinds)])
        newjmax = np.log2(self.scales[np.max(newscaleinds)])
        if (newjmin,newjmax) == (self.jmin,self.jmax):
            pass
            
        elif newjmax-newjmin > 0.5:
            self.nlevel = 10
            self.jmin = newjmin
            self.jmax = newjmax
            self.calc_scales()
            self.adjust_levels(testsignal)
        else:
            pass

    def create_comps(self,testsignal):
        comps = create_decomp_p(testsignal,self.scales)
        self.comps = comps

    def calc_mask(self,testsignal):
        comps = self.create_comps(testsignal)
        masks = []
        for i in range(len(self.scales)):
            negmask = np.ma.masked_where(comps[i] <= 0, comps[i])
            masks.append(np.ma.masked_where(comps[i] <= -np.median(np.fabs(comps[i,negmask.mask]))/0.6745, comps[i]))
        self.masks = masks
        # thresholding: wavelet tour, page 565
        

    def calc_weights(self,testsignal):
        self.calc_mask(testsignal)
        contrib_counts = np.zeros((len(self.scales)))
        for j in range(len(self.scales)):
            scalecounts = 0
            
            try:
                for k in self.masks[j].mask:
                    
                    if k == True:
                        scalecounts += 1
                contrib_counts[j] += scalecounts
            except: continue
        self.weights = contrib_counts






def determine_weights(signal,scales):
       
    sigdecomp = create_decomp_p(signal,scales) 
        
    masks = []
    for i in range(len(scales)):
        negmask = np.ma.masked_where(sigdecomp[i] <= 0, sigdecomp[i])
        levelmask = np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i])
        
        
        masks.append(np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i]))/0.6745, sigdecomp[i]))
    
    # thresholding: wavelet tour, page 565
    contrib_counts = np.zeros((len(scales)))

        
    for j,scale in enumerate(scales):
        scalecounts = 0
        
        try:
            for k in masks[j].mask:
                
                if k == True:
                    scalecounts += 1
            contrib_counts[j] += scalecounts
        except: continue
           
 
            
        
                
        
    
    return contrib_counts


def create_decomp_p(data,scales,level='all'):
   
    if level == 'all':
        inds = np.arange(len(scales))
    else:
        inds = level
    data = np.array(data)

    
    
    j = scales[inds]
        
    coef,freqs = pywt.cwt(data,j,'gaus2',method='conv')
    #coef,ssqscales = cwt(data,wavelet='cmhat',scales=j)
    
    return coef



def get_wlscales(scales):
    f = pywt.scale2frequency('gaus2', scales)
    wlscales = 1/f
    return wlscales

def icwt_ss(decomp,scales,mean,minrec):
    res = icwt(decomp,wavelet='cmhat',scales=scales,x_mean=mean,recmin=minrec)
    return res

def icwavelet(wave,scale,minrec=0,maxrec=0):
    #scale = np.logspace(jmin,jmax,num=N_level,base=2.0)
    N = len(wave[0])
    N_level = len(scale)
    js = np.log2(scale)
    jmin = js[0]
    jmax = js[-1]
    xdelta = np.zeros(N)
    xdelta[0] = 1
    Wdelta = create_decomp_p(xdelta,scale)
    resdelta = np.zeros((len(Wdelta),N))
    for j in range(len(Wdelta)):
        resdelta[j] = np.divide(Wdelta[j],(scale[j])**0.5)#np.divide(Wdelta[j],(scale[j])**(1/2))
    recdelta = np.zeros(N)
    for i in range(N):
        recdelta = np.sum(resdelta,axis=0)
    cdelta = recdelta[0]
   
    dj = (jmax-jmin) / N_level
 
    coeff = 1.0 / (cdelta)
    
    oup = np.zeros(N)
    
    result = np.zeros((len(wave),N))
    jtot = len(wave)
    if jmax >= jtot:
        
        jmax = jtot
    for j in range(len(wave)):
        result[j] = wave[j]/scale[j]**0.5
    if maxrec == 0:
        for i in range(N):
            oup[i] = np.sum(result[minrec:,i]) 
    else:
        for i in range(N):
            oup[i] = np.sum(result[minrec:-maxrec,i]) 

    oup *= coeff
    return oup