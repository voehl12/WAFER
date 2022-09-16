import numpy as np


class fluorescence:
    def __init__(self,wl,e_wl):
        self.wl = wl
        self.e_wl = e_wl
        self.spec = np.zeros(len(wl))
        self.Ftotal = float
        self.F687 = float
        self.F760 = float
        self.Fr = float
        self.wlFr = float
        self.Ffr = float
        self.wlFfr = float
        self.spec_val = float

    def evaluate_sif(self):
        """
        evaluates fluorescence at characteristic values and calculates integrated fluorescence
        Function also checks whether standard characteristic values (687,760 (O2 bands),684,735nm (rough fluorescence peaks)) are within the retrieval window
        results are written into the initiated variables of the object.
        """
        Fspectrum = self.spec
        totmin = np.argmin(np.fabs(self.wl-650))
        totmax = np.argmin(np.fabs(self.wl-800))
        self.Ftotal = np.sum(Fspectrum[totmin:totmax])*(self.wl[1]-self.wl[0])
        rel_wls = [687,760,684,735,self.e_wl]
        flags = np.zeros(len(rel_wls))
        argmins = np.zeros(len(rel_wls),dtype=int)
        for j,w in enumerate(rel_wls):
            if w > self.wl[0] and w < self.wl[-1]:
                flags[j] = 1
                argmins[j] = (np.argmin(np.fabs(self.wl-w)))

        if flags[0] == 1:
            self.F687 = Fspectrum[argmins[0]]
        else:
            self.F687 = np.nan

        if flags[1] == 1:
            self.F760 = Fspectrum[argmins[1]]
        else:
            self.F760 = np.nan
        if flags[4] == 1:
            self.spec_val = Fspectrum[argmins[1]]
        else:
            self.spec_val = np.nan

        if flags[2] == 1:
            if argmins[2]-20 < 0 or argmins[2]+20 > len(self.wl):
                self.Fr = Fspectrum[argmins[2]]
                self.wlFr = rel_wls[2]
            else:

                Fr_ind = np.argmax(Fspectrum[argmins[2]-20:argmins[2]+20])
                Fr_ind = Fr_ind + argmins[2]-20
                
                self.Fr = Fspectrum[Fr_ind]
                self.wlFr = self.wl[Fr_ind]
                if Fr_ind == argmins[2]-20 or Fr_ind == argmins[2]+20:
                    print('Did not find red peak!')
                    self.Fr = Fspectrum[argmins[2]]
            
        else:
            self.Fr = np.nan
            self.wlFr = np.nan

        if flags[3] == 1:
            if argmins[3]-20 < 0 or argmins[3]+20 > len(self.wl):
                self.Ffr = Fspectrum[argmins[3]]
                self.wlFfr = rel_wls[3]
            else:

                Ffr_ind = np.argmax(Fspectrum[argmins[3]-20:argmins[3]+20])
                Ffr_ind = Ffr_ind + argmins[3]-20
                
                self.Ffr = Fspectrum[Ffr_ind]
                self.wlFfr = self.wl[Ffr_ind]
                if Ffr_ind == argmins[3]-20 or Ffr_ind == argmins[3]+20:
                    print('Did not find far red peak!')
                    self.Ffr = Fspectrum[argmins[3]]
        else:
            self.Ffr = np.nan
            self.wlFfr = np.nan

        
        
    

class retrieval_res:
    """
    Handling storage and file writing of fluorescence retrieval results.
    """
    def __init__(self,site:str,day:str,wlmin:int,wlmax:int,eval_wl:int,name:str) -> None:
        """
        site, day: any strings specifying a retrieval (can also be model and parameters)
        wlmin,wlmax,eval_wl: retrieval wavelength window and wavelength at which to evaluate SIF
        name: info for the file name (if files should be written)
        """
        self.site = site
        self.day = day
        self.window = wlmin,wlmax
        self.filename = name+'_'+day+'_{:d}{:d}.txt'.format(wlmin,wlmax)
        self.e_wl = eval_wl
        


    def init_wl(self,wl:np.array):
        """
        needs to be called when wl array is set to initiate solution arrays
        """
        self.wl = wl
        self.F = fluorescence(wl,self.e_wl)
        self.R = np.zeros(len(wl))
        self.Fsfm = fluorescence(wl,self.e_wl)
        self.Finp = fluorescence(wl,self.e_wl)

    def initiate_ts_tofile(self):
        """
        initiates a file where results can be documented
        """
        
        with open(self.filename,'w') as f:
            print('**{}'.format(self.site),file=f)
            print('**range: {:d}-{:d} nm'.format(self.window[0],self.window[1]),file=f)
            print('**day: '+self.day,file=f)
            print('time',file=f,end=',')
            for w in self.wl:
                print(float(w),file=f,end=',')
            print(file=f)



    def write_ts_tofile(self,time):
        """
        write fluorescences and reflectances to file 
        time: time for diurnal cycles or string of parameters for modelled spectra
        """
        with open(self.filename,'a') as f:
                print('R',file=f,end=',')
                print(time,file=f,end=',')
                for val in self.R:
                    print(val,file=f,end=',')
                print(file=f)
                print('F',file=f,end=',')
                print(time,file=f,end=',')
                for val in self.F.spec:
                    print(val,file=f,end=',')
                print(file=f)
                print('Fsfm',file=f,end=',')
                print(time,file=f,end=',')
                for val in self.Fsfm.spec:
                    print(val,file=f,end=',')
                print(file=f)


    

        


