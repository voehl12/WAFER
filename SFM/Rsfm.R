Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS=TRUE) 
if(!require(remotes)) install.packages("remotes")
if(!require(devtools)) install.packages("devtools")
if(!require(FieldSpectroscopyCC)) remotes:: install_github("tommasojulitta/FieldSpectroscopyCC") 
if(!require(FieldSpectroscopyDP)) remotes:: install_github("tommasojulitta/FieldSpectroscopyDP") 
if(!require(gWidgets)) install.packages(c("gWidgets","gWidgetstcltk"),repos = "https://www.jb-hyperspectral.com/jb-uploads/",type = "source")
if(!require(RJSONIO)) install.packages("RJSONIO")
if(!require(RCurl)) install.packages("RCurl")
if(!require(foreach)) install.packages("foreach")
if(!require(doSNOW)) install.packages("doSNOW")
if(!require(DescTools)) install.packages("DescTools")
####################################################################################################################################################################################
library(devtools)
library(splines)
library(FieldSpectroscopyCC)
library(FieldSpectroscopyDP)
library(RJSONIO)
library(curl)
library(bitops)
library(foreach)
library(doParallel)
library(doSNOW)
library(DescTools)
####################################################################################################################################################################################


SFM<-function(
      ### Compute spectral fitting methods on measured data. It returns fluorescence and true reflectance
      wl,  ##<< numeric vector: wavelength vector
      L,  ##<< numeric vector: measued reflected radiance
      E,  ##<< numeric vector: measured solar radiance
      fluoFG,  ##<< numeric value: fluorescence estimate derived from iFLD method
      O2band,  ##<< character value: A or B referring to the oxygen absorption band where to compute the fluorescence estimation
      output  ##<< character value: FULL or VALUE referring to output expected. If FULL a data.fame of the spectrum in the considered range of fluorescence and true reflectance is returned. If VALUE the fluorescence and the true reflectance at the selected oxygen band is returned.
    )
    {
      if(O2band =="B")
      { 
        range<-which(wl>684& wl<700)
      }
      if(O2band =="A")
      { 
        range<-which(wl>750& wl<780)
      }
      
      WL<-wl[range]
      E_sfm<-as.numeric(E[range])
      L_sfm<-as.numeric(L[range])
      fluoFG<-fluoFG
      fg<-FirstGuess(wl = WL,L = L_sfm,E = E_sfm, fluo = fluoFG,O2band= O2band)
      print(fg)
      res<-optim(fg$first_guess$FG,fn=SpecFit,wl=WL,E=E_sfm,L=L_sfm,fm=fg$fm,run="inverse", method="BFGS",
                 O2band =O2band)
      controllb<-as.numeric(res$par) - as.numeric(fg$first_guess$lb);controllb<-which(controllb==0)
      controlub<-as.numeric(res$par) - as.numeric(fg$first_guess$ub);controlub<-which(controlub==0)
      
      warning<-0
      if(length(controllb)>0){warning<-1}
      if(length(controlub)>0){warning<-2}
      if(length(controllb)>0 & length(controlub)>0){warning<-3}
      
      convergence<-res$convergence
      costF<-res$value
      
      ###parametri conv
      ### value da optim per cost function
      Results<-SFMResults(res = res,wl = WL,output = output,fm=fg$fm,O2band = O2band) 
      #list including warnings
      out<-list(Results,convergence,costF,warning);names(out)<-c("Results","convergence","costF","warning")
      ##value<< List containing: 
      # - Results of the spectral fitting methods, (see \code{\link{SFMResults}})
      # - Convergence of the optimization, (see \code{\link{optim}})
      # - Cost function, (see \code{\link{SpecFit}})
      # - warning. 0, no warning. 1 results recalc the lower bounday first guess. 2 results recalc the upper bounday first gues. 3 results recalc the lower and upper bounday first gues. 
      return(out)
    }


ReadDFloX<-function(
      ### Read Data from csv files saved by D-FloX
      filename  ##<< character value or vector: names of the file(s) to be opened
      ,sep=";"  ##<< the field separator character
      ,na.strings = "#N/D"
      ,header=FALSE ##<< logical value indicating whether the file contains the names of the variables as its first line
      ,Ename = "WR" ##<< character value: string of the name in the ASCII file of the solar radiance vector, if any
      ,Lname ="VEG" ##<< character value: string of the name in the ASCII file of the reflected radiance vector, if any
      ,E2name ="WR2"  ##<< character value: string of the name in the ASCII file of the reflected radiance vector collected at double IT, if any

    )
    {
      
      system_data<-read.csv(filename,sep=";",na.strings = "#N/D",header=FALSE,stringsAsFactors=FALSE)
      
      where_E<-which(system_data[,1]==Ename)
      where_E2<-which(system_data[,1]==E2name)
      where_L<-which(system_data[,1]==Lname)
      
      header<-as.character(system_data[1,])
      
      E<-system_data[where_E,2:dim(system_data)[2]];
      classes<-lapply(E, class);classes<- as.vector(unlist(classes))
      E<- data.frame(matrix(unlist(E), nrow=length(where_E), byrow=F));E<-t(E);E<-na.omit(E)

      L<-system_data[where_L,2:dim(system_data)[2]];
      L<- data.frame(matrix(unlist(L), nrow=length(where_L), byrow=F));L<-t(L);L<-na.omit(L)
       
      E2<-system_data[where_E2,2:dim(system_data)[2]];
      classes<-lapply(E2, class);classes<- as.vector(unlist(classes))
      E2<- data.frame(matrix(unlist(E2), nrow=length(where_E2), byrow=F));E2<-t(E2);E2<-na.omit(E2)
      
      
      
      
      rawData<-list(E,E2,L);names(rawData)<-c("E","E2","L")
      ##value<< list containing the FloX data from QE spectrometer and ancillary data collected by the system
      return(rawData)
    } 


cal<-read.table('R_wl.csv',header=TRUE,sep=";");cal<-as.data.frame(cal)

coeff_FloX<<-cal


coeff_qe<-data.frame(coeff_FloX$wl);names(coeff_qe)<-c("wl")
wl<-coeff_qe$wl

dat<-ReadDFloX(filename = 'R_script_input.csv')
sv<-rep(-999,dim(dat$E)[2])
iFLD_O2A<-data.frame(TrueRef=sv,Fluo=sv)
iFLD_O2B<-iFLD_O2A

L<-dat$L
E<-dat$E
E2<-dat$E2

print(dim(L))
print(dim(E))
print(dim(E2))
FWHM1=0.3
iFLD_O2A<-iFLD(wl=wl,E,L,fwhm =FWHM1,O2band="A")
iFLD_O2A$Fluo[is.finite(iFLD_O2A$Fluo)==FALSE]<-NA
iFLD_O2B<-iFLD(wl=wl,E,L,fwhm =FWHM1,O2band="B")
iFLD_O2B$Fluo[is.finite(iFLD_O2B$Fluo)==FALSE]<-NA

sfm_FLD_O2A<-sv
sfm_FLD_O2B<-sv
sfm_conv_B<-sv
sfm_conv_A<-sv




for(n in 1: dim(E)[2])
        {
          
          if(is.finite(iFLD_O2A$Fluo[n])==FALSE)
          {
            sfm_FLD_O2A[n]<-NA
            sfm_conv_A[n]<-NA
          }else{
            res<-SFM(wl = wl, L = L[,n],E= E[,n],fluoFG = iFLD_O2A$Fluo[n], O2band ="A", output = "VALUE")
            sfm_FLD_O2A[n]<-res$Results[1]
            sfm_conv_A[n]<-res$convergence
          }
          
          if(is.finite(iFLD_O2B$Fluo[n])==FALSE)
          {
            sfm_FLD_O2B[n]<-NA
            sfm_conv_B[n]<-NA
          }else{
            res<-SFM(wl = wl, L = L[,n],E= E[,n],fluoFG = iFLD_O2B$Fluo[n], O2band ="B", output = "VALUE")
            sfm_FLD_O2B[n]<-res$Results[1]
            sfm_conv_B[n]<-res$convergence
          }

output<-data.frame(iFLD_O2A$Fluo,iFLD_O2B$Fluo,sfm_FLD_O2A,sfm_FLD_O2B,sfm_conv_A,sfm_conv_B)
        names(output)<-c("SIF_A_ifld [mW m-2nm-1sr-1]","SIF_B_ifld [mW m-2nm-1sr-1]","SIF_A_sfm [mW m-2nm-1sr-1]","SIF_B_sfm [mW m-2nm-1sr-1]","A_sfm_conv","B_sfm_conv")
      }
      
outData<-list(wl=wl,inp=dat,E=E,L=L,out=output)

D_out<-cbind(outData$out)
write.table(D_out,file=paste("Fluorescences_",format(Sys.time(), "%Y-%m-%d_%H_%M_%S"),".csv",sep=""),na = "#N/D" ,row.names=FALSE,sep=";")
