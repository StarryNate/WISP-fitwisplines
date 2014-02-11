#!/usr/bin/env python
# Version 5.1 in progress...
import os
import distutils
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from distutils.sysconfig import *
import math

def lingauss(x,x0,pars):# pars[0]=linear intercept, pars[1]=linear slope, pars[2]=Amplitude of gaussian, pars[3]=redshift, pars[4]=sigma line width
    return pars[0]+pars[1]*x+math.fabs(pars[2])*np.exp(-1.0*(x-(1.0+pars[3])*x0)**2/(2.0*pars[4]**2))

def hbo3(x,pars):# pars[0]=linear intercept, pars[1]=linear slope, pars[2]=Amplitude Hbeta, pars[3]=redshift, pars[4]=Hbeta width, pars[5]=OIII5007 amp, pars[6]=OIII width
    return pars[0]+pars[1]*x+math.fabs(pars[2])*np.exp(-1.0*(x-(1.0+pars[3])*4861.0)**2/(2.0*pars[4]**2))+math.fabs(pars[5])/3.0*np.exp(-1.0*(x-(1.0+pars[3])*4959.0)**2/(2.0*pars[4]**2))+math.fabs(pars[5])*np.exp(-1.0*(x-(1.0+pars[3])*5007.0)**2/(2.0*pars[4]**2))

def has2(x,pars):# pars[0]=linear intercept, pars[1]=linear slope, pars[2]=Amplitude Halpha, pars[3]=redshift, pars[4]=Halpha width, pars[5]=SII amp, pars[6]=SII width
    return pars[0]+pars[1]*x+math.fabs(pars[2])*np.exp(-1.0*(x-(1.0+pars[3])*6563.0)**2/(2.0*pars[4]**2))+math.fabs(pars[5])*np.exp(-1.0*(x-(1.0+pars[3])*6724.0)**2/(2.0*pars[4]**2))

def o3(x,pars):# pars[0]=linear intercept, pars[1]=linear slope, pars[2]=Amplitude OIII5007, pars[3]=redshift, pars[4]=OIII5007 width, 
    return pars[0]+pars[1]*x+math.fabs(pars[2])/3.0*np.exp(-1.0*(x-(1.0+pars[3])*4959.0)**2/(2.0*pars[4]**2))+math.fabs(pars[2])*np.exp(-1.0*(x-(1.0+pars[3])*5007.0)**2/(2.0*pars[4]**2))

def lgabsresid(params,x,y,ey,xinit):
    resid=[]
    for i in range(len(x)):
        yi=lingauss(x[i],xinit,params)
        resid.append((y[i]-yi)/ey[i])
    return resid

def hbo3resid(params,x,y,ey):
    resid=[]
    for i in range(len(x)):
        yi=hbo3(x[i],params)
        resid.append((y[i]-yi)/ey[i])
    return resid

def has2resid(params,x,y,ey):
    resid=[]
    for i in range(len(x)):
        yi=has2(x[i],params)
        resid.append((y[i]-yi)/ey[i])
    return resid

def o3resid(params,x,y,ey):
    resid=[]
    for i in range(len(x)):
        yi=o3(x[i],params)
        resid.append((y[i]-yi)/ey[i])
    return resid

def geterrparams(covmat,resid):
    cov=covmat
    mu=0.0
    for i in range(len(resid)):
        mu=mu+resid[i]
    sig2=0.0
    for i in range(len(resid)):
        sig2=sig2+(resid[i]-mu)**2
    sig2=sig2/(float(len(resid))-1.0)
    dimen=cov.shape
    for p in range(dimen[0]):
        for q in range(dimen[1]):
            cov[p][q]=covmat[p][q]*sig2
    return cov

def fluxgauss(amp,wid,sigamp,sigwid,sigampwid):
    amp=math.fabs(amp)
    flux = amp*math.fabs(wid)*math.sqrt(2.0*math.pi)
    sigflux = 2.0*math.pi*((amp)**2*sigwid+(wid)**2*sigamp+2.0*amp*wid*sigampwid)
    sigflux=math.sqrt(math.fabs(sigflux))
    return flux,sigflux

def ewrest(params,cov,restlam,inda,indb,indA,indW,indz):
    params[indW]=math.fabs(params[indW])
    params[indA]=math.fabs(params[indA])
    flux=params[indA]*params[indW]*math.sqrt(2.0*math.pi)
    cont=params[inda]+params[indb]*restlam*(1.0+params[indz])
    EW=flux/(cont*(1.0+params[indz]))
    partA=EW/params[indA]
    partW=EW/params[indW]
    partz=flux/(cont**2 * (1.0+params[indz])**2) * (-1.0*(cont+params[indb]*restlam*(1.0+params[indz])))
    parta=flux/(1.0+params[indz])*-1.0/(cont**2)
    partb=flux*-1.0/(cont**2)*(restlam)
    indices=[inda,indb,indA,indW,indz]
    partials=[parta,partb,partA,partW,partz]
    sigEW=0.0
    for p in range(len(indices)):
        for q in range(len(indices)):
            sigEW=sigEW+partials[p]*partials[q]*cov[p][q]
    sigEW=math.sqrt(math.fabs(sigEW))
    return EW,sigEW

def fwhmrest(W,z,sigW,sigz,sigWz):
    FWHM=2.35*math.fabs(W)*(1.0+z)
    partW=2.35*(1.0+z)
    partz=2.35*math.fabs(W)
    sigFWHM=partW**2*sigW+partz**2*sigz+2.0*partW*partz*sigWz
    sigFWHM=math.sqrt(math.fabs(sigFWHM))
    return FWHM,sigFWHM

def fithbo3(zinit,ll,lf,lfe):
    pguess=[3.0e-17,0.0,1.0e-16,zinit,75.0,1.0e-16]
    bestparams, covpars, info, mess, ier = leastsq(hbo3resid,pguess,args=(ll,lf,lfe),Dfun=None,full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
    if covpars==None or (ier!=1 and ier!=2 and ier!=3 and ier!=4):
        print "FIT FAILED"
        return [0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,bestparams]
    covparams=geterrparams(covpars,hbo3resid(bestparams,ll,lf,lfe))
    fluxHb,fluxHberr=fluxgauss(bestparams[2],bestparams[4],covparams[2][2],covparams[4][4],covparams[2][4])
    fluxO3,fluxO3err=fluxgauss(bestparams[5],bestparams[4],covparams[5][5],covparams[4][4],covparams[5][4])
    fluxO3 = fluxO3*(4.0/3.0)
    fluxO3err=fluxO3err*(4.0/3.0)
    fwhmhb,fwhmhberr=fwhmrest(bestparams[4],bestparams[3],covparams[4][4],covparams[3][3],covparams[4][3])
    fwhmO3,fwhmO3err=fwhmrest(bestparams[4],bestparams[3],covparams[4][4],covparams[3][3],covparams[4][3])
    ewhb,ewhberr=ewrest(bestparams,covparams,4861.0,0,1,2,4,3)
    ewO3,ewO3err=ewrest(bestparams,covparams,5007.0,0,1,5,4,3)
    ewO3=ewO3*(4.0/3.0)
    ewO3err=ewO3err*(4.0/3.0)
    z=bestparams[3]
    zerr=math.sqrt(math.fabs(covparams[3][3]))
    return [fluxHb,fluxHberr,fluxO3,fluxO3err,fwhmhb,fwhmhberr,fwhmO3,fwhmO3err,z,zerr,ewhb,ewhberr,ewO3,ewO3err,bestparams]

def fithas2(zinit,ll,lf,lfe):
    pguess=[3.0e-17,0.0,1.0e-16,zinit,75.0,1.0e-16]
    bestparams, covpars, info, mess, ier = leastsq(has2resid,pguess,args=(ll,lf,lfe),Dfun=None,full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
    if covpars==None or (ier!=1 and ier!=2 and ier!=3 and ier!=4):
        print "FIT FAILED"
        return [0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,bestparams]
    covparams=geterrparams(covpars,has2resid(bestparams,ll,lf,lfe))
    fluxHa,fluxHaerr=fluxgauss(bestparams[2],bestparams[4],covparams[2][2],covparams[4][4],covparams[2][4])
    fluxS2,fluxS2err=fluxgauss(bestparams[5],bestparams[4],covparams[5][5],covparams[4][4],covparams[5][4])
    fwhmha,fwhmhaerr=fwhmrest(bestparams[4],bestparams[3],covparams[4][4],covparams[3][3],covparams[4][3])
    fwhmS2,fwhmS2err=fwhmrest(bestparams[4],bestparams[3],covparams[4][4],covparams[3][3],covparams[4][3])
    ewha,ewhaerr=ewrest(bestparams,covparams,6563.0,0,1,2,4,3)
    ewS2,ewS2err=ewrest(bestparams,covparams,6724.0,0,1,5,4,3)
    z=bestparams[3]
    zerr=math.sqrt(math.fabs(covparams[3][3]))
    return [fluxHa,fluxHaerr,fluxS2,fluxS2err,fwhmha,fwhmhaerr,fwhmS2,fwhmS2err,z,zerr,ewha,ewhaerr,ewS2,ewS2err,bestparams]

def fito3(zinit,ll,lf,lfe):
    pguess=[3.0e-17,0.0,1.0e-16,zinit,75.0]
    bestparams, covpars, info, mess, ier = leastsq(o3resid,pguess,args=(ll,lf,lfe),Dfun=None,full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
    if covpars==None or (ier!=1 and ier!=2 and ier!=3 and ier!=4):
        print "FIT FAILED"
        return [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,bestparams]
    covparams=geterrparams(covpars,o3resid(bestparams,ll,lf,lfe))
    fluxO3,fluxO3err=fluxgauss(bestparams[2],bestparams[4],covparams[2][2],covparams[4][4],covparams[2][4])
    fluxO3 = fluxO3*(4.0/3.0)
    fluxO3err=fluxO3err*(4.0/3.0)
    fwhmO3,fwhmO3err=fwhmrest(bestparams[4],bestparams[3],covparams[4][4],covparams[3][3],covparams[4][3])
    ewO3,ewO3err=ewrest(bestparams,covparams,5007.0,0,1,2,4,3)
    ewO3=ewO3*(4.0/3.0)
    ewO3err=ewO3err*(4.0/3.0)
    z=bestparams[3]
    zerr=math.sqrt(math.fabs(covparams[3][3]))
    return [fluxO3,fluxO3err,fwhmO3,fwhmO3err,z,zerr,ewO3,ewO3err,bestparams]

def fitsingle(zinit,restlam,ll,lf,lfe):
    pguess=pguess=[3.0e-17,0.0,1.0e-16,zinit,75.0]
    bestparams, covpars, info, mess, ier = leastsq(lgabsresid,pguess,args=(ll,lf,lfe,restlam),Dfun=None,full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
    if covpars==None or (ier!=1 and ier!=2 and ier!=3 and ier!=4):
        print "FIT FAILED"
        return [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,bestparams]
    covparams=geterrparams(covpars,lgabsresid(bestparams,ll,lf,lfe,restlam))
    flux,fluxerr=fluxgauss(bestparams[2],bestparams[4],covparams[2][2],covparams[4][4],covparams[2][4])
    fwhm,fwhmerr=fwhmrest(bestparams[4],bestparams[3],covparams[4][4],covparams[3][3],covparams[4][3])
    ew,ewerr=ewrest(bestparams,covparams,restlam,0,1,2,4,3)
    z=bestparams[3]
    zerr=math.sqrt(math.fabs(covparams[3][3]))
    return [flux,fluxerr,fwhm,fwhmerr,z,zerr,ew,ewerr,bestparams]

def plotthem(full_l,full_f,z,fit_l,fit_f,xr1,xr2,name,yrange=[],g141only=False):
    
    #lines=[1216.0,3727.0,4341.0,4861.0,4959.0,5007.0,6563.0,6724.0,9069.0,9532.0,10830.0,10940.0,12600.0,12810.0]
    lines=[1216.0,3727.0,4861.0,4959.0,5007.0,6563.0,6724.0,9069.0,9532.0,10830.0]

    fulll=np.array(full_l)
    fullf=np.array(full_f)
    
    prange=[min(full_l),max(full_l)]
    if yrange==[] or len(yrange)!=2:
        if g141only==True:
            maskl=np.logical_and(fulll>11000.,fulll<16750.)
        else:
            maskl=np.logical_and(fulll>8000.,fulll<16750.)
        yrange=[min(fullf[maskl]),max(fullf[maskl])]
        if yrange[0]<0:
            yrange[0]=yrange[0]*1.15
        else:
            yrange[0]=yrange[0]*0.85
        if yrange[1]<0:
            yrange[1]=yrange[1]*0.85
        else:
            yrange[1]=yrange[1]*1.15
    
    plt.ion()
    plt.figure(1,figsize=(11,5))
    plt.clf()
    specplot=plt.plot(fulll,fullf,'k', linewidth=1.5)
    if len(fit_l)>0:
        fitl=np.array(fit_l)
        fitf=np.array(fit_f)
        fitplot=plt.plot(fitl,fitf,'b--', linewidth=2.5)
    for emline in lines:
        if emline*(1.0+z)>prange[0] and emline*(1.0+z)<prange[1]:
            plt.axvline(x=emline*(1.0+z),ymin=0,ymax=1,hold=None,color='g',linestyle='-.', linewidth=1.5)
    plt.axvline(x=xr1,ymin=0,ymax=1,hold=None,color='b',linestyle='-.', linewidth=1.5)
    plt.axvline(x=xr2,ymin=0,ymax=1,hold=None,color='b',linestyle='-.', linewidth=1.5)
    plt.xlim(prange)
    plt.ylim(yrange)
    plt.xlabel(r'$\lambda_{obs}$ ($\AA$)',size='xx-large')
    plt.ylabel(r'$F_{\lambda}$ ($ergs/s/cm^2/\AA$)',size='xx-large')
    plt.title(name)
    plt.draw()
    plt.draw()

def findrange(restl,zobs,simulthbo3=0,simulthas2=0):
    halfsize=500.
    if simulthbo3==0 and simulthas2==0:
        ran=[restl*(1.0+zobs)-halfsize,restl*(1.0+zobs)+halfsize]
    elif simulthbo3==1:
        ran=[restl*(1.0+zobs)-halfsize,5007.0*(1.0+zobs)+halfsize]
    elif simulthas2==1:
        ran=[restl*(1.0+zobs)-halfsize,6731.0*(1.0+zobs)+halfsize]
    
    if ran[0]<11300.0 and restl*(1.0+zobs)<11300. and ran[1]>11300.:
        dran=ran[1]-11300.
        ran[0]=ran[0]-dran
        ran[1]=11300.
    elif ran[0]<11550. and restl*(1.0+zobs)>11550.:
        dran=11550.-ran[0]
        ran[0]=11550.
        ran[1]=ran[1]+dran
    elif ran[0]<8000. and restl*(1.0+zobs)>8000.:
        dran=8000.-ran[0]
        ran[0]=8000.
        ran[1]=ran[1]+dran
    elif ran[1]>16700. and restl*(1.0+zobs)<16700.:
        dran=ran[1]-16700.
        ran[0]=ran[0]-dran
        ran[1]=16700.

    return ran

def getinrange(x,y,ey,xmin,xmax):
    xr=[]
    yr=[]
    eyr=[]
    for i in range(len(x)):
        if x[i]>=xmin and x[i]<=xmax:
            xr.append(x[i])
            yr.append(y[i])
            eyr.append(ey[i])
    return xr,yr,eyr

#def get_unique(p,o,z):
def get_unique(p,o,z,f): #MR
    unp=[]
    uno=[]
    unz=[]
    unf=[] #MR
    for i in range(len(p)):
        if i==0 or p[i]!=p[i-1] or o[i]!=o[i-1]:
            unp.append(p[i])
            uno.append(o[i])
            unz.append(z[i])
            unf.append(f[i]) #MR
    return unp,uno,unz,unf #MR
    #return unp,uno,unz

def go(linelist=' ',outfile=' '):
    if linelist==" ":
        allDirectoryFiles=os.listdir('.')
        for files in allDirectoryFiles:
            if files[0:3]=='Par':
                llpts=files.split('_')
                linelist='linelist/'+llpts[0]+'lines_with_redshifts.dat'
                break
    if linelist==" " or os.path.exists(linelist)==0:
        print "Invalid path to line list file: %s" % (linelist)
        return 0
    else:
        print "Found line list file %s" % (linelist)
    if outfile==' ':
        llpts=linelist.split('_')
        outfile=llpts[0]+'_measured.dat'

    #emlines=[1216.0,3727.0,4341.0,4861.0,5007.0,6563.0,6724.0,9069.0,9532.0,10830.0,10940.0,12600.0,12810.0]
    #emlinenames=['Ly_alpha','OII','H_gamma','H_beta','OIII_4959+5007','H_alpha','SII','SIII_9069','SIII_9532','HeI_10830','Paschen_gamma','FeI_12600','Paschen_beta']
    #emlines=[1216.0,3727.0,4341.0,4861.0,5007.0,6563.0,6724.0,9069.0,9532.0,10830.0]
    #emlinenames=['Ly_alpha','OII','H_gamma','H_beta','OIII_4959+5007','H_alpha','SII','SIII_9069','SIII_9532','HeI_10830']
    emlines=[1216.0,3727.0,4861.0,5007.0,6563.0,6724.0,9069.0,9532.0,10830.0]
    emlinenames=['Ly_alpha','OII','H_beta','OIII_4959+5007','H_alpha','SII','SIII_9069','SIII_9532','HeI_10830']
    parids=[]
    objids=[]
    zinits=[]
    flagcont=[] #MR
    print "Reading in %s" % (linelist)
    llin=open(linelist,'r')
    for line in llin:
        if line[0]!='#':
            entries=line.split()
            parids.append(entries[0])
            objids.append(entries[2])
            flagcont.append(entries[7]) #MR
            zinits.append(float(entries[8])) #MR
            #zinits.append(float(entries[7]))
    llin.close()
    #parids, objids, zinits = get_unique(parids,objids,zinits)
    parids, objids, zinits, flagcont = get_unique(parids,objids,zinits,flagcont)#MR

    rootdir='..' #Location of the directory containing WISP data. Note that the Par refers to the beginning of the directory names for each field number i.e. Par1, Par2, etc.
    specdir='/Spectra/'
    catdir='/DATA/DIRECT_GRISM/'
    
    cat110='fin_F110.cat'
    cat140='fin_F140.cat'
    cat160='fin_F160.cat'
    
    outf=open(outfile,'w')
#    print >>outf, "#1  Par ID\n#2  Object ID\n#3  RA [degs]\n#4  Dec [degs]\n#5  J magnitude [99.0 denotes no detection]\n#6  H magnitude [99.0 denotes no detection]"
#    print >>outf, "#7  SE major axis [pix]\n#8  SE minor axis [pix]\n#9  SE rotation angle"
#    print >>outf, "#10 Redshift from line fit\n#11 Redshift error from line fit\n#12 line flux [ergs/s/cm^2]\n#13 line flux error [ergs/s/cm^2]"
#    print >>outf, "#14 Rest Frame FWHM, [Angs]\n#15 Rest Frame FWHM error [Angs]\n#16 Rest-frame EW [Angs]\n#17 Rest-frame EW err [Angs]\n#18 Line ID"
    print >>outf, "#1  Par ID\n#2  Object ID\n#3  Contamination Flag [1=no, 2=yes]\n#4  RA [degs]\n#5  Dec [degs]\n#6  J magnitude [99.0 denotes no detection]\n#7  H magnitude [99.0 denotes no detection]" #MR
    print >>outf, "#8  SE major axis [pix]\n#9  SE minor axis [pix]\n#10  SE rotation angle"
    print >>outf, "#11 Redshift from line fit\n#12 Redshift error from line fit\n#13 line flux [ergs/s/cm^2]\n#14 line flux error [ergs/s/cm^2]"
    print >>outf, "#15 Rest Frame FWHM, [Angs]\n#16 Rest Frame FWHM error [Angs]\n#17 Rest-frame EW [Angs]\n#18 Rest-frame EW err [Angs]\n#19 Line ID"

    for j in range(len(parids)):
        specpath=rootdir+specdir+'Par'+parids[j]+'_BEAM_'+objids[j]+'A.dat'
        specnameg102='Par' + parids[j] + '_G102_BEAM_' + str(objids[j]) + 'A.dat'
        specnameg141='Par' + parids[j] + '_G141_BEAM_' + str(objids[j]) + 'A.dat'
        if os.path.exists(specnameg102)==0:
            specpath=specnameg141
        cat110path=rootdir+catdir+cat110
        cat140path=rootdir+catdir+cat140
        cat160path=rootdir+catdir+cat160
        
        Jmag=99.0
        Hmag=99.0
        ra=0.0
        dec=0.0
        majaxis=0.0
        minaxis=0.0
        angle=0.0
        print "Progress: %.1f percent..." % (float(j)/float(len(parids))*100.0)
        if os.path.exists(cat110path)==1:
            print "Reading in J catalog"
            catin=open(cat110path,'r')
            for line in catin:
                if line[0]!='#':
                    entries=line.split()
                    if entries[1]==objids[j]:
                        ra=float(entries[7])
                        dec=float(entries[8])
                        Jmag=float(entries[12])
                        majaxis=float(entries[4])
                        minaxis=float(entries[5])
                        angle=float(entries[6])
                        break
            catin.close()
        if os.path.exists(cat140path)==1:
            print "Reading in H catalog"
            catin=open(cat140path,'r')
            for line in catin:
                if line[0]!='#':
                    entries=line.split()
                    if entries[1]==objids[j]:
                        if ra==0.0 or dec==0.0:
                            ra=float(entries[7])
                            dec=float(entries[8])
                            majaxis=float(entries[4])
                            minaxis=float(entries[5])
                            angle=float(entries[6])
                        Hmag=float(entries[12])
                        break
            catin.close()
        elif os.path.exists(cat160path)==1:
            print "Reading in H catalog"
            catin=open(cat160path,'r')
            for line in catin:
                if line[0]!='#':
                    entries=line.split()
                    if entries[1]==objids[j]:
                        if ra==0.0 or dec==0.0:
                            ra=float(entries[7])
                            dec=float(entries[8])
                            majaxis=float(entries[4])
                            minaxis=float(entries[5])
                            angle=float(entries[6])
                        Hmag=float(entries[12])
                        break
            catin.close()
        
        spectruml=[]
        spectrumf=[]
        spectrumferr=[]
        print "Reading in spectrum lines"
        specin=open(specpath,'r')
        for line in specin:
            entries=line.split()
            if len(entries)>=3 and entries[2]!='NaN' and entries[2]!='-NaN' and entries[1]!='NaN' and entries[1]!='-NaN':
                spectruml.append(float(entries[0]))
                spectrumf.append(float(entries[1]))
                spectrumferr.append(float(entries[2]))
        specin.close()
        
        skipS2=0
        skipO3=0
        fullspecrange=[min(spectruml),max(spectruml)]
        if specpath==specnameg141:
            specl2=np.array(spectruml)
            specf2=np.array(spectrumf)
            pltrang=(specl2 > 11000.) & (specl2 < 16700.)
            fullspecrange=[min(specl2[pltrang]),max(specl2[pltrang])]
            isG141only=True
        else:
            isG141only=False

        k=0
        custrange=0
        while k<len(emlines):
            fitlams=[]
            fitfs=[]
            fitefs=[]
            resultf=[]
            if custrange==0:
                frange=findrange(emlines[k],zinits[j])
            if k==0:
                plotthem(spectruml,spectrumf,zinits[j],fitlams,resultf,frange[0],frange[1],specpath,g141only=isG141only)
            if emlines[k]*(1.0+zinits[j])<fullspecrange[0] or emlines[k]*(1.0+zinits[j])>fullspecrange[1]:
                k=k+1
                continue
            
            if (emlinenames[k]!='OIII_4959+5007' and emlinenames[k]!='SII') or (skipO3==0 and emlinenames[k]=='OIII_4959+5007') or (skipS2==0 and emlinenames[k]=='SII'):
                fline=''
            else:
                fline='Y'
            
            while fline!='Y' and fline!='y' and fline!='N' and fline!='n':
                query='Fit '+emlinenames[k]+'? [Y/N] > '
                fline=raw_input(query)
            if fline=='N' or fline=='n':
                k=k+1
                continue
            fdoub=''
            if emlinenames[k]=='H_alpha':
                while fdoub!='Y' and fdoub!='y' and fdoub!='N' and fdoub!='n': 
                    fdoub = raw_input("Fit Ha and SII as a blend? [Y/N] > ") 
                if fdoub=='y' or fdoub=='Y': #Fit Ha & SII as a doublet
                    if custrange==0:
                        frange=findrange(emlines[k],zinits[j],simulthas2=1)
                    fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                    if len(fitlams)<10:
                        print "Cannot fit %s: too few pixels." % (emlinenames[k])
                        k=k+1
                        continue
                    results=fithas2(zinits[j],fitlams,fitfs,fitefs)
                    fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[4],results[5],results[8],results[9],results[10],results[11]
                    skipS2=1
                    fluxS2,efluxS2,fwhmS2,efwhmS2,zfitS2,ezfitS2,ewS2,eewS2=results[2],results[3],results[6],results[7],results[8],results[9],results[12],results[13]
                    fparams=results[-1]
                    for h in range(len(fitlams)):
                        resultf.append(has2(fitlams[h],fparams))
                else:
                    print "Fitting Ha as single line"
                    fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                    if len(fitlams)<10:
                        print "Cannot fit %s: too few pixels." % (emlinenames[k])
                        k=k+1
                        continue
                    results=fitsingle(zinits[j],emlines[k],fitlams,fitfs,fitefs)
                    fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                    fparams=results[-1]
                    for h in range(len(fitlams)):
                        resultf.append(lingauss(fitlams[h],emlines[k],fparams))
            elif emlinenames[k]=='SII' and skipS2==1:
                fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=fluxS2,efluxS2,fwhmS2,efwhmS2,zfitS2,ezfitS2,ewS2,eewS2
            elif emlinenames[k]=='SII' and skipS2==0:
                while fdoub!='Y' and fdoub!='y' and fdoub!='N' and fdoub!='n': 
                    fdoub = raw_input("Fit SII as a single line? [Y/N] > ") 
                if fdoub=='y' or fdoub=='Y':
                    fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                    if len(fitlams)<10:
                        print "Cannot fit %s: too few pixels." % (emlinenames[k])
                        k=k+1
                        continue
                    results=fitsingle(zinits[j],emlines[k],fitlams,fitfs,fitefs)
                    fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                    fparams=results[-1]
                    for h in range(len(fitlams)):
                        resultf.append(lingauss(fitlams[h],emlines[k],fparams))
                else:
                    k=k+1
                    continue
            elif emlinenames[k]=='H_beta':
                while fdoub!='Y' and fdoub!='y' and fdoub!='N' and fdoub!='n': 
                    fdoub = raw_input("Fit Hb and OIII as a blend? [Y/N] > ") 
                if fdoub=='y' or fdoub=='Y': #Fit Hb & OIII as a doublet
                    if custrange==0:
                        frange=findrange(emlines[k],zinits[j],simulthbo3=1)
                    fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                    if len(fitlams)<10:
                        print "Cannot fit %s: too few pixels." % (emlinenames[k])
                        k=k+1
                        continue
                    results=fithbo3(zinits[j],fitlams,fitfs,fitefs)
                    fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[4],results[5],results[8],results[9],results[10],results[11]
                    skipO3=1
                    fluxO3,efluxO3,fwhmO3,efwhmO3,zfitO3,ezfitO3,ewO3,eewO3=results[2],results[3],results[6],results[7],results[8],results[9],results[12],results[13]
                    fparams=results[-1]
                    for h in range(len(fitlams)):
                        resultf.append(hbo3(fitlams[h],fparams))
                else:
                    print "Fitting Hb as single line"
                    fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                    if len(fitlams)<10:
                        print "Cannot fit %s: too few pixels." % (emlinenames[k])
                        k=k+1
                        continue
                    results=fitsingle(zinits[j],emlines[k],fitlams,fitfs,fitefs)
                    fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                    fparams=results[-1]
                    for h in range(len(fitlams)):
                        resultf.append(lingauss(fitlams[h],emlines[k],fparams))
            elif emlinenames[k]=='OIII_4959+5007' and skipO3==1:
                fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=fluxO3,efluxO3,fwhmO3,efwhmO3,zfitO3,ezfitO3,ewO3,eewO3
            elif emlinenames[k]=='OIII_4959+5007' and skipO3==0:
                if fline=='Y' or fline=='y':
                    fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                    if len(fitlams)<10:
                        print "Cannot fit %s: too few pixels." % (emlinenames[k])
                        k=k+1
                        continue
                    results=fito3(zinits[j],fitlams,fitfs,fitefs)
                    fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                    fparams=results[-1]
                    for h in range(len(fitlams)):
                        resultf.append(o3(fitlams[h],fparams))
                else:
                    k=k+1
                    continue
            else:
                print "Fitting %s as a single line..." % (emlinenames[k])
                fitlams,fitfs,fitefs=getinrange(spectruml,spectrumf,spectrumferr,frange[0],frange[1])
                if len(fitlams)<10:
                    print "Cannot fit %s: too few pixels." % (emlinenames[k])
                    k=k+1
                    continue
                results=fitsingle(zinits[j],emlines[k],fitlams,fitfs,fitefs)
                fluxline,efluxline,fwhm,efwhm,zfit,ezfit,ew,eew=results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]
                fparams=results[-1]
                for h in range(len(fitlams)):
                    resultf.append(lingauss(fitlams[h],emlines[k],fparams))
            print fdoub,emlinenames[k]
            if (emlinenames[k]!='OIII_4959+5007' and emlinenames[k]!='SII') or (skipO3==0 and emlinenames[k]=='OIII_4959+5007') or (skipS2==0 and emlinenames[k]=='SII'):
                plotthem(spectruml,spectrumf,zinits[j],fitlams,resultf,frange[0],frange[1],specpath,g141only=isG141only)
            print "Fit S/N for %s is %f" % (emlinenames[k],(fluxline/efluxline))
            fsave=' '
            while len(fsave)==0 or (fsave[0]!='s' and fsave[0]!='r' and fsave[0]!='w'):
                fsave=raw_input("Save line [s], reject line [r], or change fit window [w MIN MAX] > ")
                if len(fsave)>0 and fsave[0]=='w' and fsave!="w default":
                    try:
                        test_ran1=float(fsave.split()[1])
                    except:
                        fsave=' '
                        continue
                    try:
                        test_ran2=float(fsave.split()[2])
                    except:
                        fsave=' '
                        continue
            if fsave=='w default':
                custrange=0
            elif fsave[0]=='w':
                custrange=1
                entries=fsave.split()
                frange=[float(entries[1]),float(entries[2])]
            elif fsave=='s':
 #               print >>outf, "%s\t%s\t%.6f\t%.6f\t%.2f\t%.2f\t%.3f\t%.3f\t%.3f\t%.5f\t%.5f\t%.5e\t%.5e\t%.3f\t%.3f\t%.2f\t%.2f\t%s" % (parids[j],objids[j],ra,dec,Jmag,Hmag,majaxis,minaxis,angle,zfit,ezfit,fluxline,efluxline,fwhm,efwhm,ew,eew,emlinenames[k])
                print >>outf, "%s\t%s\t%s\t%.6f\t%.6f\t%.2f\t%.2f\t%.3f\t%.3f\t%.3f\t%.5f\t%.5f\t%.5e\t%.5e\t%.3f\t%.3f\t%.2f\t%.2f\t%s" % (parids[j],objids[j],flagcont[j],ra,dec,Jmag,Hmag,majaxis,minaxis,angle,zfit,ezfit,fluxline,efluxline,fwhm,efwhm,ew,eew,emlinenames[k])#MR
                k=k+1
                custrange=0
            elif fsave=='r':
                k=k+1
                custrange=0
    outf.close()
