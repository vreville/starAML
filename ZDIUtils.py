#    This file is part oAf starAML

#    starAML is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    starAML is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from timeit import default_timer as timer

class mysph(object):
    """ Class to do all extrapolation using spherical harmonics. Basic idea is to precompute and store the needed 
    spherical harmonics that take time to re-calculate"""
    def __init__(self,lmax,theta,phi,ndim=2):
        #self.nl = int((-3+np.sqrt(9+8*len(alpha)))/2.)
        self.nl=lmax
        self.theta = theta ; self.phi = phi
        ## Initialize the coordinates
        self.plm=[]
        self.xx = [] ; self.yy = [] ; self.zz = []
        if ndim==1:
            theta1d=self.theta
            phi1d=self.phi*0.0
        else:
            theta1d=self.theta[:,0]
            phi1d=self.phi[:,0]*0.0

        self.plm.append(ylm(0,0,phi1d,theta1d))
        for l in range(1,self.nl+1):
            for m in range(l+1):
                self.plm.append(ylm(m,l,phi1d,theta1d))
                self.xx.append(xlm(m,l,self.phi,self.theta))
                self.yy.append(ylm(m,l,self.phi,self.theta))
                self.zz.append(zlm(m,l,self.phi,self.theta))

    def pfss3d(self,alpha,rss=5.0,rb=1.0,rsph=1.0):

        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))    
                
        if (rsph < rb):
            rop=rsph
            coeff=1.0
        elif(rsph < rss):
            rop=rsph
            coeff=1.0        
        else:
            rop=rss
            coeff=(rss/rsph)**2

        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)

                myalpha=alpha[ii]*(ell*(rb/rss)**(2*ell+1)*(rop/rb)**(ell-1)+(ell+1)*(rop/rb)**(-ell-2))/(ell*(rb/rss)**(2*ell+1)+(ell+1))
                mybeta=(ell+1)*alpha[ii]*((rb/rss)**(2*ell+1)*(rop/rb)**(ell-1)-(rop/rb)**(-ell-2))/(ell*(rb/rss)**(2*ell+1)+(ell+1))
                tmp = myalpha*self.yy[ii]
                br = br + tmp.real
                tmp = mybeta*self.zz[ii]
                bt = bt + tmp.real
                tmp = mybeta*self.xx[ii]
                bp = bp + tmp.real
                ii = ii + 1 
        br*=coeff
        return br,bt,bp

    def pfss3d_2(self,alpha,rss=5.0,rb=1.0,rsph=1.0):
        
        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))    
        
        NewAlph=np.zeros(len(alpha)+2,dtype=complex)
        NewAlph[1:-1]=alpha[:]
        ii=1
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ll=float(l)
                mm=float(m)

                yy=ylm(m,l,self.phi,self.theta)
                ylp1=ylm(m,l+1,self.phi,self.theta)
                ylm1=ylm(m,l-1,self.phi,self.theta)

                Blm=NewAlph[ii]/(1+ll+ll*rss**(-2*ll-1.))
                Blm1m=NewAlph[ii-1]/(ll+(ll-1.)*rss**(-2*ll+1))
                Blp1m=NewAlph[ii+1]/(2+ll+(ll+2)*rss**(-2*ll-3.))

                Alm=-rss**(-2*ll-1.)*Blm
                Alm1m=-rss**(-2*ll+1)*Blm1m
                Alp1m=-rss**(-2*ll-3.)*Blp1m
                                          
                Rlm=np.sqrt((ll**2-mm**2)/(4*ll**2-1))
                Rlp1m=np.sqrt(((ll+1)**2-mm**2)/(4*(ll+1)**2-1))

                tmp=-(Alm*ll*rsph**(ll-1.)-Blm*(ll+1.0)*rsph**(-ll-2.))*yy
                br=br+tmp.real

                #tmp=-(Rlm*(ll-1.)*(Alm1m*rsph**(ll-1.)+Blm1m*rsph**(-ll))-Rlp1m*(ll+2.)*(Alp1m*rsph**(ll+1.)+Blp1m*rsph**(-ll-2)))*yy
                tmp=-(Alm*rsph**ll+Blm*rsph**(-ll-1.))*(ll*Rlp1m*ylp1-(ll+1.)*Rlm*ylm1)
                bt=bt+tmp.real

                tmp=-(Alm*rsph**(ll)+Blm*rsph**(-ll-1.))*1j*mm*yy
                bp=bp+tmp.real
                ii=ii+1

        bt*=1/rsph/np.sin(self.theta)
        bp*=1/rsph/np.sin(self.theta)

        return br,bt,bp

    def multipolarExpansion(self,alpha,rb=1.0,rsph=1.0):
        br = np.zeros(np.shape(self.theta))
        bt = np.zeros(np.shape(self.theta))
        bp = np.zeros(np.shape(self.theta))    

        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)
                myalpha=alpha[ii]*(rsph/rb)**(-ell-2)
                mybeta=-myalpha
                tmp = myalpha*self.yy[ii]
                br = br + tmp.real
                tmp = mybeta*self.zz[ii]
                bt = bt + tmp.real
                tmp = mybeta*self.xx[ii]
                bp = bp + tmp.real
                ii = ii + 1 
        return br.real,bt.real,bp.real

    def cmp_potential_vector(self,alpha,rb=1.0,rsph=1.0,rss=2.5):
        Ar = np.zeros(np.shape(self.theta))
        At = np.zeros(np.shape(self.theta))
        Ap = np.zeros(np.shape(self.theta))
        ii=0
        for l in range(1,self.nl+1):
            for m in range(l+1):
                ell=float(l)
                blm=alpha[ii]/(1+ell+ell*rss**(-2*ell-1))
                alm=-rss**(-2*ell-1)*blm
                tmp = (-alm/(ell+1)*rsph**(ell)+blm/(ell)*rsph**(-ell-1))*(ell+1)
                At=At+(tmp*self.xx[ii]).real
                Ap=Ap-(tmp*self.zz[ii]).real

                ii = ii + 1
                       
        return Ar,At,Ap


    def Currents(self,alpha,rss=5.0,rsph=1.0,dr=0.01):
        br,bt,bp=self.pfss3d(alpha,rss=rss,rsph=rsph)
        br_p,bt_p,bp_p=self.pfss3d(alpha,rss=rss,rsph=rsph+dr)
        br_m,bt_m,bp_m=self.pfss3d(alpha,rss=rss,rsph=rsph-dr)

        #br,bt,bp=dipole(self.theta,rsph)
        #br_p,bt_p,bp_p=dipole(self.theta,rsph+dr)
        #br_m,bt_m,bp_m=dipole(self.theta,rsph-dr)
                
        dtheta=diff(self.theta,1)
        dphi=diff(self.phi,2)
        st=(np.sin(self.theta[1:-1,1:-1]))

        Jr=1/(rsph*st)*(1/dtheta*(diff(np.sin(self.theta)*bp,1))-(diff(bt,2))/dphi)
        Jt=1/(rsph*st)*(diff(br,2)/dphi)-1/rsph*(bp_p[1:-1,1:-1]*(rsph+dr)-bp_m[1:-1,1:-1]*(rsph-dr))/2/dr
        Jp=1/rsph*((bt_p[1:-1,1:-1]*(rsph+dr)-bt_m[1:-1,1:-1]*(rsph-dr))/2/dr-diff(br,1)/dtheta)

        jj=np.sqrt(Jr**2+Jt**2+Jp**2)
        bb=np.sqrt(br[1:-1,1:-1]**2+bt[1:-1,1:-1]**2+bp[1:-1,1:-1]**2)
        Jpar=(Jr*br[1:-1,1:-1]+Jt*bt[1:-1,1:-1]+Jp*bp[1:-1,1:-1])/bb
        Jperp=jj-Jpar

        return Jr,Jt,Jp,jj,Jpar,Jperp

    def spherical_harmonics_decomposition(self,A,theta,lmax=3,sym=None,silent=True):
        """ Compute the spherical harmonics decomposition, 
        through a scalar product with each Y_l^m."""

        if(sym==None):
            period=1
            costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
            Field=A
        elif(sym=="axis"):
            prints("Field is axisymmetric",silent=silent)
            period=1
            costheta=np.asarray(np.cos(theta),dtype=float)
            Field=np.tile(A,(2*len(theta),1)).T
        else:
            period=int(sym)
            costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
            Field=A

        n_theta,n_phi=Field.shape

        # Compute integration weights:  sin(theta)dtheta
        sintheta=np.sqrt(1-costheta**2)
        weights=np.ones(n_theta)*np.pi/n_theta*sintheta

        # Compute the Fourier transform phi->m
        t1=timer()
        ft=np.zeros(Field.shape,dtype=complex)
        for i in range(0,n_theta):
            ft[i,:]=2*np.pi/(n_phi)*np.fft.fft(Field[i,:])
        dt=timer()-t1
        prints("Fourier transform: {0} s".format(dt),silent=silent)
        B=np.zeros((lmax+1,lmax+1),dtype=complex)

        # Project on the (correctly normalized) Plm
        ii=0
        for l in range(0,lmax+1):
            B[l,0] = np.sum(ft[:,0]*self.plm[ii]*weights)
            ii=ii+1
            if sym=="axis":
                ii=ii+l
            else:
                # The 2-factor accounts for m < l
                for m in range(1,l+1):
                    B[l,m] = 2*np.sum(ft[:,m]*self.plm[ii]*weights)
                    ii=ii+1

        alm=np.array([],dtype=complex)
        all=np.array([],dtype=complex)

        ii=0
        for n in range(1,lmax+1):
            all=np.append(all,B[n,0])
            for m in range(0,n+1):
                alm=np.append(alm,B[n,m])
                ii=ii+1

        return [B,alm,all]

def dipole(theta,r):
    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))

    br=2*np.cos(theta)/r**3
    bt=np.sin(theta)/r**3
    
    return br,bt,bp


def diff(field,dir=1):    
    if(dir==1):
        field1=field[:,1:-1]
        return field1[2:,:]-field1[:-2,:]
    if(dir==2):
        field1=field[1:-1,:]
        return field1[:,2:]-field1[:,:-2]



# Spherical harmonics base
def ylm(m,n,phi,theta):
    ll = float(n) ; mm = float(m)
    if(abs(m) > n):
        return 0.0
    else:
        return sph_harm(m,n,phi,theta)
def xlm(m,n,phi,theta):
    ll = float(n) ; mm = float(m)
    xlm=sph_harm(m,n,phi,theta)*1j*mm/(np.sin(theta)*(ll+1))
    xlm[np.isnan(xlm)]=0.0
    return xlm

def zlm(m,n,phi,theta):
    ll = float(n) ; mm = float(m)
    cc = ((ll+1.-mm) / (ll+1.) )/np.sqrt( ((2.*(ll+1.)+1.0)*(ll+1-mm)) / ((2.*ll+1.0)*(ll+1.+mm)) )
    zlm=(-sph_harm(m,n,phi,theta)*np.cos(theta) + cc*sph_harm(m,n+1,phi,theta))/np.sin(theta)
    zlm[np.isnan(zlm)]=0.0
    return zlm

def zlm2(m,n,phi,theta):
    ll = float(n) ; mm = float(m)
    cc = ((ll+mm) / (ll+1.) )*np.sqrt( ((2.*ll+1.0)*(ll-mm)) / ((2.*ll-1.0)*(ll+mm)) )
    return (ll/(ll+1)*sph_harm(m,n,phi,theta)*np.cos(theta) - cc*ylm(m,n-1,phi,theta))/np.sin(theta)


# Read ZDI map
def read_Bfield(myfile,dir="./"):
    filename = dir+myfile
    f = open(filename,'r')
    tmp = f.readline()
    params = f.readline().split()
    nharms = int(params[0]) ; ncomps = params[1]
    nl = int((-3+np.sqrt(9+8*nharms))/2.)
    alpha = np.zeros(nharms,dtype=complex)
    ii = 0
    for n in range(1,nl+1):
        for m in range(n+1):
            vals = f.readline().split()
            alpha[ii] = complex(float(vals[2]),float(vals[3]))
            ii = ii + 1
    tmp=f.readline()
    beta = np.zeros(nharms,dtype=complex) 
    ii = 0
    for n in range(1,nl+1):
        for m in range(n+1):
            vals = f.readline().split()
            beta[ii] = complex(float(vals[2]),float(vals[3]))
            ii = ii + 1
    tmp=f.readline()
    gamma = np.zeros(nharms,dtype=complex) 
    ii = 0
    for n in range(1,nl+1):
        for m in range(n+1):
            vals = f.readline().split()
            gamma[ii] = complex(float(vals[2]),float(vals[3]))
            ii = ii + 1
    f.close()
    return alpha,beta,gamma


# Potential extrapolation with a Source Surface
def pfss3d(alpha,theta,phi,rss=5.0,rb=1.0,rsph=1.0):

    nharms=len(alpha)
    nl = int((-3+np.sqrt(9+8*nharms))/2.)

    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))    

    if (rsph < 1.0):
        rop=rsph
        coeff=1.0
        qexp=-1.0
    elif (rsph < rss):
        rop=rsph
        coeff=1.0
        qexp=1.0
    else:
        rop=rss
        coeff=(rss/rsph)**2
        qexp=1.0
    
    ii=0
    for l in range(1,nl+1):
        for m in range(l+1):
            xx = xlm(m,l,phi,theta)
            yy = ylm(m,l,phi,theta)
            zz = zlm(m,l,phi,theta)
            ell=float(l)
            myalpha=alpha[ii]*(ell*(rb/rss)**(2*ell+1)*(rop/rb)**(ell-1)+(ell+1)*(rop/rb)**((-ell-2)*qexp))/(ell*(rb/rss)**(2*ell+1)+(ell+1))
            mybeta=(ell+1)*alpha[ii]*((rb/rss)**(2*ell+1)*(rop/rb)**(ell-1)-(rop/rb)**((-ell-2))*qexp)/(ell*(rb/rss)**(2*ell+1)+(ell+1))
            tmp = myalpha*yy
            br = br + tmp.real
            tmp = mybeta*zz
            bt = bt + tmp.real
            tmp = mybeta*xx
            bp = bp + tmp.real
            ii = ii + 1 

    br*=coeff

    return br,bt,bp


# Multipolar expansion of the surface field
def multipolarExpansion(alpha,theta,phi,rb=1.0,rsph=1.0):

    nharms=len(alpha)
    nl = int((-3+np.sqrt(9+8*nharms))/2.)

    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))    

    ii=0
    for l in range(1,nl+1):
        for m in range(l+1):
            xx = xlm(m,l,phi,theta)
            yy = ylm(m,l,phi,theta)
            zz = zlm(m,l,phi,theta)
            ell=float(l)
            myalpha=alpha[ii]*(rsph/rb)**(-ell-2)
            mybeta=-myalpha
            tmp = myalpha*yy
            br = br + tmp.real
            tmp = mybeta*zz
            bt = bt + tmp.real
            tmp = mybeta*xx
            bp = bp + tmp.real
            ii = ii + 1 

    return br,bt,bp


# Reconstruct the components of the magnetic field on the given grid of point theta,phi 
def reconstruct_B(myfile,theta,phi,dir="./",reg=True):

    alpha,beta,gamma = read_Bfield(myfile,dir=dir)
    nharms=len(alpha)
    nl = int((-3+np.sqrt(9+8*nharms))/2.)

    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))
    if (reg):
        ii=0
        for n in range(1,nl+1):
            for m in range(n+1):
                xx = xlm(m,n,phi,theta)
                yy = ylm(m,n,phi,theta)
                zz = zlm(m,n,phi,theta)
                tmp = alpha[ii]*yy
                br = br + tmp.real
                tmp = -beta[ii]*zz - gamma[ii]*xx
                bt = bt + tmp.real
                tmp = -beta[ii]*xx + gamma[ii]*zz
                bp = bp + tmp.real
                ii = ii + 1 
    else:    
        ii=0 ; Nt,Np = np.shape(theta)
        for n in range(1,nl+1):
            for m in range(n+1):
                for i in range(Nt):
                    for j in range(Np):
                        xx = xlm(m,n,[phi[i,j]],[theta[i,j]])
                        yy = ylm(m,n,[phi[i,j]],[theta[i,j]])
                        zz = zlm(m,n,[phi[i,j]],[theta[i,j]])
                        tmp = alpha[ii]*yy
                        br[i,j] = br[i,j] + tmp.real
                        tmp = -beta[ii]*zz - gamma[ii]*xx
                        bt[i,j] = bt[i,j] + tmp.real
                        tmp = -beta[ii]*xx + gamma[ii]*zz
                        bp[i,j] = bp[i,j] + tmp.real
                ii = ii + 1 
                        
    return br,bt,bp

def buildMagField(alpha,beta,gamma,theta,phi,reg=True):

    nharms=len(alpha)
    nl = int((-3+np.sqrt(9+8*nharms))/2.)

    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))
    if (reg):
        ii=0
        for n in range(1,nl+1):
            for m in range(n+1):
                xx = xlm(m,n,phi,theta)
                yy = ylm(m,n,phi,theta)
                zz = zlm(m,n,phi,theta)
                tmp = alpha[ii]*yy
                br = br + tmp.real
                tmp = -beta[ii]*zz - gamma[ii]*xx
                bt = bt + tmp.real
                tmp = -beta[ii]*xx + gamma[ii]*zz
                bp = bp + tmp.real
                ii = ii + 1 
    else:    
        ii=0 ; Nt,Np = np.shape(theta)
        for n in range(1,nl+1):
            for m in range(n+1):
                for i in range(Nt):
                    for j in range(Np):
                        xx = xlm(m,n,[phi[i,j]],[theta[i,j]])
                        yy = ylm(m,n,[phi[i,j]],[theta[i,j]])
                        zz = zlm(m,n,[phi[i,j]],[theta[i,j]])
                        tmp = alpha[ii]*yy
                        br[i,j] = br[i,j] + tmp.real
                        tmp = -beta[ii]*zz - gamma[ii]*xx
                        bt[i,j] = bt[i,j] + tmp.real
                        tmp = -beta[ii]*xx + gamma[ii]*zz
                        bp[i,j] = bp[i,j] + tmp.real
                ii = ii + 1 
                        
    return br,bt,bp


# Compute unsigned magnetic flux
def cmpMagFlux(theta,phi,br,r,abs=True):
    dtheta=np.tile(np.diff(theta[:,0]),(np.shape(phi)[1]-1,1)).transpose()
    dphi=np.tile(np.diff(phi[0,:]),(np.shape(theta)[0]-1,1))

    Opflux=0.5*(br[1:,1:]+br[:-1,:-1])*(np.sin(0.5*(theta[:-1,:-1]+theta[1:,1:])))
    Opflux*=dtheta*dphi

    if abs:
        Opflux=np.abs(Opflux)

    return np.sum(Opflux)*r**2



# Plot projected map of the field
def plot_fields_new(list_bs,theta,phi,psfile=None,vmin=-10.,vmax=10,nc=20,lon=0.0,list_lats=[90,0,-90],tits=None,suptits=None,Units='[G]'):
    fig=plt.figure(figsize=(7,6))
    if suptits == None:
        titles = ['Lat ='+str(v) for v in list_lats]
    else:
        titles = suptits
    nf = len(list_bs)
    nl = len(list_lats)
    for ii,ilat in enumerate(list_lats):
        for iff,ff in enumerate(list_bs):
            ax=fig.add_subplot(nf,nl,1+iff*nl+ii)
            m = Basemap(resolution='c',projection='ortho',lon_0=lon,lat_0=ilat)
            im1 = m.contourf(-phi*180./np.pi,-theta*180/np.pi+90.,ff,list(np.linspace(vmin,vmax,nc)),\
                                  latlon=True,cmap='RdBu_r',extend='both')
            m.drawmapboundary()
            parallels = np.arange(-80.,80,20.)
            m.drawparallels(parallels)
            if (iff == 0):
                ax.set_title(titles[ii])
            if (tits != None) and (ii == 0):
                ax.annotate(tits[iff],xycoords='axes fraction',xy=(0.01,0.01))
            plt.subplots_adjust(hspace=0.05,wspace=0.05)
            if (ii == nl-1):
                ax=plt.gca()
                divider=make_axes_locatable(ax)
                cax = divider.append_axes("right",size="5%",pad=0.1)
                cb=plt.colorbar(cax=cax)
                tick_locator = matplotlib.ticker.MaxNLocator(nbins=3)
                cb.locator = tick_locator
                cb.update_ticks()
                cax.annotate(Units,xycoords='axes fraction',xy=(5.,0.5))
                
    if (psfile != None):
        fig.savefig(psfile,bbox_inches='tight')

def plot_var_fields(list_fs,theta,phi,rad,list_sim,qty_s=r"$B_r$",vmin=-10,vmax=10,psfile=None,nc=20,lon=0.0,lat=0.0,Units='%'):    
    fig=plt.figure(figsize=(7,6))
    titles = ['year ='+str(v) for v in list_sim]
    nf = len(list_fs)
    nr = len(rad)
    var_stats=np.zeros((nf,nr))
    for ii,ifs in enumerate(list_fs):
        for rr,rfs in enumerate(ifs):
            mean=1/(4*np.pi*rad[rr]**2)*cmpMagFlux(theta,phi,rfs,rad[rr])
            varfs=(rfs-mean)/mean*100
            var_stats[ii,rr]=mean**2*(varfs**2).sum()/(rfs.shape[0]*rfs.shape[1])
            ax=fig.add_subplot(nr,nf,1+ii+rr*nf)
            m = Basemap(resolution='c',projection='cyl',lon_0=lon,lat_0=lat)
            im1 = m.contourf(-phi*180./np.pi,-theta*180/np.pi+90.,varfs,list(np.linspace(vmin,vmax,nc)),\
                                  latlon=True,cmap='RdBu_r',extend='both')
            m.drawmapboundary()
            parallels = np.arange(-80.,80,20.)
            m.drawparallels(parallels)
            if (rr == 0):
                ax.set_title(titles[ii])
            #if (ii == 0):
            #    ax.annotate("r={0}\n mean({1}) = {2}".format(rad[rr],qty_s,mean),xycoords='axes fraction',xy=(0.01,0.01))
            plt.subplots_adjust(hspace=0.05,wspace=0.05)

            #Color Bar
            axc=fig.add_axes([0.2,0.2,0.6,0.05])
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cb1 = matplotlib.colorbar.ColorbarBase(axc, cmap='RdBu_r',
                                norm=norm,extend="both",
                                orientation='horizontal')
            cb1.set_label(Units,fontsize=18)

            #if (ii == nf-1):
            #    ax=plt.gca()
            #    divider=make_axes_locatable(ax)
            #    cax = divider.append_axes("right",size="5%",pad=0.1)
            #    cb=plt.colorbar(cax=cax)
            #    tick_locator = matplotlib.ticker.MaxNLocator(nbins=3)
            #    cb.locator = tick_locator
            #    cb.update_ticks()
            #    cax.annotate(Units,xycoords='axes fraction',xy=(5.,0.5))
                
    if (psfile != None):
        fig.savefig(psfile,bbox_inches='tight')
    else:
        plt.show()

    return var_stats

# Power Spectrum
def powerSpectrum(myfile,dir='./'):
    alpha,beta,gamma = read_Bfield(myfile,dir=dir)
    nharms=len(alpha)
    nl = int((-3+np.sqrt(9+8*nharms))/2.)
    
    pow=np.array([])

    ii=0
    for l in range(1,nl+1):
        lpow=0
        for m in range(l+1):
            lpow+=abs(alpha[ii])**2
            ii+=1
        pow=np.append(pow,lpow)

    return pow

#Spherical Harmonics decomposition    
def weights_legendre(costheta):
    n_theta=len(costheta)
    weights=np.zeros(n_theta)
    x=costheta
    sintheta=np.sqrt(1-costheta**2)
    
    Pm1=1.0
    Dm1=0.0
    P=x
    D=1.0
    for n in range(2,n_theta+1):
        rn=1.0/(n*1.0)
        Pm2=Pm1
        Pm1=P
        Dm2=Dm1
        Dm1=D
        P=(2.0-rn)*x*Pm1-(1.0-rn)*Pm2
        D=Dm2+(2.0*n-1.0)*Pm1
    weights=2.0/((1.0-x*x)*D*D)
    # Add the sin(theta) term from dOmega=sin(theta)dtheta dphi
    #weights*=sintheta

    return weights

def prints(string,silent=True):
    if not silent:
        print(string)

def spherical_harmonics_decomposition(A,theta,lmax=3,sym=None,silent=True):
    """ Compute the spherical harmonics decomposition, 
    through a scalar product with each Y_l^m.
    Adapted from M. Miesch and M. DeRosa."""

    if(sym==None):
        period=1
        costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
        Field=A
    elif(sym=="axis"):
        prints("Field is axisymmetric",silent=silent)
        period=1
        costheta=np.asarray(np.cos(theta),dtype=float)
        Field=np.tile(A,(2*len(theta),1)).T
    else:
        period=int(sym)
        costheta=np.asarray(np.cos(theta[:,0]),dtype=float)
        Field=A        

    n_theta,n_phi=Field.shape
    sintheta=np.sqrt(1-costheta**2)

    t1 = timer()
    # Compute Legendre quadrature coefficients to integrate over sin(theta)dtheta
    #weights=weights_legendre(costheta)
    weights=np.ones(n_theta)*np.pi/n_theta*sintheta
    dt = timer()-t1
    prints("Legendre weights computation: {0} s".format(dt),silent=silent)

    # Compute the Fourier transform phi->m
    t2=timer()
    ft=np.zeros(Field.shape,dtype=complex)
    for i in range(0,n_theta):
        ft[i,:]=2*np.pi/(n_phi)*np.fft.fft(Field[i,:])
    dt=timer()-t2
    prints("Fourier transform: {0} s".format(dt),silent=silent)
    
    # Normalization for the spherical harmonics
    t3=timer()
    B=np.zeros((lmax+1,lmax/period+1),dtype=complex)
    N_mm=np.zeros(lmax+1)
    N_mm[0]=1.0/np.sqrt(4.0*np.pi)
    
    for m in range(1,lmax+1):
        N_mm[m]=-N_mm[m-1]*np.sqrt(1+1/(2.0*m))
    dt=timer()-t3
    prints("Compute Normalization: {0} s".format(dt),silent=silent)

    P_lm2=N_mm[0]
    P_lm1=P_lm2*costheta*np.sqrt(3.0)
    B[0,0]=np.sum(ft[:,0]*P_lm2*weights)
    B[1,0]=np.sum(ft[:,0]*P_lm1*weights)

    t4=timer()
    for l in range(2,lmax+1):
        ll=float(l)
        c1=np.sqrt(4.0-1/ll**2)
        c2=-(1.0-1.0/ll)*np.sqrt((2.0*ll+1)/(2*ll-3.0))
        P_l=c1*costheta*P_lm1+c2*P_lm2
        B[l,0]=np.sum(ft[:,0]*P_l*weights)
        
        P_lm2=P_lm1
        P_lm1=P_l

    dt=timer()-t4
    prints("Axisymetric terms: {0} s".format(dt),silent=silent)

    if not sym=="axis":
        ft*=2.0
        old_Pmm=N_mm[0]    
        t5=timer()
        for m in range(1,lmax+1):
            P_lm2=old_Pmm*sintheta*N_mm[m]/N_mm[m-1]
            P_lm1=P_lm2*costheta*np.sqrt(2.0*m+3.0)
            old_Pmm=P_lm2
    
            if(m%period==0):
                B[m,m/period]=np.sum(ft[:,m/period]*P_lm2*weights) #diagonal terms

            if(m < lmax):
                B[m+1,m/period]=np.sum(ft[:,m/period]*P_lm1*weights) #l=m+1 terms

            
            for l in range(m+2,lmax+1):
                ll=float(l)
                mm=float(m)
                c1=np.sqrt((4.0*ll**2-1.0)/(ll**2-mm**2))
                c2=-np.sqrt(((2.0*ll+1.0)*((ll-1)**2-mm**2))/((2.0*ll-3.0)*(ll**2-mm**2)))
                P_l=c1*costheta*P_lm1+c2*P_lm2
                B[l,m/period]=np.sum(ft[:,m/period]*P_l*weights)
                
                P_lm2=P_lm1
                P_lm1=P_l
        dt=timer()-t5
        prints("Other terms {0} s".format(dt),silent=silent)

    alpha=np.array([],dtype=complex)
    axis=np.array([],dtype=complex)
    ii=0
    for n in range(1,lmax+1):
        axis=np.append(axis,B[n,0])
        for m in range(0,n+1):
            alpha=np.append(alpha,B[n,m])
            ii=ii+1

    return [B,alpha,axis]


def createZDImap(alm,mapname,path="./"):
    zdimap=path+mapname
    lmax = int((-3+np.sqrt(9+8*len(alm)))/2.)
    #if not os.path.exists(zdimap):
    if True:
        f=open(zdimap,'w')
        f.write("Br spherical harmonics coeffs\n")
        f.write("{:d} {:d} {:d}\n".format(len(alm),0,0))
        k=0
        for i in range(1,lmax+1):
            for j in range(i+1):
                f.write("{:1d} {:1d} {:+.05e} {:+.05e}\n".format(i,j,alm[k].real,alm[k].imag))
                k=k+1
        f.write("\n")
        for i in range(1,lmax+1):
            for j in range(i+1):
                f.write("{:d} {:d} {:+.05e} {:+.05e}\n".format(i,j,0.,0.))
        f.write("\n")
        for i in range(1,lmax+1):
            for j in range(i+1):
                f.write("{:d} {:d} {:+.05e} {:+.05e}\n".format(i,j,0.,0.))
        f.write("\n")
        f.close()
