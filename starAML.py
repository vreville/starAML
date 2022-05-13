#    starAML is a python class made for easy computation of stellar angular momentum loss.
#    This program applies the method described in R\'eville et al. 2015b, the Astrophysical Journal 814, 99
#    Copyright (C) 2015, Victor R\'eville, Allan Sacha Brun, Antoine Strugarek
#    Laboratoire AIM Paris-Saclay, CEA/DSM/IRFU/SAP

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from subprocess import call
import scipy.interpolate
import PolyWind as pw
import SakuraiUtils as su
import ZDIUtils as zdi

class starAML(object):
    """ Python class for the computation of angular momentum loss as described in R\'eville et al. 2015b"""
    def __init__(self, lmax=15, ngrid=100, verbose=0):
        """"
           Initialize the starAML class
           This pre-load the spherical harmonics coefficients and various constants

           Optional arguments:
                 - lmax:    maximum number of spherical harmonics (default is 15)
                 - ngrid:   number of points in the longitude/latitude grid (default is 100)
                 - verbose: level of verbosity, from 0 (silent) to 3 (very talkative)
        """
        self.lmax=lmax
        self.theta, self.phi = np.mgrid[1e-5:np.pi-1e-5:(ngrid+1)*1j, 0:2*np.pi:(2*ngrid+1)*1j] # (Theta,Phi) Grid
        Grid=np.ones((np.shape(self.theta)[0],np.shape(self.theta)[1]))
        self.gridArea=zdi.cmpMagFlux(self.theta,self.phi,Grid,1.0)
        self.nharm=(lmax+1)*(lmax+2)//2-1
        self.SphHarm = zdi.mysph(self.lmax,self.theta,self.phi)
        # Constants Dictionnary
        self.Constants={'G':6.6726e-8,'Msun':1.9891e33,'Rsun':6.9634e10,'Psun':27.28,'Teff_sun':5778,
                        'mp':1.6726e-24,'kB':1.3807e-16}
        self.verbose=verbose
        
    def StellarParameters(self,Mass,Radius,Period,Teff):
        self.Mass=Mass        # Mass in solar mass
        self.Radius=Radius    # Radius in solar radii
        self.Period=Period    # Period in days
        self.Teff=Teff        # Effective temperature in Kelvin
        if hasattr(self,'Gamma'):
            self.Normalize()
        
    def WindParameters(self,Gamma,Tc,Nc,Bp=1,mapfile=None):
        self.Gamma=Gamma      # Adiabatic index
        self.Tc=Tc            # Coronal base temperature in Kelvin
        self.Nc=Nc            # Coronal base density in cm^-3
        if mapfile != None:
            self.mapfile=mapfile  # Name of the ZDI map file
        else:
            self.mapfile=self.createZDImap(Bp)
        if hasattr(self,'Mass'):
            self.Normalize()

    def cmpWind(self,rmax=100,res=400,sak=False):
        self.Normalize()
        self.readMap()
        self.cmpSurfaceFlux()
        self.cmpPolyWind(rmax=rmax,res=res)
        self.Wind=self.PolytropicWind
        if sak:
            try:
                self.cmpSakuraiWind(rmax=rmax,res=res)
                self.Wind=self.SakuraiWind
            except:
                if self.verbose > 0:
                    print("[Warning] Cannot compute magnetocentrifugal wind solution, going back to polytropic.")

    def updateWind(self,**kwargs):
        """ 
           Update the star and wind parameters, and recompute the wind if needed
           Kwargs can be Mass, Radius, Period, Teff, Tc, Nc, Bp and/or mapfile
           (see StellarParameters and WindParameters routines)
        """
        Mass=kwargs.get('Mass',self.Mass)
        Radius=kwargs.get('Radius',self.Radius)
        Period=kwargs.get('Period',self.Period)
        Teff=kwargs.get('Teff',self.Teff)
        
        self.StellarParameters(Mass,Radius,Period,Teff)
        
        Gamma=kwargs.get('Gamma',self.Gamma)
        Tc=kwargs.get('Tc',self.Tc)
        Nc=kwargs.get('Nc',self.Nc)
        Bp=kwargs.get('Bp',0)
        if Bp != 0:
            mapfile=None
        else:
            mapfile=kwargs.get('mapfile',self.mapfile)
        
        self.WindParameters(Gamma,Tc,Nc,Bp=Bp,mapfile=mapfile)
        if(len(kwargs) != 0):
            self.cmpWind()
            if hasattr(self,'spinDownTime'):
                self.cmpTorque(update=True)

    def cmpTorque(self,update=False):
        if not hasattr(self,"Wind"):
            self.cmpWind()
        if not hasattr(self,"OpFlux") or update:
            self.pressureBalance()
            self.cmpOpenFluxEstimate()            
        if self.verbose > 1:
            self.WindInfo()

        self.cmpTorqueMatt2015()
        self.cmpTorqueReville2015()

    def WindInfo(self):
        if hasattr(self,'cs_vesc'):
            print("="*35)
            print("Normalized Parameters for {}".format(self.mapfile.split('/')[-1].split('.')[0]))
            print("Gamma                   = {:.2f}".format(self.Gamma))
            print("cs_vesc                 = {:.2e}".format(self.cs_vesc))
            print("vrot_vesc               = {:.2e}".format(self.vrot_vesc))
            print("breakup ratio           = {:.2e}".format(self.breakupRatio))
            print("Mag Field Normalization = {:.2e}".format(self.Umag))
        else:
            print("Please specify first the parametrs of the star and wind by using the StellarParameters and WindParameters routines")
        if hasattr(self,'va_vesc'):
            print("va_vesc                 = {:.2e}".format(self.va_vesc))
        if hasattr(self,'Wind'):
            if hasattr(self.Wind,'Mdot'):
                unit_mass = self.Urho*self.Ulen**3
                unit_time = self.Ulen/self.Uvel
                year_norm = 3600.*24.*365.25
                coeff = unit_mass*year_norm/(self.Constants['Msun']*unit_time)
                print("Mdot (polytropic)       = {:.2e} (Msun/yr)".format(self.Wind.Mdot*coeff))
        if hasattr(self,'Rss'):
            print("Rss                     = {:.2f} (Rstar)".format(self.Rss))
        if hasattr(self,'OpFlux'):
            print("Open Flux               = {:.2e} (cgs)".format(self.OpFlux*self.Umag*(self.Radius*self.Constants['Rsun'])**2))
        if hasattr(self,'spinDownTime'):
            print("Spin down time scale    = {:.2e} yr (Reville+ 15)".format(np.abs(self.spinDownTime)/(3600*24*365.25)))
        if hasattr(self,'spt_matt'):
            print("Spin down time scale    = {:.2e} yr (Matt+ 15)".format(np.abs(self.spt_matt)/(3600*24*365.25)))
        
    def createZDImap(self,Bp,path="./"):
        if not os.path.exists(path+"ZDImaps/"):
            call("mkdir -p "+path+"ZDImaps/",shell=True)
        
        zdimap=path+'ZDImaps/dipole_{:.5f}_lmax{:d}.ZDImap'.format(Bp,self.lmax)
        if not os.path.exists(zdimap):
            f=open(zdimap,'w')
            alpha=Bp/np.sqrt(3./4./np.pi)
            f.write("Standard dipole\n")
            f.write("{:d} {:d} {:d}\n".format(self.nharm,3,-3))
            f.write("{:d} {:d} {:e} {:e}\n".format(1,0,alpha,0.))
            f.write("{:d} {:d} {:e} {:e}\n".format(1,1,0.,0.))
            for i in range(2,self.lmax+1):
                for j in range(i+1):
                    f.write("{:d} {:d} {:e} {:e}\n".format(i,j,0.,0.))
            f.write("\n")
            f.write("{:d} {:d} {:e} {:e}\n".format(1,0,0.,0.))
            f.write("{:d} {:d} {:e} {:e}\n".format(1,1,0.,0.))
            for i in range(2,self.lmax+1):
                for j in range(i+1):
                    f.write("{:d} {:d} {:e} {:e}\n".format(i,j,0.,0.))
            f.write("\n")
            f.write("{:d} {:d} {:e} {:e}\n".format(1,0,0.,0.))
            f.write("{:d} {:d} {:e} {:e}\n".format(1,1,0.,0.))
            for i in range(2,self.lmax+1):
                for j in range(i+1):
                    f.write("{:d} {:d} {:e} {:e}\n".format(i,j,0.,0.))
            f.write("\n")
            f.close()
        else:
            if self.verbose > 2:
                print("[Warning] The dummy dipolar ZDI map already exists, reading it ({})".format(zdimap))

        return zdimap

    def tau_c(self,T): 
        return 86400*(314.24*np.exp(-(T/1952.5)-(T/6250.)**18)+0.002) #(seconds, Cranmer & Saar 2011)

    def Normalize(self):
        if self.verbose > 1:
            print("="*30)
            print("Normalized Parameters for {0}".format(self.mapfile.split('/')[-1].split('.')[0]))
            print("Gamma = {0}".format(self.Gamma))
        
        G=self.Constants['G'] # cgs 
        Msun=self.Constants['Msun'] # gram
        Rsun=self.Constants['Rsun'] # cm
        Psun=self.Constants['Psun'] # days
        mp = self.Constants['mp'] # gram
        kB = self.Constants['kB'] # cgs
                
        self.Omega=2*np.pi/(self.Period*24*3600.) # rad/s
        self.Rossby=1/(self.Omega*self.tau_c(self.Teff))
        self.breakupRatio=self.Omega*(self.Radius*Rsun)**(1.5)*(G*self.Mass*Msun)**(-0.5)
        self.vesc=np.sqrt(2*G*self.Mass*Msun/self.Radius/Rsun)
        PonRho=2*kB*self.Tc/mp
        self.cs_vesc=np.sqrt(self.Gamma*PonRho)/self.vesc
        self.vrot_vesc=self.Radius*Rsun*self.Omega/self.vesc
        
        if self.verbose > 1:
            print("cs_vesc = {0}".format(self.cs_vesc))
            print("vrot_vesc = {0}".format(self.vrot_vesc))
            print("breakup ratio = {0}".format(self.breakupRatio))

        # Normalization factors
        self.Rs   = self.Radius*self.Constants['Rsun']
        self.Ms   = self.Mass*self.Constants['Msun']
        self.Ulen = self.Rs
        self.Uvel = np.sqrt(G*self.Ms/self.Rs)
        self.Urho = self.Nc*mp

        self.Umag=self.vesc/np.sqrt(2.)*np.sqrt(4*np.pi*self.Urho) 
        if self.verbose > 1:
            print("Mag Field Normalization = {0}".format(self.Umag))

    def readMap(self,axis=False,diponly=False):
        filename = self.mapfile
        f = open(filename,'r')
        tmp = f.readline()
        params = f.readline().split()
        nharm = int(params[0]) ; ncomps = params[1]
        nl = int((-3+np.sqrt(9+8*nharm))/2.)
        if(nharm > self.nharm):
            raise ValueError("Not enough spherical harmonics precomputed. Please initialize with higher lmax.")

        alpha = np.zeros(self.nharm,dtype=complex)
        alpha_dip = np.zeros(self.nharm,dtype=complex)
        alpha_axis = np.zeros(self.nharm,dtype=complex)
        ii = 0
        for n in range(1,nl+1):
            for m in range(n+1):
                vals = f.readline().split()
                alpha[ii] = complex(float(vals[2]),float(vals[3]))
                if(n==1):
                    alpha_dip[ii]=complex(float(vals[2]),float(vals[3]))
                if(m==0):
                    alpha_axis[ii]=complex(float(vals[2]),float(vals[3]))
                ii = ii + 1
        f.close()
        
        self.partdipole=np.sum(alpha_dip*alpha_dip.conjugate())/np.sum(alpha*alpha.conjugate())
        self.partaxis=np.sum(alpha_axis*alpha_axis.conjugate())/np.sum(alpha*alpha.conjugate())

        if(diponly):
            self.alpha=alpha_dip/self.Umag
        elif(axis):
            self.alpha=alpha_axis/self.Umag
        else:
            self.alpha=alpha/self.Umag

    def cmpSurfaceFlux(self):
        br,bt,bp=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=1.0)
        self.surfaceFlux=zdi.cmpMagFlux(self.theta,self.phi,br,1.0)
        self.va_vesc=self.surfaceFlux/4/np.pi/np.sqrt(2.) # rhostar=1.0
        if self.verbose > 1:
            print("va_vesc = {0}".format(self.va_vesc))
        
    def cmpPolyWind(self,rmax=100,res=400):
        self.PolytropicWind=WindProfile(np.linspace(1,rmax,res),verbose=self.verbose)
        self.PolytropicWind.fromPoly(self.Gamma,self.cs_vesc)
        self.PolytropicWind.checkPoly(self.Gamma)
        if self.verbose > 1:
            print("Mdot (polytropic) = {0} (Normalized Units)".format(self.PolytropicWind.Mdot))

    def cmpSakuraiWind(self,rmax=100,res=2000,to_dir="./",cmp=True):
        self.SakuraiWind=WindProfile(np.linspace(1,rmax,res),verbose=self.verbose)
        self.SakuraiWind.fromSakurai(self.Gamma,self.cs_vesc,self.vrot_vesc,self.va_vesc,self.breakupRatio,to_dir=to_dir,cmp=cmp)
        if(self.SakuraiWind.cmpSak):
            self.SakuraiWind.checkSakurai(self.Gamma)
            if self.verbose > 1:
                print("Mdot (Sakurai)= {0} (Normalized Units)".format(self.SakuraiWind.Mdot))

    def pressureBalanceNR(self,dr=0.1,plot=False):
        hydroPressure=scipy.interpolate.InterpolatedUnivariateSpline(self.Wind.r,self.Wind.pr+self.Wind.pram)
        r0=self.Wind.r[0]
        if(plot):
            rad=np.linspace(1.0,10.0,15)
            f=np.zeros(len(rad))
            for ii,rr in enumerate(rad):
                br,bt,bp=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=rr)
                bb=np.sqrt(br**2+bt**2+bp**2)
                magpr=zdi.cmpMagFlux(self.theta,self.phi,bb**2/2.,rr)/self.gridArea/(rr)**2
                f[ii]=magpr-hydroPressure(rr)
                
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(rad,f)
            ax.set_yscale('log')
            plt.show()

        # Starting Newton - Raphson method
        if self.verbose > 2:
            print("Starting Newton-Raphson method for r_ss computation")
               
        br_0,bt_0,bp_0=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=r0)
        br_h,bt_h,bp_h=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=r0+dr)

        bb0=np.sqrt(br_0**2+bt_0**2+bp_0**2)
        bbh=np.sqrt(br_h**2+bt_h**2+bp_h**2)

        magpr0=zdi.cmpMagFlux(self.theta,self.phi,bb0**2/2.,r0)/self.gridArea/r0**2
        magprh=zdi.cmpMagFlux(self.theta,self.phi,bbh**2/2.,r0+dr)/self.gridArea/(r0+dr)**2
        
        ObjFunc=magpr0-hydroPressure(r0)
        slope=(magprh-hydroPressure(r0+dr)-ObjFunc)/dr
        rn=r0
        if self.verbose > 2:
            print("Slope 0 {0}".format(slope))
        if (ObjFunc<0):
            print("Mag Field is too weak to apply this method")
            print("Pressure balance (r0 = {0}) ={1}".format(r0,diff0))
        else:
            while(np.abs(ObjFunc) > 1e-12):
                
                rn=rn-ObjFunc/slope
                br_n,bt_n,bp_n=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=rn)
                bbn=np.sqrt(br_n**2+bt_n**2+bp_n**2)
                magprn=zdi.cmpMagFlux(self.theta,self.phi,bbn**2/2.,rn)/self.gridArea/rn**2
                ObjFunc=magprn-hydroPressure(rn)
                
                dr=dr
                br_h,bt_h,bp_h=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=rn+dr)
                bbh=np.sqrt(br_h**2+bt_h**2+bp_h**2)
                magprh=zdi.cmpMagFlux(self.theta,self.phi,bbh**2/2.,rn+dr)/self.gridArea/(rn+dr)**2
                slope=(magprh-hydroPressure(rn+dr)-ObjFunc)/dr
                if self.verbose > 2:
                    print("rn = {0}, dr = {1}, pressure diff = {2}, slope = {3}".format(rn,dr,ObjFunc,slope))

        try:
            self.Rss=rn
        except NameError:
            self.Rss=2.0
        if self.verbose > 1:
            print("Rss = {:.2f}".format(self.Rss))

    def pressureBalance(self):
        start=timer()
        hydroPressure=scipy.interpolate.InterpolatedUnivariateSpline(self.Wind.r,self.Wind.pr+self.Wind.pram)
        rl=self.Wind.r[0]
        rh=self.Wind.r[-1]
        # Starting bisection method
        if self.verbose > 2:
            print("Starting bisection method for r_ss computation")
        epsilon=rh-rl
        br_l,bt_l,bp_l=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=rl)
        br_h,bt_h,bp_h=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=rh)

        bbl=np.sqrt(br_l**2+bt_l**2+bp_l**2)
        bbh=np.sqrt(br_h**2+bt_h**2+bp_h**2)

        magprl=zdi.cmpMagFlux(self.theta,self.phi,bbl**2/2.,rl)/self.gridArea/rl**2
        magprh=zdi.cmpMagFlux(self.theta,self.phi,bbh**2/2.,rh)/self.gridArea/rh**2
        
        diffLow=magprl-hydroPressure(rl)
        diffHigh=magprh-hydroPressure(rh)
        
        if (not((diffLow>0) and (diffHigh<0))):
            print("Warning bad boundaries for the bisection method")
            print("Pressure balance (rl={0}) ={1}".format(rl,magprl-hydroPressure(rl)))
            print("Pressure balance (rh={0}) ={1}".format(rh,magprh-hydroPressure(rh)))
        else:
            while(epsilon > 0.05):
                a=np.abs(np.log10(np.abs(diffHigh)))
                b=np.abs(np.log10(np.abs(diffLow)))
                rm=(a*rl+b*rh)/(a+b)
                br_m,bt_m,bp_m=self.SphHarm.multipolarExpansion(self.alpha,rb=1.0,rsph=rm)
                bbm=np.sqrt(br_m**2+bt_m**2+bp_m**2)

                magprm=zdi.cmpMagFlux(self.theta,self.phi,bbm**2/2.,rm)/self.gridArea/rm**2
                diffMiddle=magprm-hydroPressure(rm)
                if (diffMiddle>0):
                    rl=rm
                    diffLow=diffMiddle                   
                else:
                    rh=rm
                    diffHigh=diffMiddle

                epsilon=rh-rl
                if self.verbose > 2:
                    print("low r = {0}, high r = {1}, pressure diff = {2}".format(rl,rh,magprm-hydroPressure(rm)))

        try:
            self.Rss=rm
        except NameError:
            self.Rss=1.0
        if self.verbose > 1:
            print("Rss = {:.2f}".format(self.Rss))

    def pressureBalRamMag(self,print_timings=False):
        start=timer()
        hydroPressure=scipy.interpolate.InterpolatedUnivariateSpline(self.Wind.r,self.Wind.pram)
        if (print_timings):
            dt = timer() - start ; print(">>time>> Wind_profile: {.2f} s".format(dt)) ; start=timer()
        
        rl=self.Wind.r[0]
        rh=self.Wind.r[-1]
        # Starting bisection method
        print("Starting bisection method for r_ss computation")
        epsilon=rh-rl
        br_l,bt_l,bp_l=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=rl)
        br_h,bt_h,bp_h=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=rh)
        if (print_timings):
            dt = timer() - start ; print(">>time>> multip_expansion: {.2f} s".format(dt)) ; start=timer()

        bbl=np.sqrt(br_l**2+bt_l**2+bp_l**2)
        bbh=np.sqrt(br_h**2+bt_h**2+bp_h**2)

        magprl=zdi.cmpMagFlux(self.theta,self.phi,bbl**2/2.,rl)/self.gridArea/rl**2
        magprh=zdi.cmpMagFlux(self.theta,self.phi,bbh**2/2.,rh)/self.gridArea/rh**2
        
        diffLow=magprl-hydroPressure(rl)
        diffHigh=magprh-hydroPressure(rh)
        
        if (not((diffLow>0) and (diffHigh<0))):
            print("Warning bad boundaries for the bisection method")
            print("Pressure balance (rl={0}) ={1}".format(rl,magprl-hydroPressure(rl)))
            print("Pressure balance (rh={0}) ={1}".format(rh,magprh-hydroPressure(rh)))
        else:
            while(epsilon > 0.05):
                a=np.abs(np.log10(np.abs(diffHigh)))
                b=np.abs(np.log10(np.abs(diffLow)))
                rm=(a*rl+b*rh)/(a+b)

                br_m,bt_m,bp_m=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=rm)
                if (print_timings):
                    dt = timer() - start ; print(">>time>> multip expansion: {.2f} s".format(dt)) ; start=timer()
                bbm=np.sqrt(br_m**2+bt_m**2+bp_m**2)

                magprm=zdi.cmpMagFlux(self.theta,self.phi,bbm**2/2.,rm)/self.gridArea/rm**2
                if (print_timings):
                    dt = timer() - start ; print(">>time>> cmpmagflux: {.2f} s".format(dt)) ; start=timer()
                diffMiddle=magprm-hydroPressure(rm)
                if (print_timings):
                    dt = timer() - start ; print(">>time>> hydro pressure: {.2f} s".format(dt)) ; start=timer()
                if (diffMiddle>0):
                    rl=rm
                    diffLow=diffMiddle                   
                else:
                    rh=rm
                    diffHigh=diffMiddle

                epsilon=rh-rl
                print("low r = {0}, high r = {1}, pressure diff = {2}".format(rl,rh,magprm-hydroPressure(rm)))

        print("Eq radius = {}".format((rh+rl)/2.))
        
    def cmpOpenFluxEstimate(self):
        start = timer()
        br_ex,bp_ex,bt_ex=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=self.Rss)
        self.OpFlux=zdi.cmpMagFlux(self.theta,self.phi,br_ex,self.Rss)

    def cmpTorqueMatt2015(self):
        Teff_sun=self.Constants['Teff_sun'] #K 
        Psun=self.Constants['Psun'] # days
        Msun=self.Constants['Msun'] #gram  
        Rsun=self.Constants['Rsun'] #cm 
        Om_sun=2*np.pi/(Psun*24*3600)

        T0=9.5e30*self.Radius**3.1*self.Mass**0.5 #(erg)
        tau_c = lambda T: 314.24*np.exp(-(T/1952.5)-(T/6250.)**18)+0.002 #(days)

        Rossby=1/(self.Omega*tau_c(self.Teff))
        Rossby_sun=1/(Om_sun*tau_c(Teff_sun))
        p=2
        chi=10
        MI=0.1*self.Mass*Msun*(self.Radius*Rsun)**2 # Baraffe et al.
        
        if self.verbose > 0:
            print("="*30)
            print("Matt et al. 2015")
            print("="*30)

        if (Rossby_sun/Rossby >= chi):
            if self.verbose > 0:
                print("Saturated regime")
            self.saturated=1
            Torque=-T0*chi**p*self.Omega/Om_sun
            self.spt_matt=MI*self.Omega/Torque/chi**p
        else:
            if self.verbose > 0:
                print("Unsaturated regime")
            self.saturated=0
            Torque=-T0*(Rossby_sun/Rossby)**p*(self.Omega/Om_sun)
            self.spt_matt=MI*self.Omega/Torque/p*(tau_c(Teff_sun)/tau_c(self.Teff))**p

        self.TorqueMatt2015=Torque
        if self.verbose > 0:
            print("Torque = {0} (ergs)".format(Torque))
            print("Spin down time scale = {0} yr".format(np.abs(self.spt_matt)/(3600*24*365.25)))

    def cmpTorqueReville2015(self):
        Teff_sun=self.Constants['Teff_sun'] #K 
        Psun=self.Constants['Psun'] # days
        Msun=self.Constants['Msun'] #gram  
        Rsun=self.Constants['Rsun'] #cm 
        Om_sun=2*np.pi/(Psun*24*3600)        

        m=0.31
        K3=1.2
        K4=0.06

        MI=0.1*self.Mass*Msun*(self.Radius*Rsun)**2 # Baraffe et al.
        tau_c = lambda T: 314.24*np.exp(-(T/1952.5)-(T/6250.)**18)+0.002 #(days)
        p=2
        chi=10
        unit_rho=self.Nc*1.67e-24
        unit_v=self.vesc/np.sqrt(2.)
        unit_length=self.Radius*Rsun
        unit_b=self.Umag
        unit_mass=unit_rho*unit_length**3
        unit_time = unit_length/unit_v
        year_norm = 3600.*24.*365.25
        coeff = unit_mass*year_norm/(Msun*unit_time)

        if self.verbose > 0:
            print("="*30)
            print("Reville et al. 2015")
            print("="*30)

        Mdot_cgs=self.Wind.Mdot*unit_rho*unit_v*unit_length**2
        Mdot_norm=self.Wind.Mdot        
        if self.verbose > 0:
            print("Mdot = {0} (Msun/yr), {1} (cgs)".format(self.Wind.Mdot*coeff, Mdot_cgs))

        self.rA_rs=K3*(self.OpFlux**2/(1+self.breakupRatio**2/K4**2)**(0.5)/Mdot_norm/np.sqrt(2.))**(0.31)
        if self.verbose > 0:
            print("rA/r* = {0}".format(self.rA_rs))

        self.TorqueReville2015=Mdot_cgs*2*np.pi/(self.Period*24*3600)*(self.rA_rs*(self.Radius*Rsun))**2
        m=0.31
        K3=0.55
        self.OpFlux_cgs=self.OpFlux*self.Umag*(self.Radius*Rsun)**2
        self.TorqueReville2=Mdot_cgs**(1-2*m)*self.Omega*(self.Radius*Rsun)**(2-4*m)*K3**2*(self.OpFlux_cgs**2/self.vesc)**(2*m)
        self.spinDownTime=MI*self.Omega/self.TorqueReville2015#/p*(tau_c(Teff_sun)/tau_c(self.Teff))**p
        if self.verbose > 0:
            print("Surface Flux = {0} (cgs)".format(self.surfaceFlux*unit_b*unit_length**2))
            print("Open Flux = {0} (cgs)".format(self.OpFlux*unit_b*unit_length**2))
            print("Tau_w = {0} (ergs)".format(self.TorqueReville2015))
            print("Spin down time scale = {0} yr".format(np.abs(self.spinDownTime)/(3600*24*365.25)))

    def cmpTorqueFromRa(self,rA,Mdot):
        Msun=self.Constants['Msun'] #gram
        Rsun=self.Constants['Rsun'] #cm 
        Mdot_cgs=Mdot*self.Urho*self.Uvel*self.Ulen**2
        return [Mdot_cgs*2*np.pi/(self.Period*24*3600)*(rA*self.Radius*Rsun)**2,Mdot_cgs]

    def recompAlfRad(self):
        # Recompute Rss with the new wind profile
        rl=self.Wind.r[0]
        rh=self.Wind.r[-1]
        WindProfile=self.Wind
        self.pressureBalance()
        
        # Profile interpolation
        iSqRam=scipy.interpolate.InterpolatedUnivariateSpline(WindProfile.r,np.sqrt(WindProfile.rho)*WindProfile.v)
                
        # Starting bisection method
        if self.verbose > 2:
            print("============================================")
            print("Starting bisection method for rA computation")
        epsilon=rh-rl
        br_l,bt_l,bp_l=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=rl)
        br_h,bt_h,bp_h=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=rh)

        bpl=np.sqrt(br_l**2+bt_l**2)
        bph=np.sqrt(br_h**2+bt_h**2)

        poloidalFieldAverageLow=zdi.cmpMagFlux(self.theta,self.phi,bpl,rl)/self.gridArea/rl**2
        poloidalFieldAverageHigh=zdi.cmpMagFlux(self.theta,self.phi,bph,rh)/self.gridArea/rh**2

        if (not((poloidalFieldAverageLow-iSqRam(rl)>0.) and (poloidalFieldAverageHigh-iSqRam(rh)<0.))):
            print("Warning bad boundaries for the bisection method")
            print("Pressure balance (rl={0}) ={1}".format(rl,poloidalFieldAverageLow-iSqRam(rl)))
            print("Pressure balance (rh={0}) ={1}".format(rh,poloidalFieldAverageHigh-iSqRam(rh)))
            rm=rh
        else:
            while(epsilon > 0.05):
                rm=(rl+rh)/2.
                br_m,bt_m,bp_m=self.SphHarm.pfss3d(self.alpha,rss=self.Rss,rb=1.0,rsph=rm)
                bpm=np.sqrt(br_m**2+bt_m**2)
                poloidalFieldAverage=zdi.cmpMagFlux(self.theta,self.phi,bpm,rm)/self.gridArea/rm**2
                meanField=np.mean(bpm)
                if self.verbose > 1:
                    print("Mean Field = {0}, {1}, sqrt(rho)*v = {2} at r= {3}".format(poloidalFieldAverage,meanField,iSqRam(rm),rm))
                if (poloidalFieldAverage-iSqRam(rm)>0):
                    rl=rm
                else:
                    rh=rm
        
                epsilon=rh-rl
                if self.verbose > 1:
                    print("low r = {0}, high r = {1}, pressure diff = {2}".format(rl,rh,poloidalFieldAverage-iSqRam(rm)))

        if self.verbose > 0:
            print("Recomputed Alfv\'en radius = {0}".format(rm))
            print("Torque = {0}".format(self.cmpTorqueFromRa(rm,WindProfile.Mdot)))
        return rm

    def cmpRssOpt(self,OpenFlux,eps=0.05):
        rl=self.Wind.r[0]
        rh=self.Wind.r[-1]
        if self.verbose > 2:
            print("=====================================================")
            print("Starting bisection method for Optimal Rss computation")
        epsilon=rh-rl
        br_l,bt_l,bp_l=self.SphHarm.pfss3d(self.alpha,rss=rl,rb=1.0,rsph=rl)
        br_h,bt_h,bp_h=self.SphHarm.pfss3d(self.alpha,rss=rh,rb=1.0,rsph=rh)

        opfl=zdi.cmpMagFlux(self.theta,self.phi,br_l,rl)
        opfh=zdi.cmpMagFlux(self.theta,self.phi,br_h,rh)
        
        if (not((opfl-OpenFlux>0) and (opfh-OpenFlux<0))):
            print("Warning bad boundaries for the bisection method")
            print("Flux difference (rl={0}) ={1}".format(rl,opfl-OpenFlux))
            print("Flux difference (rh={0}) ={1}".format(rh,opfh-OpenFlux))
        else:
            while(epsilon > eps):
                rm=(rl+rh)/2.
                br_m,bt_m,bp_m=self.SphHarm.pfss3d(self.alpha,rss=rm,rb=1.0,rsph=rm)
                opfm=zdi.cmpMagFlux(self.theta,self.phi,br_m,rm)
                if (opfm-OpenFlux>0):
                    rl=rm
                else:
                    rh=rm
        
                epsilon=rh-rl                
        
        try:
            self.OptRss=rm
        except NameError:
            self.OptRss=2.0
        if self.verbose > 0:
            print("Optimal Rss = {0}".format(self.OptRss))

class WindProfile(object):
    def __init__(self,r,verbose=0):
        self.r=r
        self.v=np.zeros(len(r))
        self.rho=np.zeros(len(r))
        self.pr=np.zeros(len(r))
        self.pram=np.zeros(len(r))
        self.Mdot=0.0
        self.verbose=verbose

    def fromPoly(self,Gamma,cs_vesc,wrc=0.0,offset=0.0):
        self.rstar=1.0
        self.rhostar=1.0
        self.pstar=2.*self.rhostar*cs_vesc**2./Gamma
        self.rc=pw.getRc(Gamma,cs_vesc,wrc=wrc)*self.rstar
        self.vc=np.sqrt(self.rstar/(2.*self.rc))
        self.v=pw.polyWind(Gamma,len(self.r),self.r/self.rc)*self.vc
        if(self.r[0]==1.0):
            self.rho=self.rhostar*self.rstar**2*self.v[0]/(self.v*self.r**2)
        else:
            iv=scipy.interpolate.InterpolatedUnivariateSpline(self.r,self.v)
            self.rho=self.rhostar*self.rstar**2*iv(1.0)/(self.v*self.r**2)

        self.pr=self.pstar*(self.rho/self.rhostar)**Gamma
        self.pram=self.rho*self.v**2
        self.Mdot=4*np.pi*self.r[-1]**2*self.rho[-1]*self.v[-1]
        
        if (offset != 0.0):
            self.r=self.r+offset
            self.rc=self.rc+offset

    def get_b(self,va_vesc):
        self.b=va_vesc*np.sqrt(2.)/self.r**2

    def getToroidal(self,Omega,va_vesc):
        if not hasattr(self,"bphi"):
            self.b=va_vesc*np.sqrt(2.)/self.r**2            
            idx_ra=np.where(self.v > self.b/np.sqrt(self.rho))[0][0]
            ra=self.r[idx_ra]
            self.bphi=Omega*(ra**2/self.r-self.r)*(self.v/self.b-self.b/(self.rho*self.v))**(-1.0)
            self.vphi=self.v*self.bphi/self.b+Omega*self.r

    def checkPoly(self,Gamma):
        ics=scipy.interpolate.InterpolatedUnivariateSpline(self.r,np.sqrt(Gamma*self.pr/self.rho))
        iv=scipy.interpolate.InterpolatedUnivariateSpline(self.r,self.v)
        
        diff=abs(ics(self.rc)-iv(self.rc))
        if(diff>=1e-5):            
            print("Warning ! Wrong critical radius for the polytropic solution")
        if self.verbose > 0:
            print("Difference cs-v = {0} at the critical radius {1}".format(diff,self.rc))
        
    def fromSakurai(self,Gamma,cs_vesc,vrot_vesc,va_vesc,breakupRatio,to_dir="./",cmp=True):
        self.rstar=1.0
        self.rhostar=1.0
        self.Omega=vrot_vesc*np.sqrt(2.)
        self.pstar=2.*self.rhostar*cs_vesc**2./Gamma
        self.cmpSak=1
        self.vkep=1.0

        # find matching (Theta, Omega)
        if(Gamma!=1.05):
            print("Warning! Gamma != 1.05, need to remap (Theta,Omega)")
            self.cmpSak=0
        elif(cmp==False):
            self.cmpSak=0
        elif(breakupRatio>0.01):
            try:
                FinalTheta,FinalOmega,xs,ys,xf,yf,Beta,E=su.findThetaOmega(to_dir+'thetaomega.npz',Gamma,self.vkep,cs_vesc,vrot_vesc,va_vesc)
            except ValueError:
                self.cmpSak=0
                print("=============================================================================================")
                print("==========!!! No match for (Theta,Omega), Sakurai solution will not be computed !!!==========")
                print("--> Warning, breakup ratio>=0.01, the magneto-centrifugal effect is probably not negligible !")

        else:
            self.cmpSak=0
            if self.verbose > 1:
                print("================================================================================")
                print("Breakup ratio < 0.01 the Sakurai Solution is not needed and will not be computed")

        # Compute Solution
        if (self.cmpSak):
            FinalSolution=su.SakuraiSolution(Gamma,FinalTheta,FinalOmega,xs,ys,xf,yf,Beta,E)
            x0=[xs,ys,xf,yf,Beta,E]
            xfinal=FinalSolution.NewtonRaphson(x0,1e-10)
            #FinalSolution.solutionCheck()
            self.r_a, self.rho_a=FinalSolution.alfvenParam(self.vkep,cs_vesc,vrot_vesc,self.rstar,self.rhostar)
            self.rs=xs*self.r_a; self.rhos=ys*self.rho_a            
            self.rf=xf*self.r_a; self.rhof=yf*self.rho_a
            if self.verbose > 0:
                print("r_a = {0}".format(self.r_a))
                print("rho_a ={0}".format(self.rho_a))

            x=self.r/self.r_a
            m0=va_vesc*np.sqrt(2.)*self.rstar**2*np.sqrt(self.rho_a)

            self.rho=FinalSolution.rhoProfile(x,self.vkep,cs_vesc,vrot_vesc,self.rstar,self.rhostar)*self.rho_a
            self.v=m0/self.rho/(self.r)**2
            self.b=va_vesc*np.sqrt(2.)/self.r**2

            self.bphi=self.Omega*(self.r_a**2/self.r-self.r)*(self.v/self.b-self.b/(self.rho*self.v))**(-1.0)
            self.vphi=self.v*self.bphi/self.b+self.Omega*self.r
        

            self.pr=self.pstar*(self.rho/self.rhostar)**(Gamma)
            self.pram=self.rho*self.v**2
            self.Mdot=4*np.pi*self.r[-1]**2*self.rho[-1]*self.v[-1]

            self.cs2=Gamma*self.pr/self.rho

    def checkSakurai(self,Gamma):
        iva=scipy.interpolate.InterpolatedUnivariateSpline(self.r,self.b/np.sqrt(self.rho))
        ics=scipy.interpolate.InterpolatedUnivariateSpline(self.r,np.sqrt(Gamma*self.pr/self.rho))
        ivas=scipy.interpolate.InterpolatedUnivariateSpline(self.r,np.sqrt(0.5*(self.b**2+self.bphi**2+Gamma*self.pr)/self.rho-0.5*np.sqrt(((self.b**2+self.bphi**2+Gamma*self.pr)/self.rho)**2-4*Gamma*self.b**2*self.pr/self.rho**2)))
        ivaf=scipy.interpolate.InterpolatedUnivariateSpline(self.r,np.sqrt(0.5*(self.b**2+self.bphi**2+Gamma*self.pr)/self.rho+0.5*np.sqrt(((self.b**2+self.bphi**2+Gamma*self.pr)/self.rho)**2-4*Gamma*self.b**2*self.pr/self.rho**2)))
        iv=scipy.interpolate.InterpolatedUnivariateSpline(self.r,self.v)
        
        diffSlow=abs(ivas(self.rs)-iv(self.rs))
        if(diffSlow>=1e-2):            
            print("Warning ! Wrong velocity at the slow magnetosonic critical point")                
        if self.verbose > 0:
            print("Difference vas-v = {0} at rs = {1}".format(diffSlow,self.rs))        
        self.OmegaEff=1/self.r*(self.vphi-self.v/self.b*self.bphi)

############################################################################



