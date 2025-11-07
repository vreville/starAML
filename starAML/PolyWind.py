#    This file is part of starAML

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

import numpy as np

# Critical radius for polytropic wind
def getRc(Gamma,cs_vesc,wrc=0.0):
    """Compute the critical radius for polytropic wind"""
    # BLA BLA
    if (Gamma >= 1+2*cs_vesc**2):
        print("Gamma is too high for the coronal sound speed value")
        return 1000
    elif (cs_vesc > 0.5):
        print("Tempeature is too high, critical radius is below the surface")
        return 1000
    else:
        a=1/4./cs_vesc**2
        if(wrc != 0.0):
            x=1.0
        else:
            x=0.001
        xref=10.
        foo = lambda x: (a*x**(-2.*Gamma+3.))**((Gamma+1.)/(Gamma-1.))-(a*x**(-2.*Gamma+3.))*(4./x+(5.-3.*Gamma)/(Gamma-1.))+2./(Gamma-1.)*x**(2.-2.*Gamma)
        f=foo(x)
        while (abs(f) >= 1e-5):
            fprime = a**((Gamma+1.)/(Gamma-1.))*(3.-2.*Gamma)*(Gamma+1.)/(Gamma-1.)*1./x**((2.*Gamma-3.)*(Gamma+1.)/(Gamma-1.)+1)+(a*1./x**(2.*Gamma-3.))*((4./x+(5.-3.*Gamma)/(Gamma-1.))*(2.*Gamma-3.)*1./x+4./x**2.)-4.*x**(1.-2.*Gamma)

            xref=x    
            x=x-f/fprime

            while (x <= 0):
                x=(xref+x)/2
            f=foo(x)
        return 1./x

# Polytropic velocity profile
def polyWind(Gamma,res,x,breeze=0.0):
    """ Compute the velocity profile normalized to the critical speed for a polytropic wind"""    
    w=np.zeros(res)+0.01
    for i in range(0,res):
        if ((x[i] >= 1) and (breeze==0.0)): 
            w[i]=10

        f=w[i]**(Gamma+1.)-w[i]**(Gamma-1.)*(4./x[i]+(5.-3.*Gamma)/(Gamma-1.))+2./(Gamma-1.)*x[i]**(2.-2.*Gamma)+breeze
        while (abs(f) >= 1e-11):
            fprime=(Gamma+1.)*w[i]**(Gamma)-w[i]**(Gamma-2.)*(Gamma-1.)*(4./x[i]+(5.-3.*Gamma)/(Gamma-1.))
            wref=w[i]
            w[i]=w[i]-f/fprime
            while (w[i] <= 0):
                w[i]=0.5*(w[i]+wref)
            f=w[i]**(Gamma+1.)-w[i]**(Gamma-1.)*(4./x[i]+(5.-3.*Gamma)/(Gamma-1.))+2./(Gamma-1.)*x[i]**(2.-2.*Gamma)+breeze
    return w

# Parker's isothermal solution
def parkerWind(r):
    """Compute the isothermal wind solution (Parker 1958), velocity profile normalized by critical speed"""
    v=np.zeros(len(r))+1e-7

    for i in range(0,len(r)):
        f=v[i]**2.-2.*np.log(v[i])-4./r[i]-4.*np.log(r[i])+3.
        j=0
        if(r[i]>= 1): 
            v[i]=10
        while(abs(f) >= 1e-11):
            vref=v[i]
            v[i]=v[i]-0.5*f/(v[i]-1./v[i])
            while(v[i] <= 0):
                v[i]=(vref+v[i])/2
            f=v[i]**2.-2.*np.log(v[i])-4./r[i]-4.*np.log(r[i])+3.
            j=j+1
    return v

def parkerAcc(r):
    """Compute the isothermal accretion solution (Parker 1958), velocity profile normalized by critical speed"""
    v=np.zeros(len(r))+1e-7

    for i in range(0,len(r)):
        f=v[i]**2.-2.*np.log(v[i])-4./r[i]-4.*np.log(r[i])+3.
        j=0
        if(r[i]<= 1): 
            v[i]=10
        while(abs(f) >= 1e-11):
            vref=v[i]
            v[i]=v[i]-0.5*f/(v[i]-1./v[i])
            while(v[i] <= 0):
                v[i]=(vref+v[i])/2
            f=v[i]**2.-2.*np.log(v[i])-4./r[i]-4.*np.log(r[i])+3.
            j=j+1
    return v


def cmpParkerTemp(Rorb,Vorb):
    """ Compute the coronal temperature for a given speed Vorb,
    at a given orbit Rorb """

    mp=1.67e-27  # kg
    kb=1.38e-23  # J/K
    G=6.67e-11   # SI
    Msun=1.98e30 #kg
    
    DeltaV=Vorb
    epsilon=1e-3
    T=6000 # Initial (photopsheric) temperature
    dT=1000 # Temperature step
    r=np.array([Rorb])
    while(DeltaV > epsilon):
        c_s2=2*kb/mp*T
        r_c=G*Msun/2/c_s2
        v=parkerWind(r/r_c)*np.sqrt(c_s2)
        
        print("Temperature = {0}, Wind Speed = {1}".format(T,v))
        T=T+dT
        DeltaV=Vorb-v    

    return T

def cmpCoronalDensity(Rorb,Norb,Tcor):
    """ Compute the coronal base density for a given density Norb, 
    at a given orbit Rorb, for a given coronal temperature Tcor """
    
    mp=1.67e-27  # kg
    kb=1.38e-23  # J/K
    G=6.67e-11   # SI
    Msun=1.98e30 # kg
    Rsun=6.9e8   # m
    
    c_s2=2*kb/mp*Tcor
    r_c=G*Msun/2/c_s2
    r=np.array([Rsun,Rorb])
    v0,vOrb=parkerWind(r/r_c)*np.sqrt(c_s2)
    print(v0,vOrb)
    
    return Norb*vOrb*Rorb**2/v0/Rsun**2
    
