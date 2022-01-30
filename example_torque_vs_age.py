import sys
import numpy as np
import matplotlib.pyplot as plt
import starAML as st

# Solar wind 
Mass=1.0   # Solar Mass
Radius=1.0 # Solar Radius
Period=28  # Days
Teff=5778  # Kelvin

Gamma=1.05 # Polytropic index
Tc=1.5e6   # Coronal base temperature (Kelvin)
Nc=1e8     # Coronal base density (cm^-3)
Bp=1.5     # Magnetic dipole moment of the Sun (Gauss)

s=st.starAML(verbose=0)
s.StellarParameters(Mass,Radius,Period,Teff)
s.WindParameters(Gamma,Tc,Nc,Bp=Bp)
s.cmpWind()
s.cmpTorque()
s.WindInfo()

print("Modify solar wind parameter doubling the period")
s.updateWind(Period=14)
s.cmpTorque()
s.WindInfo()

# Parameter study with Period
#tc=s.tau_c(s.Teff) # Convective turnover time (Cranmer & Saar 2011, seconds)
rangePeriods=np.linspace(2.8,28,50)

# Evolve Density and Temperature at the base of the Corona (see Ivanova & Tamm 2003, Holzwarth & Jardine 2007, R/'eville+ 2016)
rangeTemp=Tc*(rangePeriods/28)**(-0.1)
rangeDens=Nc*(rangePeriods/28)**(-0.6)
rangeBmag=Bp*(rangePeriods/28)**(-1.5) # Dynamo law
verb=0
Torques=[]

for ii,pp in enumerate(rangePeriods):
    print("====> Computing Period {:.2f} days [{:.0f}%]\r".format(pp,(100.*ii)/len(rangePeriods)),end='\r',flush=True)
    s.updateWind(Period=pp,Tc=rangeTemp[ii],Nc=rangeDens[ii],Bp=rangeBmag[ii])
    if verb!=0: # print updated wind and torque parameters
        s.WindInfo()
    Torques.append(s.TorqueReville2015)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(rangePeriods/28,Torques/Torques[-1])
ax.plot(rangePeriods/28,1.5*(rangePeriods/28)**(-3))
ax.annotate(r"$\propto \Omega^3$",xy=(rangePeriods[25]/28.,1.7*(rangePeriods[25]/28)**(-3)),xycoords='data')
ax.set_title("Wind Torque vs Rotation Period")
ax.set_xlabel(r"$P/P_{\odot}$",fontsize=18)
ax.set_ylabel(r"$\tau/\tau_{\odot}$",fontsize=18)
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()




