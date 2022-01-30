import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import starAML as st

# Solar wind 
Mass=1.0   # Solar Mass
Radius=1.0 # Solar Radius
Period=28  # Days
Teff=5778  # Kelvin

Gamma=1.05 # Polytropic index
Tc=1.5e6   # Coronal base temperature (Kelvin)
Nc=1e8     # Coronal base density (cm^-3)
Bp=1.5     # Magnetic dipole moment of the Sun (Gauss) # Not used if a map is defined

s=st.starAML(verbose=0)
s.StellarParameters(Mass,Radius,Period,Teff)
s.WindParameters(Gamma,Tc,Nc,mapfile="ZDImap")
s.cmpWind()
s.readMap()
s.cmpTorque()
s.WindInfo()

br_surf,bt_surf,bp_surf=s.SphHarm.pfss3d(s.alpha,rss=s.Rss,rb=1.0,rsph=1.0)
br_ss,bt_ss,bp_ss=s.SphHarm.pfss3d(s.alpha,rss=s.Rss,rb=1.0,rsph=s.Rss)

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,8))

lon=s.SphHarm.phi[0,:]*180/np.pi
lat=(np.pi/2.-s.SphHarm.theta[:,0])*180/np.pi
bmax=np.abs(br_surf).max()
im1=ax1.pcolormesh(lon,lat,br_surf,vmin=-bmax,vmax=bmax,cmap='bwr',shading='auto')
divider=make_axes_locatable(ax1)
cax=divider.append_axes("right",size="5%",pad=0.1)
cbar=plt.colorbar(im1,cax=cax)
ax1.set_title(r"$B_r (r_{\odot})$ (G)")

bmax=np.abs(br_ss).max()
im2=ax2.pcolormesh(lon,lat,br_ss,vmin=-bmax,vmax=bmax,cmap='bwr',shading='auto')
divider=make_axes_locatable(ax2)
cax=divider.append_axes("right",size="5%",pad=0.1)
cbar=plt.colorbar(im2,cax=cax)
ax2.set_title(r"$B_r (r_{ss})$ (G)")
plt.show()




