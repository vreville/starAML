import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SakuraiUtils as su
import starAML as st

the=np.linspace(0.1,10,2000)
om=np.linspace(0.1,1000,5000)

if not os.path.exists("./thetaomega.npz"):
    su.cmpThetaOmegaMap("./thetaomega.npz",1.05,the,om)

npzfile=np.load("./thetaomega.npz")

The=npzfile['arr_0']
Om=npzfile['arr_1']
xs=npzfile['arr_2']
xf=npzfile['arr_3']
ys=npzfile['arr_4']
yf=npzfile['arr_5']
E=npzfile['arr_6']
Beta=npzfile['arr_7']
Ff=npzfile['arr_8']

fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(12,6))

im1=ax1.pcolormesh(The,Om,E,shading='auto')
divider=make_axes_locatable(ax1)
cax=divider.append_axes("right",size="10%",pad=0.1)
cbar=plt.colorbar(im1,cax=cax)

im2=ax2.pcolormesh(The,Om,Beta,shading='auto',cmap="plasma")
divider=make_axes_locatable(ax2)
cax=divider.append_axes("right",size="10%",pad=0.1)
cbar=plt.colorbar(im2,cax=cax)

ax1.set_title(r"$E$")
ax2.set_title(r"$\beta$")

ax1.set_xlabel(r"$\Theta$",fontsize=15)
ax1.set_ylabel(r"$\Omega$",fontsize=15)
ax2.set_xlabel(r"$\Theta$",fontsize=15)

# Computes case 47 of Réville et al. 2015
Gamma=1.05
vkep=1.0
cs_vesc=0.222
vrot_vesc=0.197/np.sqrt(2.)
va_vesc=1.51
T,O,xs,ys,xf,yf,Beta,E=su.findThetaOmega("./thetaomega.npz",Gamma,vkep,cs_vesc,vrot_vesc,va_vesc)
print("Values to compare with Réville et al. 2015")
print(T,O,xs,ys,xf,yf,Beta,E)

r=np.linspace(1,30,1000)
sakurai_wind=st.WindProfile(r,verbose=True)
sakurai_wind.fromSakurai(Gamma,cs_vesc,vrot_vesc,va_vesc,vrot_vesc*np.sqrt(2.),cmp="True")

fig,ax=plt.subplots(1,1)
unit_velocity=437 #km/s
ax.plot(sakurai_wind.r,sakurai_wind.v*unit_velocity,label=r"$v$")
ax.plot(sakurai_wind.r,sakurai_wind.b/np.sqrt(sakurai_wind.rho)*unit_velocity,label=r"$v_A$")
ax.legend(loc="best")
ax.set_xlabel(r"$r$",fontsize=15)
ax.set_ylabel(r"$km/s$",fontsize=15)

plt.show()
