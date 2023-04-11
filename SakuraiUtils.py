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
import matplotlib.pyplot as plt
import itertools
import find_intersections as fi

#################################################################
class SakuraiSolution(object):

    def __init__(self,Gamma,Theta,Omega,xs,ys,xf,yf,Beta,E):
        self.Gamma=Gamma
        self.Theta=Theta
        self.Omega=Omega
        self.xs=xs
        self.ys=ys
        self.xf=xf
        self.yf=yf
        self.Beta=Beta
        self.E=E

##########################

# Functions components

# f1=H-E at x=xs, y=ys
# f2=H-E at x=xf, y=yf
# f3=dH/dx at x=xs,y=ys
# f4=dH/dx at x=xf,y=yf
# f5=dH/dy at x=xs,y=ys
# f6=dH/dy at x=xf,y=yf

##########################

    def f1(self,x1,y1,x2,y2,b,e):
        return b/2./x1**4./y1**2.+self.Theta/(self.Gamma-1.)*y1**(self.Gamma-1.)-1./x1+self.Omega/2.*((x1-1./x1)**2/(y1-1.)**2.-x1**2.)-e

    def f2(self,x1,y1,x2,y2,b,e):
        return b/2./x2**4./y2**2.+self.Theta/(self.Gamma-1.)*y2**(self.Gamma-1.)-1./x2+self.Omega/2.*((x2-1./x2)**2/(y2-1.)**2.-x2**2.)-e

    def f3(self,x1,y1,x2,y2,b,e):
        return -2.*b/x1**5./y1**2.+1./x1**2.+self.Omega*((x1-1./x1**3.)/(y1-1.)**2.-x1)

    def f4(self,x1,y1,x2,y2,b,e):
        return -2.*b/x2**5./y2**2.+1./x2**2.+self.Omega*((x2-1./x2**3.)/(y2-1.)**2.-x2)

    def f5(self,x1,y1,x2,y2,b,e):
        return -b/x1**4./y1**3.+self.Theta*y1**(self.Gamma-2.)-self.Omega*((x1-1./x1)**2./(y1-1)**3.)

    def f6(self,x1,y1,x2,y2,b,e):
        return -b/x2**4./y2**3.+self.Theta*y2**(self.Gamma-2.)-self.Omega*((x2-1./x2)**2./(y2-1)**3.)

    def fun(self,x):
        return np.array([self.f1(x[0],x[1],x[2],x[3],x[4],x[5]),
                         self.f2(x[0],x[1],x[2],x[3],x[4],x[5]),
                         self.f3(x[0],x[1],x[2],x[3],x[4],x[5]),
                         self.f4(x[0],x[1],x[2],x[3],x[4],x[5]),
                         self.f5(x[0],x[1],x[2],x[3],x[4],x[5]),
                         self.f6(x[0],x[1],x[2],x[3],x[4],x[5])])


##############################
# Jacobian matrix components
##############################
    def H(self,x,y,b):
        return b/2./x**4./y**2.+self.Theta/(self.Gamma-1.)*y**(self.Gamma-1.)-1./x+self.Omega/2.*((x-1./x)**2/(y-1.)**2.-x**2.)

    def dHdx(self,x,y,b):
        return -2.*b/x**5./y**2.+1./x**2.+self.Omega*((x-1./x**3.)/(y-1.)**2.-x)

    def dHdy(self,x,y,b):
        return -b/x**4./y**3.+self.Theta*y**(self.Gamma-2.)-self.Omega*((x-1./x)**2/(y-1.)**3.)

    def dHdb(self,x,y,b):
        return 1./2./x**4./y**2.

    def d2Hdx2(self,x,y,b):
        return 10.*b/x**6./y**2.-2./x**3.+self.Omega*((1.+3./x**4.)/(y-1.)**2.-1.)

    def d2Hdxdy(self,x,y,b):
        return 4.*b/x**5./y**3.-2.*self.Omega*((x**4.-1.)/x**3./(y-1.)**3.)

    def d2Hdy2(self,x,y,b):
        return 3.*b/x**4./y**4.+(self.Gamma-2.)*self.Theta*y**(self.Gamma-3.)+3.*self.Omega*(x-1./x)**2./(y-1.)**4.

    def d2Hdxdb(self,x,y,b):
        return -2./x**5./y**2.

    def d2Hdydb(self,x,y,b):
        return -1./x**4./y**3.

    def NewtonRaphson(self,x0,tol,update=True):

        step=0
        xn=x0
        fn=self.fun(xn)    

        while ((np.linalg.norm(fn)>=tol) and (step<1000)):
        
            xsn=xn[0]
            ysn=xn[1]
            xfn=xn[2]
            yfn=xn[3]
            bn=xn[4]
        
            J11=self.dHdx(xsn,ysn,bn); J12=self.dHdy(xsn,ysn,bn); J13=0.0; J14=0.0; J15=self.dHdb(xsn,ysn,bn); J16=-1.
            J21=0.0; J22=0.0; J23=self.dHdx(xfn,yfn,bn); J24=self.dHdy(xfn,yfn,bn); J25=self.dHdb(xfn,yfn,bn); J26=-1.
            J31=self.d2Hdx2(xsn,ysn,bn); J32=self.d2Hdxdy(xsn,ysn,bn); J33=0.0; J34=0.0; J35=self.d2Hdxdb(xsn,ysn,bn); J36=0.0;
            J41=0.0; J42=0.0; J43=self.d2Hdx2(xfn,yfn,bn); J44=self.d2Hdxdy(xfn,yfn,bn); J45=self.d2Hdxdb(xfn,yfn,bn); J46=0.0;
            J51=self.d2Hdxdy(xsn,ysn,bn); J52=self.d2Hdy2(xsn,ysn,bn); J53=0.0; J54=0.0; J55=self.d2Hdydb(xsn,ysn,bn); J56=0.0;
            J61=0.0; J62=0.0; J63=self.d2Hdxdy(xfn,yfn,bn); J64=self.d2Hdy2(xfn,yfn,bn); J65=self.d2Hdydb(xfn,yfn,bn); J66=0.0;

            J=np.matrix([[J11, J12, J13, J14, J15, J16],
                         [J21, J22, J23, J24, J25, J26],
                         [J31, J32, J33, J34, J35, J36],
                         [J41, J42, J43, J44, J45, J46],
                         [J51, J52, J53, J54, J55, J56],
                         [J61, J62, J63, J64, J65, J66]])
        
            I=J.getI()
        
            XN=np.matrix(xn).getT()
            FN=np.matrix(fn).getT()

            XNP=XN-I*FN

#            for i in range(0,6):
#                count=0
#                while((XNP[i,0] <= 0) and (count<=10)):
#                    XNP[i,0]=0.5*(XNP[i,0]+XN[i,0])
#                    count+=1
#                if (count==1000):
#                    print XN
#                    print XNP

            XN=XNP

            xn=[XN[0,0],XN[1,0],XN[2,0],XN[3,0],XN[4,0],XN[5,0]]
            fn=self.fun(xn)
            
            step=step+1

            if(step>=998):
                print('Theta = {0}, Omega ={1}'.format(self.Theta,self.Omega))
                print('xn ={0} ||fn||={1}'.format(xn, np.linalg.norm(fn)))

        if(np.linalg.norm(fn)>=1e-7):
            print('Warning one point might not have fully converged')
            
        if(update):
            self.xs=xn[0]
            self.ys=xn[1]
            self.xf=xn[2]
            self.yf=xn[3]
            self.Beta=xn[4]
            self.E=xn[5]


        return xn

    def alfvenParam(self,vkep,cs_vesc,vrot_vesc,rstar,rhostar):
        vesc=vkep*np.sqrt(2.)
        cs=cs_vesc*vesc
        vrot=vrot_vesc*np.sqrt(2.)

        rA=(self.Omega*vkep**2/vrot**2)**(1./3.)*rstar
        rhoA=(self.Theta*vkep**2*rstar/rA/cs**2)**(1./(self.Gamma-1.))*rhostar

        return [rA,rhoA]

    def rhoProfile(self,x,vkep,cs_vesc,vrot_vesc,rstar,rhostar):
        rho_profile=np.zeros(len(x))+0.01
        rA,rhoA=self.alfvenParam(vkep,cs_vesc,vrot_vesc,rstar,rhostar)
        print(rA, rhoA)
        print(self.Theta, self. Omega, self.xs, self.ys, self.xf, self.yf, self.Beta, self.E)

        for i in range(0,len(x)):
            if(i==0):
                rho_profile[i]=1/rhoA
            elif (x[i]<=self.xs):
                rho_profile[i]=rho_profile[i-1]*0.99
            elif(x[i]<=0.95):
                rho_profile[i]=rho_profile[i-1]*0.95
            elif((x[i]>0.95) and (x[i]<1.05)):
                slope=(rho_profile[i-1]-rho_profile[i-2])/(x[i-1]-x[i-2])
                b=(rho_profile[i-2]*x[i-1]-rho_profile[i-1]*x[i-2])/(x[i-1]-x[i-2])
                rho_profile[i]=x[i]*slope+b
            elif((x[i]>=1.05) and (x[i]<=self.xf)):
                rho_profile[i]=rho_profile[i-1]*0.99
            elif(x[i]>=self.xf):
                rho_profile[i]=rho_profile[i-1]*0.95

            fn=self.H(x[i],rho_profile[i],self.Beta)-self.E
            rho_ref=10.0
            
            while(abs(fn) >= 1e-10):
                
                fprime=self.dHdy(x[i],rho_profile[i],self.Beta)
                rho_profile[i]=rho_profile[i]-fn/fprime

                while(rho_profile[i] <= 0):
                    rho_profile[i]=0.5*(rho_ref+rho_profile[i])

                fn=self.H(x[i],rho_profile[i],self.Beta)-self.E

                rho_ref=rho_profile[i]

        return rho_profile            

    def solutionCheck(self):
        print(self.dHdx(self.xs,self.ys,self.Beta))
        print(self.dHdx(self.xf,self.yf,self.Beta))
        print(self.dHdy(self.xs,self.ys,self.Beta))
        print(self.dHdy(self.xf,self.yf,self.Beta))
        print(self.H(self.xs,self.ys,self.Beta)-self.E)
        print(self.H(self.xf,self.yf,self.Beta)-self.E)

####################################################################

def cmpThetaOmegaMap(filename,Gamma,the,om):

    """ Compute (Theta, Omega) Map of xs,ys,xf,yf,E,Beta and the Newton-Raphson
    Error given Gamma and the boundary/resolution """


    if(Gamma !=1.05):
        print("Warning, Gamma != 1.05, the method might not converge")
    
    #Close solution for Theta=0.5 and Omega=0.25 and Gamma=1.05
    x0=[0.85,1.85,1.3,0.6,0.6,10]

    lineOm=np.linspace(om[0],0.25,500)
    lineThe=np.linspace(the[0],0.5,500)

    for k in range(1,len(lineOm)+1):
        Theta=lineThe[-k]
        Omega=lineOm[-k]
    
        Sol=SakuraiSolution(Gamma,Theta,Omega,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5])
        x1,y1,x2,y2,b,e=Sol.NewtonRaphson(x0,1e-10)
        x0=[x1,y1,x2,y2,b,e]
        Err=np.linalg.norm(Sol.fun(x0))
        if Err!=Err:
            raise ValueError("Nan detected, check your Theta0, Omega0")


    xs=np.zeros((len(the),len(om)))
    xf=np.zeros((len(the),len(om)))
    ys=np.zeros((len(the),len(om)))
    yf=np.zeros((len(the),len(om)))
    E=np.zeros((len(the),len(om)))
    Beta=np.zeros((len(the),len(om)))
    The=np.zeros((len(the),len(om)))
    Om=np.zeros((len(the),len(om)))
    Err=np.zeros((len(the),len(om)))

    for i in range(0,len(the)):

        if(i>0):
            x0=[xs[i-1,0],ys[i-1,0],xf[i-1,0],yf[i-1,0],Beta[i-1,0],E[i-1,0]]
            
        for j in range(0,len(om)):
            print("====> Computing Theta Omega Map : progress = {:.5f}%\r".format(100.0*(len(om)*i+j)/(len(om)*len(the))),end='\r',flush=True)

            Theta=the[i]
            Omega=om[j]

            Sol=SakuraiSolution(Gamma,Theta,Omega,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5])        
            x1,y1,x2,y2,b,e=Sol.NewtonRaphson(x0,1e-10)

            xs[i,j]=x1
            ys[i,j]=y1
            xf[i,j]=x2
            yf[i,j]=y2
            Beta[i,j]=b
            E[i,j]=e
            The[i,j]=Theta
            Om[i,j]=Omega
        
            x0=[x1,y1,x2,y2,b,e]
            Err[i,j]=np.linalg.norm(Sol.fun(x0))
            if Err[i,j]!=Err[i,j]:
              raise ValueError("Nan detected, refine your map resolution!")              

    np.savez(filename,The,Om,xs,xf,ys,yf,E,Beta,Err)


def findThetaOmega(filename,Gamma,vkep,cs_vesc,vrot_vesc,va_vesc):
    """find corresponding (Theta, Omega) for given stellar parameters """

    vesc=vkep*np.sqrt(2.)
    vrot=vrot_vesc*vesc
    cs=cs_vesc*vesc
    va=va_vesc*vesc

    qty1=(cs**2/(Gamma-1.)-vkep**2-vrot**2./2.)/(vkep**2*vrot)**(2./3.)
    qty2=cs**2*va**(2*(Gamma-1.))/(vkep**(2*(2.*Gamma-4./3.)))/vrot**(2*(4./3.-Gamma))

    npzfile=np.load(filename)
    The=npzfile['arr_0']
    Om=npzfile['arr_1']
    xs=npzfile['arr_2']
    xf=npzfile['arr_3']
    ys=npzfile['arr_4']
    yf=npzfile['arr_5']
    E=npzfile['arr_6']
    Beta=npzfile['arr_7']
    Ff=npzfile['arr_8']

    Q1=E/Om**(1./3.)
    Q2=Beta**(Gamma-1.)*The/Om**(4./3.-Gamma)

    fig,ax=plt.subplots(1,1)
    cnt_q1=ax.contour(The,Om,Q1,[qty1])
    path_q1=cnt_q1.collections[0].get_paths()

    cnt_q2=ax.contour(The,Om,Q2,[qty2])
    path_q2=cnt_q2.collections[0].get_paths()
    
    xi=np.array([])
    yi=np.array([])

    i=0
    ncombos = (sum([len(x.get_paths()) for x in cnt_q1.collections]) * 
               sum([len(x.get_paths()) for x in cnt_q2.collections]))
    for linecol1, linecol2 in itertools.product(cnt_q1.collections, cnt_q2.collections):
        for path1, path2 in itertools.product(linecol1.get_paths(),linecol2.get_paths()):
            i += 1
            #print('line combo {0} of {1}'.format(i, ncombos))        
            xinter, yinter = fi.linelineintersect(path1.vertices, path2.vertices)

            xi = np.append(xi, xinter)
            yi = np.append(yi, yinter)

    #print np.column_stack((xi,yi))

    if (len(xi) != 1):
        raise ValueError("Error, no or more than one set of (Theta, Omega) can be picked")
    else:
        FinalTheta=xi[0]
        FinalOmega=yi[0]

    ax.set_xlim([0.5*FinalTheta, 1.5*FinalTheta])
    ax.set_ylim([0.5*FinalOmega, 1.5*FinalOmega])

    idx=np.where(The[:,0] >= FinalTheta)[0][0]
    idy=np.where(Om[0,:] >= FinalOmega)[0][0]

    return [FinalTheta,FinalOmega,xs[idx,idy],ys[idx,idy],xf[idx,idy],yf[idx,idy],Beta[idx,idy],E[idx,idy]]
