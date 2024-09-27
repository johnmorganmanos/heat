import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def ice_conductivity(temp):
    ''' Temperature dependent conductivity '''
    return 9.828*np.exp(-0.0057*(273.15 + temp))

def heat(t_surf,
         tmax = 1000,
         tmin = 0,
         zmax = 100,
         geoflux = 0.046,
         nz = 39,
         nt = 99,
         accumulation = 0,
         S = np.nan 
        ):
    
    '''
    Solves the advection/diffusion equation with mixed temperature/heat flux boundary conditions
    '''
    ρ_ice = 917. # kg/m3
    c_p = 2097.
    spy = 3.154e7
    k_0 = 2.22 #initial conductivity to use for the dTdz calc
    
    dTdz_init = geoflux / k_0
    
    if np.isnan(S).any():
        S = np.zeros((nz+1,nt+1))
        
    z0=0    
    dz = zmax/(nz+1)
    dt = (tmax-tmin)/nt
    t = np.linspace(tmin,tmax, nt+1)
    z = np.linspace(dz,zmax, nz+1)
    
    # Initial condition: gradient equal to basal gradient and equal to surface temp.
    U = np.zeros((nz+1,nt+1))
    U[:,0] = t_surf[0] + z*dTdz_init
    


#     w = - accumulation * np.ones(nz) # WRONG!
    w = - accumulation * np.linspace(0,1,nz) # CORRECT!
    abc = w*dt/(2*dz)
    Az = np.diag(abc,k=1) - np.diag(abc,k=-1)
    Az[0,:] =0
    Az[-1,:]=0


    
    b= np.zeros((nz+1,1))




    for k in range(nt):
        # ice diffusivity profile
        k_profile = ice_conductivity(U[:,k])
        alpha = (k_profile / (ρ_ice * c_p)) * spy
        
        # cfl is now time dependent (diffusivity is coupled to temperature)
        cfl = alpha*dt/(dz**2)
        Azz = np.diag(1+2*cfl) + np.diag(-cfl[:-1],k=1)\
        + np.diag(-cfl[:-1],k=-1)
        
        A = Azz - Az
        
        # calculate dTdz at the bed
        dTdz = geoflux / k_profile[-1]
        
        # Neumann boundary at the bed
        A[nz,nz-1] = -2*cfl[nz]
        b[nz] =  2*cfl[nz]*dz * dTdz

        
        b[0] = cfl[0]*t_surf[k]    #  Dirichlet boundary condition

        c = U[:,k] + b.flatten() + S[:,k+1]*dt # previous values + dirichlet + sources
        U[:,k+1] = np.linalg.solve(A,c)

    return U,t,z

def heat_plot(t,t_surf,start_year,end_year,z,U,plot_start_frac=0.95):
    nt=len(t)
    plt.subplots(2,2,figsize=(8,9))
    plt.subplot(221)
    plt.plot(t,t_surf)
    plt.xlim([start_year,end_year])
    plt.grid()
    plt.title('a. Surface temperature forcing')

    plt.subplot(224)
    strt = int((nt+1)*plot_start_frac)
#     for i in range( strt,nt):
#         plt.plot((U[:,i]),z,label=f't={t[i]:.2f}')
    plt.plot((U[:,-1]),z,'-k',linewidth=3,label=f't={t[-3]:.2f}')

    plt.ylim([max(z),0])
    plt.legend()
    plt.title('c. Temperature profile at last time step')
    


    plt.subplot(223)
    c=plt.pcolormesh(t,z,U)
    plt.colorbar(c,location='bottom')
    plt.ylim([max(z),0])
#     plt.xlabel('Calendar year')
    plt.title('b. Temperature distribution through time')

    plt.tight_layout()
    plt.show()
