"""Week 9 MAC cavity solver (same scheme as the class notebook), vectorized with numpy.
Used only to produce CFD reference fields for checking NN extrapolation; never for training."""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

def pressure_factor(n):
    h=1.0/n
    A=lil_matrix((n*n,n*n))
    for j in range(n):
        for i in range(n):
            k=j*n+i
            for jj,ii in [(j-1,i),(j+1,i),(j,i-1),(j,i+1)]:
                if 0<=jj<n and 0<=ii<n:
                    A[k,jj*n+ii]=1/h**2
                    A[k,k]-=1/h**2
    A[0,:]=0; A[0,0]=1
    return splu(A.tocsc())

def tentative_velocity_loop(u,v,nu,dt,h):
    # literal copy of the notebook's numba kernel, for checking the vectorized version
    n=u.shape[0]; us=u.copy(); vs=v.copy()
    for j in range(n):
        for i in range(1,n):
            uc=u[j,i]
            vc=.25*(v[j,i-1]+v[j,i]+v[j+1,i-1]+v[j+1,i])
            below=u[j-1,i] if j>0 else -uc
            above=u[j+1,i] if j<n-1 else 2.0-uc
            dx=(uc-u[j,i-1])/h if uc>=0 else (u[j,i+1]-uc)/h
            dy=(uc-below)/h if vc>=0 else (above-uc)/h
            lap=(u[j,i-1]+u[j,i+1]+below+above-4*uc)/(h*h)
            us[j,i]=uc+dt*(-uc*dx-vc*dy+nu*lap)
    for j in range(1,n):
        for i in range(n):
            vc=v[j,i]
            uc=.25*(u[j-1,i]+u[j-1,i+1]+u[j,i]+u[j,i+1])
            left=v[j,i-1] if i>0 else -vc
            right=v[j,i+1] if i<n-1 else -vc
            dx=(vc-left)/h if uc>=0 else (right-vc)/h
            dy=(vc-v[j-1,i])/h if vc>=0 else (v[j+1,i]-vc)/h
            lap=(left+right+v[j-1,i]+v[j+1,i]-4*vc)/(h*h)
            vs[j,i]=vc+dt*(-uc*dx-vc*dy+nu*lap)
    return us,vs

def tentative_velocity(u,v,nu,dt,h):
    n=u.shape[0]; us=u.copy(); vs=v.copy()
    uc=u[:,1:n]
    vc=.25*(v[:n,:n-1]+v[:n,1:]+v[1:,:n-1]+v[1:,1:])
    below=np.vstack((-uc[:1],u[:n-1,1:n]))
    above=np.vstack((u[1:,1:n],2.0-uc[-1:]))
    lt,rt=u[:,:n-1],u[:,2:]
    dx=np.where(uc>=0,(uc-lt)/h,(rt-uc)/h)
    dy=np.where(vc>=0,(uc-below)/h,(above-uc)/h)
    lap=(lt+rt+below+above-4*uc)/(h*h)
    us[:,1:n]=uc+dt*(-uc*dx-vc*dy+nu*lap)
    vc=v[1:n,:]
    uc=.25*(u[:n-1,:n]+u[:n-1,1:]+u[1:n,:n]+u[1:n,1:])
    left=np.hstack((-vc[:,:1],v[1:n,:n-1]))
    right=np.hstack((v[1:n,1:],-vc[:,-1:]))
    dn,up=v[:n-1,:],v[2:,:]
    dx=np.where(uc>=0,(vc-left)/h,(right-vc)/h)
    dy=np.where(vc>=0,(vc-dn)/h,(up-vc)/h)
    lap=(left+right+dn+up-4*vc)/(h*h)
    vs[1:n,:]=vc+dt*(-uc*dx-vc*dy+nu*lap)
    return us,vs

def solve_cavity(reynolds,n=50,max_steps=20000,steady_tol=2e-5,factor=None):
    h=1/n; nu=1/reynolds; dt=min(.2*h,.2*h*h/nu)
    factor=pressure_factor(n) if factor is None else factor
    u=np.zeros((n,n+1)); v=np.zeros((n+1,n)); p=np.zeros((n,n))
    change=np.inf
    for step in range(max_steps):
        us,vs=tentative_velocity(u,v,nu,dt,h)
        rhs=((us[:,1:]-us[:,:-1]+vs[1:,:]-vs[:-1,:])/h/dt).ravel()
        rhs-=rhs.mean(); rhs[0]=0
        p=factor.solve(rhs).reshape(n,n)
        us[:,1:-1]-=dt*(p[:,1:]-p[:,:-1])/h
        vs[1:-1,:]-=dt*(p[1:,:]-p[:-1,:])/h
        change=max(np.max(np.abs(us-u)),np.max(np.abs(vs-v)))/dt
        u,v=us,vs
        if not np.isfinite(change) or np.max(np.abs(u))>10:
            raise ValueError('CFD unstable: no field accepted.')
        if step>100 and change<steady_tol: break
    div=(u[:,1:]-u[:,:-1]+v[1:,:]-v[:-1,:])/h
    p-=p.mean()
    field=np.stack(((u[:,1:]+u[:,:-1])/2,(v[1:,:]+v[:-1,:])/2,p),axis=-1)
    return field,step+1,change,np.sqrt(np.mean(div**2))
