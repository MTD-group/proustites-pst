import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from diverging_line_v0 import test_cm

# Define constants
hbar = 6.582e-16 # eV s
c = 299792458 # m/s
c *= 1e10 # A/s
m = 0.511e6 / c**2 # eV s^2 A^-2

# Pauli matrices
sigma0 = np.array([[1, 0], [0, 1]])
sigma1 = np.array([[0, 1], [1, 0]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma3 = np.array([[1, 0], [0, -1]])

def compute_bands(k, sof_x, sof_y, sof_z):
    k = np.asarray(k)
    soc_x = sof_x(k)
    soc_y = sof_y(k)
    soc_z = sof_z(k)
    
    e_soc = np.sqrt(soc_x**2 + soc_y**2 + soc_z**2)
    e_0 = hbar**2 * np.linalg.norm(k, axis=1)**2 / (2*m)
    
    e_p = e_0 + e_soc
    e_n = e_0 - e_soc
    
    wf_p = np.array([[ (soc_z[i] + e_soc[i]) / (soc_x[i] + 1j * soc_y[i]), 1 ] 
                     for i in range(len(k))])
    wf_n = np.array([[ (soc_z[i] - e_soc[i]) / (soc_x[i] + 1j * soc_y[i]), 1 ] 
                     for i in range(len(k))])
    
    # Normalize wavefunctions
    wf_p = wf_p / np.linalg.norm(wf_p, axis=1)[:,None]
    wf_n = wf_n / np.linalg.norm(wf_n, axis=1)[:,None]
    
    # In the case that soc_x = soc_y = 0, wf_p,n will have NaNs. Need to set to (1,0) and (0,1) manually.
    for index in np.nonzero(np.isnan(wf_p))[0]:
        if soc_z[index] != 0:
            wf_p[index] = [1, 0]
            wf_n[index] = [0, 1]
        else:
            wf_p[index] = [0, 0]
            wf_n[index] = [0, 0]
        
    return (e_p, e_n, wf_p, wf_n)

def compute_spins(wf):
    sx = np.array(
        [ np.linalg.multi_dot([w.conj(), sigma1, w]) 
         for w in wf ])
    sy = np.array(
        [ np.linalg.multi_dot([w.conj(), sigma2, w]) 
         for w in wf ])
    sz = np.array(
        [ np.linalg.multi_dot([w.conj(), sigma3, w]) 
         for w in wf ])
    return np.array([sx, sy, sz]).T

## SOC Hamiltonian
## [ axx, axy, axz ]
## [ ayx, ayy, ayz ]
## [ azx, azy, azz ]
## H = a_ij k_i sigma_j

# Linear SOC parameters
soc_coeffs = np.array([
    [ 0, 0.5*0.195165, 0 ],
    [ 0.155137, 0,    -1.12069 ],
    [ 0,    0,    0 ]
])

#soc_coeffs = np.array([
#    [ 0, 1, 0 ],
#    [ 1, 0, 0 ],
#    [ 0,    0,    0 ]
#]) * 1e-1

# Cubic SOC parameters
a_xxy_x = -261.7272301
a_yyy_x = 257.78
a_yzz_x = 0

a_xxx_y = -490
a_xxz_y = 0
a_xyy_y = 480.65491703
a_yyz_y = 0
a_xzz_y = 0
a_zzz_y = 0

a_xxy_z = 1932.51342526
a_yyy_z = -1894.21
a_yzz_z = 0

# Spin-orbit field components
def sof_x_lin(k):
    return k.dot(soc_coeffs[:,0])

def sof_y_lin(k):
    return k.dot(soc_coeffs[:,1])

def sof_z_lin(k):
    return k.dot(soc_coeffs[:,2])
    
# SOF with cubic terms 
def sof_x_cub(k):
    return sof_x_lin(k) + (a_xxy_x * kpts[:,0]**2 * kpts[:,1]
                           + a_yyy_x * kpts[:,1]**3
                           + a_yzz_x * kpts[:,1] * kpts[:,2]**2)

def sof_y_cub(k):
    return sof_y_lin(k) + (a_xxx_y * kpts[:,0]**3
                           + a_xxz_y * kpts[:,0]**2 * kpts[:,2]
                           + a_xyy_y * kpts[:,0] * kpts[:,1]**2
                           + a_yyz_y * kpts[:,1]**2 * kpts[:,2]
                           + a_xzz_y * kpts[:,0] * kpts[:,2]**2
                           + a_zzz_y * kpts[:,2]**3)

def sof_z_cub(k):
    return sof_z_lin(k) + (a_xxy_z * kpts[:,0]**2 * kpts[:,1]
                           + a_yyy_z * kpts[:,1]**3
                           + a_yzz_z * kpts[:,1] * kpts[:,2]**2)

#b = soc_coeffs[:, 0].dot(k)
#c = soc_coeffs[:, 1].dot(k)
#d = soc_coeffs[:, 2].dot(k)
# More simply, b, c, d = k.dot(soc_coeffs)

klim_x = 0.05
klim_y = 0.05
kmesh = np.mgrid[-klim_x:klim_x:21j, 0:klim_y:21j, 0:0:1j]
kpts = kmesh.reshape(3,-1).T

e_p, e_n, wf_p, wf_n = compute_bands(kpts, sof_x_lin, sof_y_lin, sof_z_lin)
#e_p, e_n, wf_p, wf_n = compute_bands(kpts, sof_x_cub, sof_y_cub, sof_z_cub)

s_p = compute_spins(wf_p)
s_n = compute_spins(wf_n)

#s_n[np.abs(s_n) < 1e-6] = 0

## Computing PST direction
spin_cbm = s_n[np.argmin(e_n)]
spin_theta = np.arctan(spin_cbm[0]/spin_cbm[2])
pst_component = s_n[:,0] * np.sin(spin_theta) + s_n[:,2] * np.cos(spin_theta)

band_ind = (kpts[:,1] == 0)

## RdYlBu colormap
#colors_upper = mpl.cm.RdYlBu(np.linspace(0.6, 1, 256))
#colors_lower = mpl.cm.RdYlBu(np.linspace(0.5, 0.6, 256*4))
#colors_all = np.vstack((colors_lower, colors_upper))
#pst_cmap = mpl.colors.LinearSegmentedColormap.from_list('pst_cmap', colors_all)
#pst_norm = mpl.colors.Normalize(vmin=0, vmax=1)

#colors_upper = mpl.cm.RdYlBu(np.linspace(0.6, 1, 256))
#colors_middle = mpl.cm.RdYlBu(np.linspace(0.4, 0.6, 256*8))
#colors_lower = mpl.cm.RdYlBu(np.linspace(0, 0.4, 256))
#colors_combined = np.vstack((colors_lower, colors_middle, colors_upper))
#pst_cmap = mpl.colors.LinearSegmentedColormap.from_list('pst_cmap',
#        colors_combined)
#pst_norm = mpl.colors.Normalize(vmin=-1, vmax=1)

# Custom colormap
colors_upper = test_cm(np.linspace(0.6, 1, 256))
colors_middle = test_cm(np.linspace(0.4, 0.6, 256*8))
colors_lower = test_cm(np.linspace(0, 0.4, 256))
colors_combined = np.vstack((colors_lower, colors_middle, colors_upper))
pst_cmap = mpl.colors.LinearSegmentedColormap.from_list('pst_cmap',
        colors_combined)
pst_norm = mpl.colors.Normalize(vmin=-1, vmax=1)

#pst_cmap = test_cm

pst_comp_mesh = pst_component.reshape(21,21)

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

#ax.quiver(kmesh[0], kmesh[1], s_n[:,0], s_n[:,1])#, s_n[:,2].real)
qv = ax.quiver(kmesh[0], kmesh[1],
        s_n[:,2], s_n[:,0],
        #s_n[:,0], s_n[:,1],
        pst_component.real,
        #clim=(-1, 1),
        cmap=pst_cmap,
        norm=pst_norm,
        width=0.006, scale=24)

#sc = ax.scatter(kmesh[0][pst_comp_mesh>0.98],
#        kmesh[1][pst_comp_mesh>0.98])
#ax.quiver(kmesh[0], kmesh[1], s_n[:,1], s_n[:,2], s_n[:,0].real,
#         clim=(-1, 1), width=0.004, scale=35)

#ax.plot(kpts[band_ind][:,0], e_p[band_ind])
#ax.plot(kpts[band_ind][:,0], e_n[band_ind])

ax.set_xlim(-klim_x, klim_x)
#ax.set_ylim(-0.02, 0.02)
ax.set_ylim(0, klim_y)

#ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
#ax.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
#ax.set_yticks([0, 0.01, 0.02])

ax.set_xlabel('kx')
ax.set_ylabel('ky')

#font = {'weight': 'normal',
#        'size': 20}
#mpl.rc('font', **font)

#ax.set_aspect(2)
#cb = fig.colorbar(qv)
fig.show()

#plt.rcParams['svg.fonttype'] = 'none'
#fig.savefig('pst_0.5.svg')
