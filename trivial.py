"""A trivial example of a cube (L**3) diffusing within a cube of extracellular
 space (Lecs**3) validated against a solution for the origin voxel of size dx.
It will produce figure 4 apart from the dx=1/3 and dx=1/9 relative errors
shown in (c) which are obtained by re-running with the appropriate voxel size.
"""
from neuron import h, crxd as rxd
import numpy
from matplotlib import pyplot
from matplotlib_scalebar import scalebar
from scipy.special import erf
h.load_file('stdrun.hoc')

sec = h.Section() #NEURON requires at least 1 section

# enable extracellular RxD
rxd.options.enable.extracellular = True

# simulation parameters
dx = 1.0    # voxel size
L = 9.0     # length of initial cube
Lecs = 21.0 # lengths of ECS

# define the extracellular region
extracellular = rxd.Extracellular(-Lecs/2., -Lecs/2., -Lecs/2.,
                                  Lecs/2., Lecs/2., Lecs/2., dx=dx,
                                  volume_fraction=0.2, tortuosity=lambda x,y,z: 1.6)

# define the extracellular species
k_rxd = rxd.Species(extracellular, name='k', d=2.62, charge=1,
                    initial=lambda nd: 1.0 if abs(nd.x3d) <= L/2. and
                    abs(nd.y3d) <= L/2. and abs(nd.z3d) <= L/2. else 0.0)

# copy of the initial state to plot (figure 4a upper panel)
states_init = k_rxd[extracellular].states3d.copy()

# record the concentration at (0,0,0)
ecs_vec = h.Vector()
ecs_vec.record(k_rxd[extracellular].node_by_location(0, 0, 0)._ref_value)
# record the time
t_vec = h.Vector()
t_vec.record(h._ref_t)

h.finitialize()
h.dt = 0.1
h.continuerun(100) #run the simulation
# record states to plot (figure 4a lower panel)
states_mid = k_rxd[extracellular].states3d.copy()
h.continuerun(200)



# functions to calculate the solution for comparison
def greens(diff, time, offset):
    """ Calculates the integral of the greens function to determine the
    concentration at the central voxel.

    Args:
        diff:  the diffusion coefficient
        time:  the duration (in ms) of diffusing
        offset: a list of length 3 and should be a multiple of the size of the
        extracellular space (Lecs). Using method of mirrors to calculate
        the contribution due to reflection at the boundaries.

    Returns:
        A contribution to the concentration at the origin voxel at time t
    """
    Lx, Ly, Lz = offset
    b = 4.0*(diff*time)**0.5
    sx = 0
    sy = 0
    sz = 0
    for a in [L-Lx, L+Lx]:
        sx += 0.5*((b/(numpy.pi**0.5))*
                   (numpy.exp(-(a+dx)**2/b**2) - numpy.exp(-(a-dx)**2/b**2)) +
                   (dx-a)*erf((a-dx)/b) + (dx+a)*erf((a+dx)/b))
    for a in [L-Ly, L+Ly]:
        sy += 0.5*((b/(numpy.pi**0.5))*
                   (numpy.exp(-(a+dx)**2/b**2) - numpy.exp(-(a-dx)**2/b**2)) +
                   (dx-a)*erf((a-dx)/b) + (dx+a)*erf((a+dx)/b))
    for a in [L-Lz, L+Lz]:
        sz += 0.5*((b/(numpy.pi**0.5))*
                   (numpy.exp(-(a+dx)**2/b**2) - numpy.exp(-(a-dx)**2/b**2)) +
                   (dx-a)*erf((a-dx)/b) + (dx+a)*erf((a+dx)/b))
    sol = (sx*sy*sz)/(8.0*dx**3)

    return sol

def solution(diff, t, n=5):
    """ Uses an integral of the Greens function to determine the concentration at the origin voxel.

    Args:
        diff:  diffusion coefficient
        t:  time solution is calculated
        n:  the number of reflections to consider in each direction

    Returns:
        The concentration at the origin voxel at time t.
    """
    sol = 0
    for i in range(-n, n):
        for j in range(-n, n):
            for k in range(-n, n):
                sol += greens(diff, t, [2.0*i*Lecs, 2.0*j*Lecs, 2.0*k*Lecs])
    return sol


# compare the rxd solution with the integral solution at each time-step
an_sol = []
for s in t_vec:
    if s == 0:
        an_sol.append(1.0)
    else:
        an_sol.append(solution(2.62/1.6**2, s))

ecsV = numpy.array(ecs_vec.to_python())
cmpV = numpy.array(an_sol)

# save the results
tv = t_vec.to_python()
data = numpy.zeros((3, cmpV.shape[0]))
data[0, :] = tv
data[1, :] = ecsV
data[2, :] = cmpV
fout = open('trivial_%1.2f.npy' % dx, 'wb')
numpy.save(fout, data)
fout.close()

# plot functions
def boxoff(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def add_scalebar(ax, scale=1e-6):
    sb = scalebar.ScaleBar(scale)
    sb.location = 'lower left'
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.add_artist(sb)


# plot the states in middle (z=0) of the cube (figure 4a)
fig = pyplot.figure()
ax1 = pyplot.subplot(2, 3, 1)
im1 = pyplot.imshow(states_init[:, :, int(numpy.ceil(extracellular._nz/2))])
add_scalebar(ax1)
ax1.text(-0.1, 1.4, "A", transform=ax1.transAxes, size=12, weight='bold')
pyplot.colorbar()

ax2 = pyplot.subplot(2, 3, 4)
pyplot.imshow(states_mid[:, :, int(numpy.ceil(extracellular._nz/2))]*1e3)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
pyplot.colorbar()

# plot the concentrations at the origin (figure 4b)
ax3 = pyplot.subplot(1, 3, 2)
pyplot.plot(tv, ecsV, label="rxd")
pyplot.plot(tv, cmpV, label="analytic")
pyplot.xlim([0, 200])
ax3.legend(frameon=False)
pyplot.ylabel('concentration')
pyplot.xlabel('time (ms)')
boxoff(ax3)
ax3.text(-0.1, 1.1, "B", transform=ax3.transAxes, size=12, weight='bold')

# plot insets
hax1 = pyplot.axes([.425, .55, .15, .3])
hax1.plot(tv[0:25], ecsV[0:25])
hax1.plot(tv[0:25], cmpV[0:25])
hax1.set_xticks([])
hax1.set_yticks([])

hax2 = pyplot.axes([.425, .2, .15, .3])
hax2.plot(tv[100:125], ecsV[100:125])
hax2.plot(tv[100:125], cmpV[100:125])
hax2.set_xticks([])
hax2.set_yticks([])


# plot the relative error (figure 4c)
ax4 = pyplot.subplot(1, 3, 3)
pyplot.plot(tv, 100*abs(ecsV-cmpV)/cmpV, label="relative error")
pyplot.xlim([0, 200])
boxoff(ax4)
ax4.text(-0.1, 1.1, "C", transform=ax4.transAxes, size=12, weight='bold')
pyplot.ylabel('relative error (%)')
pyplot.xlabel('time (ms)')

pyplot.show()
