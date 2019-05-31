"""Simulation of spreading depression"""
from mpi4py import MPI
from neuron import h, crxd as rxd
from neuron.crxd import rxdmath
from matplotlib import pyplot, colors, colorbar
from matplotlib_scalebar import scalebar
from mpl_toolkits.mplot3d import Axes3D
import numpy
import argparse
import os
import sys
import pickle

#when using multiple processes get the relevant id and number of hosts 
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()

# set the save directory and if buffering or inhomogeneous tissue
# characteristics are used.
try:
    parser = argparse.ArgumentParser(description = '''Run the spreading
                                     depression simulation''')
    
    parser.add_argument('--edema', dest='edema', action='store_const',
                        const=True, default=False,
                        help='''Use inhomogeneous tortuosity and volume
                        fraction to simulate edema''')
    parser.add_argument('--buffer', dest='buffer', action='store_const',
                        const=True, default=False,
                        help='Use a reaction to model astrocytic buffering')
    parser.add_argument('--tstop', nargs='?', type=float, default=200,
                        help='''duration of the simulation in ms (defaults
                        to 200ms)''')
    parser.add_argument('dir', metavar='dir', type=str,
                        help='a directory to save the figures and data')
    args = parser.parse_args()
except:
    os._exit(1)

outdir = os.path.abspath(args.dir)
if pcid == 0 and not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
    except:
        print("Unable to create the directory %r for the data and figures"
              % outdir)
        os._exit(1)

rxd.nthread(4)  # set the number of rxd threads
rxd.options.enable.extracellular = True # enable extracellular rxd

h.load_file('stdrun.hoc')
h.celsius = 37

numpy.random.seed(6324555+pcid)    # use a difference seed for each process

# simulation parameters
Lx, Ly, Lz = 1000, 1000, 1000      # size of the extracellular space mu m^3
Kceil = 15.0                       # threshold used to determine wave speed
Ncell = int(9e4*(Lx*Ly*Lz*1e-9))   # number of neurons (90'000 per mm^3)
Nrec = 1000

somaR = 11.0     # soma radius
dendR = 1.4      # dendrite radius
dendL = 100.0    # dendrite length
doff = dendL + somaR

alpha0, alpha1 = 0.07, 0.2  # anoxic and normoxic volume fractions 
tort0, tort1 = 1.8, 1.6     # anoxic and normoxic tortuosities 
r0 = 100                    # radius for initial elevated K+

class Neuron:
    """ A neuron with soma and dendrite with; fast and persistent sodium
    currents, potassium currents, passive leak and potassium leak and an
    accumulation mechanism. """
    def __init__(self, x, y, z, rec=False):
        self.x = x
        self.y = y
        self.z = z

        self.soma = h.Section(name='soma', cell=self)
        # add 3D points to locate the neuron in the ECS  
        self.soma.pt3dadd(x, y, z + somaR, 2.0*somaR)
        self.soma.pt3dadd(x, y, z - somaR, 2.0*somaR)
    
        self.dend = h.Section(name='dend', cell=self)
        self.dend.pt3dadd(x, y, z - somaR, 2.0*dendR)
        self.dend.pt3dadd(x, y, z - somaR - dendL, 2.0*dendR)
        #self.dend.nseg = 10 # multiple dendrite segments were used in the
                             # paper but are not necessary for spreading
                             # depression
        self.dend.connect(self.soma, 1,0)
        
        # insert the same mechanisms with the same parameters in both the soma 
        # and the dendrite 
        for mechanism in ['tnak', 'tnap', 'taccumulation3', 'kleak']:
            self.soma.insert(mechanism)
            self.dend.insert(mechanism)

        # the sodium/potassium pump is not used in this model
        self.soma(0.5).tnak.imax = 0
        self.dend(0.5).tnak.imax = 0

        if rec: # record membrane potential (shown in figure 1C)
            self.somaV = h.Vector()
            self.somaV.record(self.soma(0.5)._ref_v, rec)
            self.dendV = h.Vector()
            self.dendV.record(self.dend(0.5)._ref_v, rec)

# Randomly distribute 1000 neurons which we record the membrane potential
# every 100ms
rec_neurons = [Neuron(
    (numpy.random.random()*2.0 - 1.0) * (Lx/2.0 - somaR), 
    (numpy.random.random()*2.0 - 1.0) * (Ly/2.0 - somaR), 
    (numpy.random.random()*2.0 - 1.0) * (Lz/2.0 - somaR), 100)
    for i in range(0, int(Nrec/nhost))]

# Randomly distribute the remaining neurons
all_neurons = [Neuron(
    (numpy.random.random()*2.0 - 1.0) * (Lx/2.0 - somaR),
    (numpy.random.random()*2.0 - 1.0) * (Ly/2.0 - somaR),
    (numpy.random.random()*2.0 - 1.0) * (Lz/2.0 - somaR))
    for i in range(int(Nrec/nhost), int(Ncell/nhost))]

if args.edema:
    # to simulate edema use functions for the diffusion characteristics
    def alpha(x, y, z) :
        return (alpha0 if x**2 + y**2 + z**2 < r0**2
                else min(alpha1, alpha0 +(alpha1-alpha0)
                *((x**2+y**2+z**2)**0.5-r0)/(Lx/2)))

    def tort(x, y, z) :
        return (tort0 if x**2 + y**2 + z**2 < r0**2
                else max(tort1, tort0 - (tort0-tort1)
                *((x**2+y**2+z**2)**0.5-r0)/(Lx/2)))
else:
    # otherwise use the normoxic constants for the diffusion characteristics   
    alpha = alpha1
    tort = tort1


# Where? -- define the extracellular space
ecs = rxd.Extracellular(-Lx/2.0, -Ly/2.0,
                        -Lz/2.0, Lx/2.0, Ly/2.0, Lz/2.0, dx=10,
                        volume_fraction=alpha, tortuosity=tort) 

# What? -- define the species
k = rxd.Species(ecs, name='k', d=2.62, charge=1, initial=lambda nd: 40 
                if nd.x3d**2 + nd.y3d**2 + nd.z3d**2 < r0**2 else 3.5,
                ecs_boundary_conditions=3.5)

na = rxd.Species(ecs, name='na', d=1.78, charge=1, initial=133.574,
                 ecs_boundary_conditions=133.574)

if args.buffer:
    # Additional species are used for a phenomenological model of astrocytic
    # buffering 
    kb = 0.0008
    kth = 15.0
    kf = kb / (1.0 + rxdmath.exp(-(k - kth)/1.15))
    Bmax = 10

    A = rxd.Species(ecs,name='buffer', charge=1, d=0,
                    initial = lambda nd: 0 if nd.x3d**2 + nd.y3d**2 + nd.z3d**2
                    < r0**2 else Bmax)
    AK = rxd.Species(ecs,name='bound', charge=1, d=0,
                    initial = lambda nd: Bmax if nd.x3d**2 + nd.y3d**2 + 
                    nd.z3d**2 < r0**2 else 0)

    # What? -- specify the reactions involved
    buffering = rxd.Reaction(k + A, AK, kf, kb)

pc.set_maxstep(100) # required when using multiple processes

# initialize and set the intracellular concentrations
h.finitialize()
for sec in h.allsec():
    sec.nai = 4.297

def progress_bar(tstop, size=40):
    """ report progress of the simulation """
    prog = h.t/float(tstop)
    fill = int(size*prog)
    empt = size - fill
    progress = '#' * fill + '-' * empt
    sys.stdout.write('[%s] %2.1f%% %6.1fms of %6.1fms\r' % (progress, 100*prog, pc.t(0), tstop))
    sys.stdout.flush()

def plot_rec_neurons():
    """ Produces plots of record neurons membrane potential (shown in figure 1C) """
    # load the recorded neuron data
    somaV, dendV, pos = [], [], []
    for i in range(nhost):
        fin = open(os.path.join(outdir,'membrane_potential_%i.pkl' % i),'rb')
        [sV, dV, p] = pickle.load(fin)
        fin.close()
        somaV.extend(sV)
        dendV.extend(dV)
        pos.extend(p)

        for idx in range(somaV[0].size()):  
            # create a plot for each record (100ms)

            fig = pyplot.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.set_position([0.0,0.05,0.9,0.9])
            ax.set_xlim([-Lx/2.0, Lx/2.0])
            ax.set_ylim([-Ly/2.0, Ly/2.0])
            ax.set_zlim([-Lz/2.0, Lz/2.0])
            ax.set_xticks([int(Lx*i/4.0) for i in range(-2,3)])
            ax.set_yticks([int(Ly*i/4.0) for i in range(-2,3)])
            ax.set_zticks([int(Lz*i/4.0) for i in range(-2,3)])

            cmap = pyplot.get_cmap('jet')
            for i in range(Nrec):
                x = pos[i]
                soma_z = [x[2]-somaR,x[2]+somaR]
                cell_x = [x[0],x[0]]
                cell_y = [x[1],x[1]]
                scolor = cmap((somaV[i].get(idx)+70.0)/70.0)
                # plot the soma
                ax.plot(cell_x, cell_y, soma_z, linewidth=2, color=scolor, 
                        alpha=0.5)
    
                dcolor = cmap((dendV[i].get(idx)+70.0)/70.0)
                dend_z = [x[2]-somaR, x[2]-somaR - dendL]
                # plot the dendrite
                ax.plot(cell_x, cell_y, dend_z, linewidth=0.5, color=dcolor, 
                        alpha=0.5)

            norm = colors.Normalize(vmin=-70,vmax=0)
            pyplot.title('Neuron membrane potentials; t = %gms' % (idx * 100))

            # add a colorbar 
            ax1 = fig.add_axes([0.88,0.05,0.04,0.9])
            cb1 = colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,
                                                orientation='vertical')
            cb1.set_label('mV')
            
            # save the plot
            filename = 'neurons_{:05d}.png'.format(idx)
            pyplot.savefig(os.path.join(outdir,filename))
            pyplot.close()

def plot_image_data(data, min_val, max_val, filename, title):
    """Plot a 2d image of the data"""
    sb = scalebar.ScaleBar(1e-6)
    sb.location='lower left'
    pyplot.imshow(data, extent=k[ecs].extent('xy'), vmin=min_val,
                  vmax=max_val, interpolation='nearest', origin='lower')
    pyplot.colorbar()
    sb = scalebar.ScaleBar(1e-6)
    sb.location='lower left'
    ax = pyplot.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.add_artist(sb)
    pyplot.title(title)
    pyplot.xlim(k[ecs].extent('x'))
    pyplot.ylim(k[ecs].extent('y'))
    pyplot.savefig(os.path.join(outdir,filename))
    pyplot.close()

    
h.dt = 10  # use a large time step as we are not focusing on spiking behaviour
           # but on slower diffusion


def run(tstop):
    """ Run the simulations saving figures every 100ms and recording the wave progression every time step"""
    if pcid == 0:
        # record the wave progress (shown in figure 2)
        name = '' if not args.edema else '_edema'
        name += '' if not args.buffer else '_buffer'
        fout = open(os.path.join(outdir,'wave_progress%s.txt' % name),'a')

    while pc.t(0) < tstop:
        if int(pc.t(0)) % 100 == 0:
            # plot extracellular concentrations averaged over depth every 100ms 
            if pcid == 0:
                plot_image_data(k[ecs].states3d.mean(2), 3.5, 40,
                                'k_mean_%05d' % int(pc.t(0)/100),
                                'Potassium concentration; t = %6.0fms'
                                % pc.t(0))

            if pcid == nhost - 1 and args.buffer:
                plot_image_data(AK[ecs].states3d.mean(2), 0, 10,
                                'buffered_mean_%05d' % int(pc.t(0)/100),
                                'Buffered concentration; t = %6.0fms' % pc.t(0))
        if pcid == 0: progress_bar(tstop)
        pc.psolve(pc.t(0)+h.dt)  # run the simulation for 1 time step
        
        # determine the furthest distance from the origin where
        # extracellular potassium exceeds Kceil (dist)
        # And the shortest distance from the origin where the extracellular
        # extracellular potassium is below Kceil (dist1)
        if pcid == 0:
            dist = 0
            dist1 = 1e9
            for nd in k.nodes:
                r = (nd.x3d**2+nd.y3d**2+nd.z3d**2)**0.5
                if nd.concentration>Kceil and r > dist:
                    dist = r
                if nd.concentration<=Kceil and r < dist1:
                    dist1 = r

            fout.write("%g\t%g\t%g\n" %(pc.t(0), dist, dist1))
            fout.flush()
    if pcid == 0:
        progress_bar(tstop)
        fout.close()
        print("\nSimulation complete. Plotting membrane potentials")

    # save membrane potentials
    soma, dend, pos = [], [], []
    for n in rec_neurons:
        soma.append(n.somaV)
        dend.append(n.dendV)
        pos.append([n.x,n.y,n.z])
    pout = open(os.path.join(outdir,"membrane_potential_%i.pkl" % pcid),'wb')
    pickle.dump([soma,dend,pos],pout)
    pout.close()
    pc.barrier()    # wait for all processes to save

    # plot the membrane potentials (shown in figure 1C)
    if pcid == 0:
        plot_rec_neurons()

#run the simulation
run(args.tstop)
