'''
This module houses the functions that are to be run by each FireWorks rocket
for SPARC calculations. Both this module and the `qe_functions.py` module are
hard-coded for the HPC centers that we use. If you use GASpy, you'll need to
make your own versions.
'''

import os
import uuid
import numpy as np
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator as SPC

from sparc.sparc_core import SPARC
from sparc.ion import read_ion, write_ion


__author__ = ['Kevin Tran (CMU)', 'Ben Comer (Georgia Tech)']

# Need to handle setting the pseudopotential directory, probably in the submission
# config if it stays constant? (sparc_qadapter.yaml)


def runSPARC(fname_in, fname_out, sparcflags):
    '''
    This function is meant to be sent to each cluster and then used to run our rockets.
    As such, it has algorithms to run differently depending on the cluster that is trying
    to use this function.

    Inputs:
        fname_in     The filename of the input atoms object
        fname_out    The filename of the output atoms object
        sparcflags   The input parameters for SPARC
    '''
    fname_in = str(fname_in)
    fname_out = str(fname_out)

    # read the input atoms object
    atoms = read(str(fname_in))

    # Check that the unit vectors obey the right-hand rule, (X x Y points in Z) and if not
    # Flip the order of X and Y to enforce this so that SPARC is happy.
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])

    sparc_cmd = 'sparc -name'
    os.environ['PBS_SERVER'] = 'shared-sched.pace.gatech.edu'

    if 'PBS_NODEFILE' in os.environ:
        # double equals??
        NPROCS = NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())
    elif 'SLURM_CLUSTER_NAME' in os.environ:
        if 'SLURM_NPROCS' in os.environ:
            # We're on cori haswell
            NPROCS = int(os.environ['SLURM_NPROCS'])
        else:
            # we're on cori KNL, just one processor
            NPROCS = 1

    # If we're on Gilgamesh...
    if 'PBS_NODEFILE' in os.environ and os.environ['PBS_SERVER'] == 'gilgamesh.cheme.cmu.edu':
        sparc_cmd = '/home-research/zhongnanxu/opt/SPARC/lib/sparc -name PREFIX > log'
        NPROCS = NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())
        mpicall = lambda x, y: 'mpirun -np %i %s' % (x, y)  # noqa: E731
        raise Exception('Not implemented')
    # PACE
    elif 'PBS_NODEFILE' in os.environ and 'pace.gatech.edu' in os.environ['HOSTNAME']:
        sparc_cmd = '/gpfs/pace1/project/chbe-medford/medford-share/builds/sparc/dev_new_inputs/SPARC/lib/sparc -name PREFIX > log'
        NPROCS = NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())
        mpicall = lambda x, y: 'mpirun -np %i %s' % (x, y)  # noqa: E731

    # If we're on Arjuna...
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME'] == 'arjuna':
        raise Exception('Not implemented')

    # If we're on Cori, use SLURM. Note that we decrease the priority by 1000
    # in order to prioritize other things higher, such as modeling and prediction
    # in GASpy_regression
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME'] == 'cori':
        # If we're on a Haswell node...
        if os.environ['CRAY_CPU_TARGET'] == 'haswell' and 'knl' not in os.environ['PATH']:
            NNODES = int(os.environ['SLURM_NNODES'])
            raise Exception('Not implemented')
            mpicall = lambda x, y: 'srun -n %d %s' % (x, y)  # noqa: E731
        # If we're on a KNL node...
        elif 'knl' in os.environ['PATH']:
            mpicall = lambda x, y: 'srun -n %d -c8 --cpu_bind=cores %s' % (x*32, y)  # noqa: E731
            raise Exception('Not implemented')

    # Set the pseudopotential type by setting 'xc' in SPARC()

    pseudopotential = sparcflags['psps']
    os.environ['SPARC_PSP_PATH'] = os.environ['SPARC_PSP_BASE'] + '/' + str(pseudopotential) + '/'
    del sparcflags['pp_version']

    os.environ['ASE_SPARC_COMMAND'] = mpicall(NPROCS, sparc_cmd)

    # Detect whether or not there are constraints that cannot be handled by SPARC
    allowable_constraints = ['FixAtoms', 'FixedLine', 'FixedPlane']
    constraint_not_allowable = [constraint.todict()['name']
                                not in allowable_constraints
                                for constraint in atoms.constraints]
    sparc_incompatible_constraints = np.any(constraint_not_allowable)

    # If there are incompatible constraints, we need to switch to an ASE-based optimizer
    if sparc_incompatible_constraints:
        sparcflags['relax_flag'] = 0
        calc = SPARC(**sparcflags)
        atoms.set_calculator(calc)
        qn = BFGS(atoms, logfile='relax.log', trajectory='all.traj')
        qn.run(fmax=sparcflags['tol_relax'] if 'tol_relax' in sparcflags else 0.05)
        finalimage = atoms

    else:
        # set up the calculation and run
        calc = SPARC(**sparcflags)
        atoms.set_calculator(calc)

        # Trigger the calculation
        # a [label].traj file is automatically made and the atoms are
        # updated
        atoms.get_potential_energy()

        #calc.parse_output(parse_traj = True)
        os.rename('{}.traj'.format(sparcflags['label'],),
                  'all.traj')
         
        """
        for atoms in read('sparcrun.xml', ':'):
            catoms = atoms.copy()
            catoms = catoms[calc.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[calc.resort]))
            atomslist += [catoms]
        """

        # Get the final trajectory
        finalimage = atoms

    # Write the final structure
    finalimage.write(fname_out)

    # Write a text file with the energy
    with open('energy.out', 'w') as fhandle:
        fhandle.write(str(finalimage.get_potential_energy()))

    return str(atoms), open('all.traj', 'r').read().encode('hex'), finalimage.get_potential_energy()


def atoms_to_hex(atoms):
    '''
    Turn an atoms object into a hex string so that we can pass it through fireworks

    Input:
        atoms   The ase.Atoms object that you want to hex encode
    '''
    # We need to write the atoms object into a file before encoding it. But we don't
    # want multiple calls to this function to interfere with each other, so we generate
    # a random file name via uuid to reduce this risk. Then we delete it.
    fname = str(uuid.uuid4()) + '.traj'
    atoms.write(fname)
    with open(fname) as fhandle:
        _hex = fhandle.read().encode('hex')
        os.remove(fname)
    return _hex


def hex_to_file(fname_out, atomHex):
    '''
    Write a hex string into a file. One application is to unpack hexed atoms objects in
    local fireworks job directories
    '''
    # Dump the hex encoded string to a local file
    with open(fname_out, 'w') as fhandle:
        fhandle.write(atomHex.decode('hex'))
