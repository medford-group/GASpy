'''
This submodule contains various functions that help us manage
and interact with FireWorks.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import os
import warnings
import uuid
from datetime import datetime
import getpass
import pandas as pd
import ase.io
from fireworks import (Firework,
                       LaunchPad,
                       PyTask,
                       FileWriteTask,
                       ScriptTask,
                       Workflow)
from .utils import print_dict, read_rc


def get_launchpad():
    '''
    This function returns an instance of a `fireworks.LaunchPad` object that is
    connected to our FireWorks launchpad.

    Returns:
        lpad    An instance of a `fireworks.LaunchPad` object
    '''
    configs = read_rc('fireworks_info.lpad')
    configs['port'] = int(configs['port'])  # Make sure that the port is an integer
    lpad = LaunchPad(**configs)
    return lpad


def find_n_rockets(query, dft_settings, _testing=False):
    '''
    This function will check if we have something currently running in our
    FireWorks launcher. It will also warn you if we have a lot of fizzles.

    Args:
        query           A dictionary that can be passed as a `query` argument
                        to the `fireworks` collection of our FireWorks database.
        dft_settings    A dictionary of dft settings. These will be
                        automatically parsed into the `query` argument.
        _testing        Boolean indicating whether or not you are currently
                        doing a unit test. You should probably not be
                        changing the default from False.
    Returns:
        n_running   An integer for how many FireWorks are running that match
                    the query
        n_fizzles   An integer for how many FireWorks have fizzled that match
                    the query
    '''
    # Parse the DFT settings into the FireWorks query, then grab the docs
    for key, value in dft_settings.items():
        query['name.dft_settings.%s' % key] = value
    docs = _get_firework_docs(query=query, _testing=_testing)

    # Warn the user if there are a bunch of fizzles
    n_fizzles = __get_n_fizzles(docs)

    # Check if we are running currently. 'COMPLETED' is considered running
    # because it's offically done when it's in our atoms collection, not
    # when it's done in FireWorks
    running_states = set(['COMPLETED', 'READY', 'RESERVED', 'RUNNING', 'PAUSED'])
    docs_running = [doc for doc in docs if doc['state'] in running_states]
    n_running = len(docs_running)

    return n_running, n_fizzles


def _get_firework_docs(query, _testing):
    '''
    This function will get some documents from our FireWorks database.

    Args:
        query       A dictionary that can be passed as a `query` argument
                    to the `fireworks` collection of our FireWorks database.
        _testing    Boolean indicating whether or not you are currently
                    doing a unit test. You should probably not be
                    changing the default from False.
    Returns:
        docs    A list of dictionaries (i.e, Mongo documents) obtained
                from the `fireworks` collection of our FireWorks Mongo.
    '''
    lpad = get_launchpad()

    # Grab the correct collection, depending on whether or not we are
    # unit testing
    if _testing is False:
        collection = lpad.fireworks
    else:
        collection = lpad.fireworks.database.get_collection('unit_testing_fireworks')

    try:
        docs = list(collection.find(query))
    finally:    # Make sure we close the connection
        collection.database.client.close()
    return docs


def __get_n_fizzles(docs):
    '''
    Get the number of times a FireWork has fizzled.

    Arg:
        docs    A list of dictionaries that we got from our FireWorks database
    Returns:
        An integer for how many times this FireWork has fizzled.
    '''
    # Find the FireWork IDs of each of the fizzles
    docs_fizzled = [doc for doc in docs if doc['state'] == 'FIZZLED']
    fwids_fizzled = [doc['fw_id'] for doc in docs_fizzled]
    if len(docs_fizzled) > 0:
        message = ('We have fizzled a calculation %i time[s] so far:\n'
                   % (len(fwids_fizzled)))

        # Print out where it failed
        launch_docs = _get_launch_docs(fwids_fizzled)
        for launch_doc in launch_docs:
            fwid = launch_doc['fw_id']
            host = launch_doc['host']
            launch_dir = launch_doc['launch_dir']
            message += ('    fwid:  %i\n'
                        '        host:  %s\n'
                        '        launch_dir:  %s\n'
                        % (fwid, host, launch_dir))
        warnings.warn(message, RuntimeWarning)
    return len(fwids_fizzled)


def _get_launch_docs(fwids, _testing=False):
    '''
    Gets the document from the FireWorks launch collection

    Arg:
        fwids   A sequence of integers of the FireWorks ID number of the rocket
                you're trying to get information for.
    Returns:
        docs    A list of dictionaries from the FireWorks `launches` collection
                that have the keys 'fw_id', 'host', and 'launch_dir'.
    '''
    lpad = get_launchpad()

    # Grab the correct collection, depending on whether or not we are
    # unit testing
    if _testing is False:
        collection = lpad.launches
    else:
        collection = lpad.launches.database.get_collection('unit_testing_launches')

    try:
        docs = list(collection.find({'fw_id': {'$in': fwids}},
                                    {'fw_id': 1,
                                     'host': 1,
                                     'launch_dir': 1,
                                     '_id': 0}))
    finally:    # Make sure we close the connection
        collection.database.client.close()
    return docs


def make_firework(atoms, fw_name, dft_settings):
    '''
    This function identifies whether you're trying to do a VASP relaxation or a
    Quantum Espresso relaxation. It then calls the appropriate helper function
    to make a FireWorks rocket for you.

    Args:
        atoms           `ase.Atoms` object to relax
        fw_name         Dictionary of tags/etc to use as the FireWorks name
        dft_settings    Dictionary of DFT settings
    Returns:
        firework    An instance of a `fireworks.Firework` object that is set up
                    to perform a VASP relaxation
    '''
    # Warn the user if they're submitting a big one
    if len(atoms) > 80:
        warnings.warn('You are making a firework with %i atoms in it. This may '
                      'take awhile.' % len(atoms), RuntimeWarning)

    # We'll make one type of firework rocket for VASP, and another for Quantum
    # Espresso. And there's a QE subtype for RISM, too.
    if dft_settings['_calculator'] == 'vasp':
        firework = _make_vasp_firework(atoms, fw_name, dft_settings)
    elif dft_settings['_calculator'] == 'qe':
        firework = _make_qe_firework(atoms, fw_name, dft_settings,
                                     espresso_function='espresso_tools.run_qe')
    elif dft_settings['_calculator'] == 'rism':
        firework = _make_qe_firework(atoms, fw_name, dft_settings,
                                     espresso_function='espresso_tools.run_rism')
    elif dft_settings['_calculator'] == 'SPARC':
        firework = _make_sparc_firework(atoms, fw_name, dft_settings)

    # Yell if we try to run anything else
    else:
        raise RuntimeError('The %s calculator is not recognized, so we do not '
                           'know how the make a FireWork.'
                           % dft_settings['_calculator'])

    return firework


def _make_vasp_firework(atoms, fw_name, vasp_settings):
    '''
    This function creates a FireWorks rocket specifically tailored to do VASP
    calculations.

    Args:
        atoms           `ase.Atoms` object to relax
        fw_name         Dictionary of tags/etc to use as the FireWorks name
        vasp_settings    Dictionary of settings to pass to VASP
    Returns:
        firework    An instance of a `fireworks.Firework` object that is set up
                    to perform a VASP relaxation
    '''
    # Take the `vasp_functions` submodule in GASpy and then pass it out to each
    # FireWork rocket to use.
    vasp_filename = '/home/GASpy/gaspy/vasp_functions.py'
    with open(vasp_filename) as file_handle:
        vasp_functions_contents = file_handle.read()
    pass_vasp_functions = FileWriteTask(files_to_write=[{'filename': 'vasp_functions.py',
                                                         'contents': vasp_functions_contents}])

    # Convert the atoms object to a string so that we can pass it through
    # FireWorks, and then tell the FireWork rocket to use our `vasp_functions`
    # submodule to unpack the string
    atom_trajhex = encode_atoms_to_trajhex(atoms)
    read_atoms_file = PyTask(func='vasp_functions.hex_to_file',
                             args=['slab_in.traj', atom_trajhex])

    # Tell the FireWork rocket to perform the relaxation
    relax = PyTask(func='vasp_functions.runVasp',
                   args=['slab_in.traj', 'slab_relaxed.traj', vasp_settings],
                   stored_data_varname='opt_results')

    fw_name['user'] = getpass.getuser()
    firework = Firework([pass_vasp_functions, read_atoms_file, relax], name=fw_name)
    return firework


def _make_qe_firework(atoms, fw_name, qe_settings, espresso_function):
    '''
    This function creates a FireWorks rocket specifically tailored to do
    Quantum Espresso calculations.

    Args:
        atoms               `ase.Atoms` object to relax
        fw_name             Dictionary of tags/etc to use as the FireWorks name
        qe_settings         Dictionary of settings to pass to
                            espresso_tools/Quantum Espresso
        espresso_function   A string indicating the full path of the function
                            within espresso_tools that you want to run
    Returns:
        firework    An instance of a `fireworks.Firework` object that is set up
                    to perform a VASP relaxation
    '''
    # Clone the espresso_tools repository, which will help create the input
    # files
    clone_command = ('git clone git@github.com:ulissigroup/espresso_tools.git || '
                     'git clone https://github.com/ulissigroup/espresso_tools.git')
    clone_espresso_tools = ScriptTask.from_str(clone_command)

    # Tell the FireWork rocket to run the job using espresso_tools
    atom_trajhex = encode_atoms_to_trajhex(atoms)
    relax = PyTask(func=espresso_function,
                   args=[atom_trajhex, qe_settings],
                   stored_data_varname='opt_results')

    # espresso_tools is big. Let's remove it and then leave behind the commit
    # number we used (for traceability)
    cleaning_command = ('cd espresso_tools && '
                        'git rev-parse --verify HEAD > ../espresso_tools_version.log && '
                        'cd .. && '
                        'rm -rf espresso_tools')
    clean_up = ScriptTask.from_str(cleaning_command)

    fw_name['user'] = getpass.getuser()
    firework = Firework([clone_espresso_tools, relax, clean_up], name=fw_name)
    return firework

def _make_sparc_firework(atoms, fw_name, qe_settings, espresso_function):
    '''
    This function creates a FireWorks rocket specifically tailored to do
    SPARC-X calculations.

    Args:
        atoms               `ase.Atoms` object to relax
        fw_name             Dictionary of tags/etc to use as the FireWorks name
        sparc_settings      Dictionary of settings to pass to
                            espresso_tools/Quantum Espresso
        sparc_function      A string indicating the full path of the function
                            within espresso_tools that you want to run
    Returns:
        firework    An instance of a `fireworks.Firework` object that is set up
                    to perform a VASP relaxation
    '''
    # Clone the espresso_tools repository, which will help create the input
    # files
    clone_command = ('git clone https://github.com/SPARC-X/pysparcx.git ||'
                     'git clone git@github.com:SPARC-X/pysparcx.git')
    clone_sparc_tools = ScriptTask.from_str(clone_command)

    # Tell the FireWork rocket to run the job using espresso_tools
    atom_trajhex = encode_atoms_to_trajhex(atoms)
    relax = PyTask(func=espresso_function,
                   args=[atom_trajhex, qe_settings],
                   stored_data_varname='opt_results')

    # espresso_tools is big. Let's remove it and then leave behind the commit
    # number we used (for traceability)
    cleaning_command = ('cd pysparcx && '
                        'git rev-parse --verify HEAD > ../sparc_version.log && '
                        'cd .. && '
                        'rm -rf py')
    clean_up = ScriptTask.from_str(cleaning_command)

    fw_name['user'] = getpass.getuser()
    firework = Firework([clone_sparc_tools, relax, clean_up], name=fw_name)
    return firework



def encode_atoms_to_trajhex(atoms):
    '''
    Encode a trajectory-formatted atoms object into a hex string.
    Differs from `encode_atoms_to_hex` since this method is hex-encoding
    the trajectory, not an atoms object.

    As of the writing of this docstring, we intend to use this mainly
    to store atoms objects in the FireWorks DB, *not* the GASdb (AKA AuxDB).

    Arg:
        atoms   ase.Atoms object to encode
    Output:
        hex_    A hex-encoded string object of the trajectory of the atoms object
    '''
    # Make the trajectory
    fname = read_rc('temp_directory') + str(uuid.uuid4()) + '.traj'
    atoms.write(fname)

    # Encode the trajectory
    with open(fname, 'rb') as fhandle:
        hex_ = fhandle.read().hex()

    # Clean up
    os.remove(fname)
    return hex_


def decode_trajhex_to_atoms(hex_, index=-1):
    '''
    Decode a trajectory-formatted atoms object into a hex string.

    As of the writing of this docstring, we intend to use this mainly
    to store atoms objects in the FireWorks DB, *not* the GASdb (AKA AuxDB).

    Arg:
        hex_    A hex-encoded string of a trajectory of atoms objects.
        index   Trajectories can contain multiple atoms objects.
                The `index` is used to specify which atoms object to return.
                -1 corresponds to the last image.
    Output:
        atoms   The decoded ase.Atoms object
    '''
    # Make the trajectory from the hex
    fname = read_rc('temp_directory') + str(uuid.uuid4()) + '.traj'
    with open(fname, 'wb') as fhandle:
        fhandle.write(bytes.fromhex(hex_))

    # Open up the atoms from the trajectory
    atoms = ase.io.read(fname, index=index)

    # Clean up
    os.remove(fname)
    return atoms


def submit_fwork(fwork, _testing=False):
    '''
    This function will package a FireWork into a workflow for you and then add
    it to our FireWorks launchpad.

    Args:
        fwork       Instance of a `fireworks.Firework` object
        _testing    Boolean indicating whether or not you're doing a unit test.
                    You probably shouldn't touch this.
    Returns:
        wflow   An instance of the `firework.Workflow` that was added to the
                FireWorks launch pad.
    '''
    wflow = Workflow([fwork], name='dft optimization')

    if not _testing:
        lpad = get_launchpad()
        lpad.add_wf(wflow)
        print('Submitted the following FireWork rocket:')
        print_dict(fwork.name, indent=1)

    return wflow


def get_atoms_from_fwid(fwid, index=-1):
    '''
    Given a Fireworks ID, this function will give you the initial `ase.Atoms`
    object and the final (post-relaxation) `ase.Atoms` object.

    Arg:
        fwid    Integer indicating the FireWorks ID that you're trying to get
                the atoms object for
        index   Integer referring to the index of the trajectory file that
                you want to pull the `ase.Atoms` object from. `0` will be
                the starting image; `-1` will be the final image; etc.
    Returns:
        atoms           The relaxed `ase.Atoms` object of the FireWork
        starting_atoms  The unrelaxed `ase.Atoms` object of the FireWork
    '''
    lpad = get_launchpad()
    fw = lpad.get_fw_by_id(fwid)
    atoms = get_atoms_from_fw(fw, index=index)
    return atoms


def get_atoms_from_fw(fw, index=-1):
    '''
    This function will return an `ase.Atoms` object given a Firework from our
    LaunchPad.

    Args:
        fw      Instance of a `fireworks.core.firework.Firework` class that
                should probably get obtained from our Launchpad
        index   Integer referring to the index of the trajectory file that
                you want to pull the `ase.Atoms` object from. `0` will be
                the starting image; `-1` will be the final image; etc.
    Returns
        atoms   `ase.Atoms` instance from the Firework you provided
    '''
    # There are different methods to pull it out whether it was a VASP
    # calculation or a Quantum Espresso calculation. Figure out which to use
    # and then call it.
    if fw.name['dft_settings']['_calculator'] == 'vasp':
        atoms = _get_atoms_from_vasp_fw(fw, index)
    elif fw.name['dft_settings']['_calculator'] == 'qe':
        atoms = _get_atoms_from_qe_fw(fw, index)
    return atoms


def _get_atoms_from_vasp_fw(fw, index=-1):
    '''
    This function will return an `ase.Atoms` object given a Firework from our
    LaunchPad that happened to use VASP to perform the relaxation.

    Note:  The relaxation often mangles the tags and constraints due to
    limitations in vasp() calculators. We fix this by getting the original
    `ase.Atoms` object that we gave to the Firework, and the putting the tags
    and constraints from the original object onto the decoded one.

    Args:
        fw      Instance of a `fireworks.core.firework.Firework` class that
                should probably get obtained from our Launchpad
        index   Integer referring to the index of the trajectory file that
                you want to pull the `ase.Atoms` object from. `0` will be
                the starting image; `-1` will be the final image; etc.
    Returns
        atoms   `ase.Atoms` instance from the Firework you provided
    '''
    # Get the `ase.Atoms` object from FireWork's results
    atoms_trajhex = fw.launches[-1].action.stored_data['opt_results'][1]
    atoms = decode_trajhex_to_atoms(atoms_trajhex, index=index)

    # Get the Firework task that was meant to convert the original hexstring to
    # a trajectory file. We'll get the original atoms from this task (in
    # hexstring format). Note that over the course of our use, we have had
    # different names for these FireWorks tasks, so we check for them all.
    function_names_of_hex_encoders = {'vasp_functions.hex_to_file',
                                      'fireworks_helper_scripts.atoms_hex_to_file',
                                      'fireworks_helper_scripts.atomsHexToFile'}
    trajhexes = [task['args'][1] for task in fw.spec['_tasks']
                 if task.get('func', '') in function_names_of_hex_encoders]

    # If there was not one task, then we're screwed
    if len(trajhexes) != 1:
        raise RuntimeError('We tried to get the atoms object\'s trajhex from a '
                           'FireWork, but could not find the FireWork task that '
                           'contains the trajhex (FWID %i)' % fw.fw_id)

    # We can grab the original trajhex and then transfer its tags & constraints
    # to the newly decoded atoms
    original_atoms = decode_trajhex_to_atoms(trajhexes[0])
    try:
        atoms.set_tags(original_atoms.get_tags())
        atoms.set_constraint(original_atoms.constraints)

    # Sometimes the length of the initial atoms and the final atoms are
    # different. If this happens, then add a more useful error message.
    except ValueError as error:
        raise ValueError('The number of atoms from beginning to end of '
                         'calculation has changed for FireWork ID %i'
                         % fw.fw_id).with_traceback(error.__traceback__)

    # Patch some old, kinda-broken atoms
    patched_atoms = __patch_old_atoms_tags(fw, atoms)

    return patched_atoms


def _get_atoms_from_qe_fw(fw, index=-1):
    '''
    This function will return an `ase.Atoms` object given a Firework from our
    LaunchPad that happened to use Quantum Espresso to perform the relaxation.

    Args:
        fw      Instance of a `fireworks.core.firework.Firework` class that
                should probably get obtained from our Launchpad
        index   Integer referring to the index of the trajectory file that
                you want to pull the `ase.Atoms` object from. `0` will be
                the starting image; `-1` will be the final image; etc.
    Returns
        atoms   `ase.Atoms` instance from the Firework you provided
    '''
    # Get the `ase.Atoms` object from FireWork's results
    atoms_trajhex = fw.launches[-1].action.stored_data['opt_results'][1]
    atoms = decode_trajhex_to_atoms(atoms_trajhex, index=index)

    # Get the Firework task that was meant to run Quantum Espresso. It takes
    # the atoms object hexstring as an argument. We'll get the original atoms
    # from this argument.
    trajhexes = [task['args'][0] for task in fw.spec['_tasks']
                 if task.get('func', '') == 'espresso_tools.run_qe']

    # If there was no match, then we're screwed
    if len(trajhexes) != 1:
        raise RuntimeError('We tried to get the atoms object\'s trajhex from a '
                           'FireWork, but could not find the FireWork task '
                           'argument that contains the trajhex (FWID %i)'
                           % fw.fw_id)

    # We can grab the original trajhex and then transfer its tags & constraints
    # to the newly decoded atoms
    original_atoms = decode_trajhex_to_atoms(trajhexes[0])
    try:
        atoms.set_tags(original_atoms.get_tags())
        atoms.set_constraint(original_atoms.constraints)

    # Sometimes the length of the initial atoms and the final atoms are
    # different. If this happens, then add a more useful error message.
    except ValueError as error:
        raise ValueError('The number of atoms from beginning to end of '
                         'calculation has changed for FireWork ID %i'
                         % fw.fw_id).with_traceback(error.__traceback__)

    return atoms


def __patch_old_atoms_tags(fw, atoms):
    '''
    In an older version of GASpy, we did not use tags to identify whether an
    atom was part of the slab or an adsorbate. We fix that by setting the tags
    correctly here.

    Args:
        fw      Instance of a `fireworks.core.firework.Firework` class that
                should probably get obtained from our Launchpad
        atoms   Instance of the `ase.Atoms` object
    '''
    if (fw.created_on < datetime(2017, 7, 20) and
            fw.name['calculation_type'] == 'slab+adsorbate optimization'):

        # In this old version, the adsorbates were added onto the slab. Thus
        # the slab atoms came before the adsorbate atoms in the indexing. We
        # use this information to figure out what the tags are supposed to be.
        n_ads_atoms = len(fw.name['adsorbate'])
        n_slab_atoms = len(atoms) - n_ads_atoms
        tags = [0]*n_slab_atoms
        tags.extend([1]*n_ads_atoms)

        atoms.set_tags(tags)
    return atoms


def defuse_lost_runs():
    '''
    Sometimes FireWorks desynchronizes with the job management systems, and runs
    become "lost". This function finds and clears them
    '''
    # Find the lost runs
    lpad = get_launchpad()
    lost_launch_ids, lost_fw_ids, inconsistent_fw_ids = lpad.detect_lostruns()

    # We reverse the list, because early-on in the list there's some really bad
    # launches that cause this script to hang up. If we run in reverse, then
    # it should be able to get the recent ones.
    # TODO:  Find a better solution
    lost_fw_ids.reverse()

    # Defuse them
    for _id in lost_fw_ids:
        lpad.defuse_fw(_id)


def check_jobs_status(user_ID, num_jobs):
    '''
    This function returns the status of the submitted FW_jobs as a pandas
    dataframe. The job status are displayed in reversed order (last job to
    first job).

    For example, if Zack submitted 2 jobs (with fwid 10000, 10001),
    and wants to check their status, he will get
    fwi    mpid    miller_index    shift    top    calculation_type             user       job status    directory
    10001  mp-XXX  [1, 1, 0]       0.0      True   slab+adsorbate optimization  'zulissi'  RUNNING       /home/zulissi/fireworks/block_2018-11-25-21-28...
    10000  mp-YYY  [1, 1, 0]       0.0      True   slab+adsorbate optimization  'zulissi'  RUNNING       /home-research/zulissi/fireworks/blocks/block_...

    Args:
        user:       Your cori user ID, which is usually your CMU andrew ID, input as a string.
                    For example: 'zulissi' or 'ktran'.
        num_jobs    Number of submitted job you want to check
    Returns:
        dataframe   A Pandas DataFrame that contains FW job status. Information includes:
                    user, mpid, miller index, shift, calculation_type (e.g slab_adsorbate
                    optimization, slab optimization), top, adsorbate (if any), job status
                    (e.g. COMPLETED, FIZZLED, READY, DEFUSED), and directories.
    '''
    lpad = get_launchpad()
    user_fwids = lpad.get_fw_ids({'name.user': user_ID})
    user_fwids.sort(reverse=True)

    fireworks_info = []
    for fwid in user_fwids[:num_jobs]:
        fw = lpad.get_fw_by_id(fwid)

        # EAFP to get the launch directory, which does not exists for unlaunched fireworks
        try:
            launch_dir = fw.launches[0].launch_dir
        except IndexError:
            launch_dir = ''

        fw_info = (fwid,
                   fw.name.get('mpid', ''),
                   fw.name.get('miller', ''),
                   fw.name.get('shift', ''),
                   fw.name.get('top', ''),
                   fw.name.get('calculation_type', ''),
                   fw.name.get('adsorbate', ''),
                   fw.name.get('user', ''),
                   fw.state,
                   launch_dir)
        fireworks_info.append(fw_info)

    data_labels = ('fwid',
                   'mpid',
                   'miller',
                   'shift',
                   'top',
                   'calculation_type',
                   'adsorbate',
                   'user',
                   'state',
                   'launch_dir')
    dataframe = pd.DataFrame(fireworks_info, columns=data_labels)
    pd.set_option('display.max_colwidth', -1)
    pd.options.display.max_rows = None

    return dataframe
