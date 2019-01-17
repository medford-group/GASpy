'''
This module houses the functions needed to make and submit FireWorks rockets
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import pickle
import math
import numpy as np
import luigi
from .atoms_generators import GenerateGas, GenerateBulk, GenerateAdslabs
from .. import defaults
from ..mongo import make_atoms_from_doc
from ..utils import unfreeze_dict, turn_site_into_str
from ..fireworks_helper_scripts import make_firework, submit_fwork

GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS
SLAB_SETTINGS = defaults.SLAB_SETTINGS
ADSLAB_SETTINGS = defaults.ADSLAB_SETTINGS


class MakeGasFW(luigi.Task):
    '''
    This task will create and submit a gas relaxation for you.

    Args:
        gas_name        A string indicating which gas you want to relax
        vasp_settings   A dictionary containing your VASP settings
    '''
    gas_name = luigi.Parameter()
    vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])

    def requires(self):
        return GenerateGas(gas_name=self.gas_name)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the input atoms object
        with open(self.input().path, 'rb') as file_handle:
            doc = pickle.load(file_handle)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        vasp_settings = unfreeze_dict(self.vasp_settings)
        fw_name = {'calculation_type': 'gas phase optimization',
                   'gasname': self.gas_name,
                   'vasp_settings': vasp_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              vasp_settings=vasp_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


class MakeBulkFW(luigi.Task):
    '''
    This task will create and submit a bulk relaxation for you.

    Args:
        mpid            A string indicating the mpid of the bulk
        vasp_settings   A dictionary containing your VASP settings
    '''
    mpid = luigi.Parameter()
    vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return GenerateBulk(mpid=self.mpid)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the input atoms object
        with open(self.input().path, 'rb') as file_handle:
            doc = pickle.load(file_handle)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        vasp_settings = unfreeze_dict(self.vasp_settings)
        fw_name = {'calculation_type': 'unit cell optimization',
                   'mpid': self.mpid,
                   'vasp_settings': vasp_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              vasp_settings=vasp_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


class MakeAdslabFW(luigi.Task):
    '''
    This task will create and submit an adsorbate+slab (adslab) calculation

    Args:
        adsorption_site         A 3-tuple of floats containing the Cartesian
                                coordinates of the adsorption site you want to
                                make a FW for
        shift                   A float indicating the shift of the slab
        top                     A Boolean indicating whether the adsorption
                                site is on the top or the bottom of the slab
        vasp_settings           A dictionary containing your VASP settings
                                for the adslab relaxation
        adsorbate_name          A string indicating which adsorbate to use. It
                                should be one of the keys within the
                                `gaspy.defaults.ADSORBATES` dictionary. If you
                                want an adsorbate that is not in the dictionary,
                                then you will need to add the adsorbate to that
                                dictionary.
        rotation                A dictionary containing the angles (in degrees)
                                in which to rotate the adsorbate after it is
                                placed at the adsorption site. The keys for
                                each of the angles are 'phi', 'theta', and
                                psi'.
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to enumerate sites from
        miller_indices          A 3-tuple containing the three Miller indices
                                of the slab[s] you want to enumerate sites from
        min_xy                  A float indicating the minimum width (in both
                                the x and y directions) of the slab (Angstroms)
                                before we enumerate adsorption sites on it.
        slab_generator_settings We use pymatgen's `SlabGenerator` class to
                                enumerate surfaces. You can feed the arguments
                                for that class here as a dictionary.
        get_slab_settings       We use the `get_slabs` method of pymatgen's
                                `SlabGenerator` class. You can feed the
                                arguments for the `get_slabs` method here
                                as a dictionary.
        bulk_vasp_settings      A dictionary containing the VASP settings of
                                the relaxed bulk to enumerate slabs from
    '''
    adsorption_site = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    vasp_settings = luigi.DictParameter(ADSLAB_SETTINGS['vasp'])

    # Passed to `GenerateAdslabs`
    adsorbate_name = luigi.Parameter()
    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return GenerateAdslabs(adsorbate_name=self.adsorbate_name,
                               rotation=self.rotation,
                               mpid=self.mpid,
                               miller_indices=self.miller_indices,
                               min_xy=self.min_xy,
                               slab_generator_settings=self.slab_generator_settings,
                               get_slab_settings=self.get_slab_settings,
                               bulk_vasp_settings=self.bulk_vasp_settings)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the possible adslab structures and find the one that matches
        # the site, shift, and top values we're looking for
        with open(self.input().path, 'rb') as file_handle:
            adslab_docs = pickle.load(file_handle)
        doc = _find_matching_adslab_doc(adslab_docs=adslab_docs,
                                        adsorption_site=self.adsorption_site,
                                        shift=self.shift,
                                        top=self.top)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        vasp_settings = unfreeze_dict(self.vasp_settings)
        fw_name = {'calculation_type': 'slab+adsorbate optimization',
                   'adsorbate': self.adsorbate_name,
                   'adsorbate_rotation': self.rotation,
                   'adsorption_site': turn_site_into_str(self.adsorption_site),
                   'mpid': self.mpid,
                   'miller': self.miller_indices,
                   'shift': self.shift,
                   'top': self.top,
                   'slabrepeat': doc['slab_repeat'],
                   'vasp_settings': vasp_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              vasp_settings=vasp_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


def _find_matching_adslab_doc(adslab_docs, adsorption_site, shift, top):
    '''
    This helper function is used to parse through a list of documents created
    by the `GenerateAdslabs` task, and then find one that has matching values
    for site, shift, and top. If it doesn't find one, then it'll throw an
    error.  If there's more than one match, then it will just return the first
    one without any notification

    Args:
        adslab_docs     A list of dictionaryies created by `GenerateAdslabs`
        adsorption_site A 3-long sequence of floats indicating the Cartesian
                        coordinates of the adsorption site
        shift           A float indicating the shift (i.e., slab termination)
        top             A Boolean indicating whether or not the site is on
                        the top or the bottom of the slab
    Returns:
        doc     The first dictionary within the `adslab_docs` list that
                has matching site, shift, and top values
    '''
    for doc in adslab_docs:
        if np.allclose(doc['adsorption_site'], adsorption_site):
            if math.isclose(doc['shift'], shift):
                if doc['top'] == top:
                    return doc

    raise RuntimeError('You just tried to make an adslab FireWork rocket '
                       'that we could not enumerate. Try changing the '
                       'adsorption site, shift, top, or miller.')