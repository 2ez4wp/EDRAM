from fuel.datasets import H5PYDataset
from fuel.transformers import *
from fuel.utils import find_in_data_path
import numpy as np
import scipy as sc


class MNISTCluttered(H5PYDataset):
    def fix_representation(data):
        features, locations, labels = data

        return (np.asarray([np.array(sc.misc.imresize(im[0], (12, 12)), ndmin=3) for im in features]),)

    default_transformers = (
        (Mapping, [fix_representation, ('features_coarse',)], {}),
        (ScaleAndShift, [1 / 255.0, 0], {'which_sources': ('features', 'features_coarse')}),
        (Cast, ['floatX'], {'which_sources': ('features', 'features_coarse', 'locations')}),
    )

    def __init__(self, which_sets, **kwargs):
        super(MNISTCluttered, self).__init__(file_or_path=find_in_data_path('mnist_cluttered.hdf5'), which_sets=which_sets, **kwargs)
