import logging
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class Synapse(object):
    """Represents a single synapse.
    """

    def __init__(self, id=None, id_segm_pre=None, id_segm_post=None,
                 location_pre=None, location_post=None, score=None,
                 id_skel_pre=None, id_skel_post=None):
        self.id = id
        self.id_segm_pre = id_segm_pre
        self.id_segm_post = id_segm_post
        self.location_pre = location_pre
        self.location_post = location_post
        self.id_skel_pre = id_skel_pre
        self.id_skel_post = id_skel_post
        self.score = score

    def __repr__(self):
        output_str = 'seg_ids: [%s, %s], skel_ids: [%s, %s]' % (
            str(self.id_segm_pre),
            str(self.id_segm_post),
            str(self.id_skel_pre),
            str(self.id_skel_post))
        return output_str


def create_synapses(sources, targets, scores=None):
    """Creates a list of synapses.

    Args:
        sources (list): List with source positions.
        targets (list): List with target positions.

    """
    assert len(sources) == len(targets)
    synapses = []
    score = None
    counter = 0
    for presite, postsite in zip(*[sources, targets]):
        if scores is not None:
            score = scores[counter]
        synapses.append(Synapse(location_pre=presite,
                                location_post=postsite,
                                score=score))
        counter += 1
    return synapses


def write_synapses_into_cremiformat(synapses, filename, offset=None,
                                    overwrite=False):
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    for synapse in synapses:
        types.extend(['presynaptic_site', 'postsynaptic_site'])
        ids.extend([id_nr, id_nr + 1])
        partners.extend([np.array((id_nr, id_nr + 1))])
        assert synapse.location_pre is not None and synapse.location_post is not None
        locations.extend(
            [np.array(synapse.location_pre), np.array(synapse.location_post)])
        id_nr += 2
    if overwrite:
        h5_file = h5py.File(filename, 'w')
    else:
        h5_file = h5py.File(filename, 'a')
    dset = h5_file.create_dataset('annotations/ids', data=ids,
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/locations',
                                  data=np.stack(locations, axis=0).astype(np.float32),
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                  data=np.stack(partners, axis=0).astype(np.uint32),
                                  compression='gzip')
    dset = h5_file.create_dataset('annotations/types', data=np.array(types, dtype='S'),
                                  compression='gzip')

    if offset is not None:
        h5_file['annotations'].attrs['offset'] = offset
    h5_file.close()
    logger.debug('File written to {}'.format(filename))

