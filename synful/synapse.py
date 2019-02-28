import logging

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
