from __future__ import division

import logging
import time

import gunpowder as gp
import numpy as np
from gunpowder import BatchFilter
from gunpowder.contrib.points import PreSynPoint, PostSynPoint
from funlib.math import cantor_number

from .. import detection, synapse, database

logger = logging.getLogger(__name__)


class ExtractSynapses(BatchFilter):
    '''Extract synaptic partners from 2 prediction channels. One prediction map
    indicates the location (m_channel), the second map (z_channel) indicates the
    direction to its synaptic partner. Optionally, writes it to a database.

    Args:

        array (:class:``ArrayKey``):
            The key of the array to extract points from.

        array (:class:``ArrayKey``):
            The key of the array to extract vectors from.

        points (:class:``PointsKey``):
            The key of the presynaptic points to create. Note, that only those
            synaptic partners will be in roi, where both pre and postsynaptic
            site are fully contained in ROI.

        points (:class:``PointsKey``):
            The key of the postsynaptic points to create.

        settings (:class:``SynapseExtractionParameters``):
            Which settings to use to extract synapses.

        db_name (``string``):
            If db_name and db_host provided, synapses are written out into
            database. Only those synapses are written out, where the postsynaptic
            site is contained in output ROI (the location of the presynaptic
            site does not matter).

        db_host (``string``):
            Database host.

        db_col_name (``string``):
            Name of the mongodb collection, that the synapses are written to.

        pre_to_post (``bool``):
            If set to True, it is assumed that m_array is indicating the
            presence of presynaptic location, and d_array is encoding the
            direction to the postsynaptic partner. If set to False, m_array
            indicates postsynaptic location, and d_array the direction to the
            presynaptic partner. (Only relevant for writing synapses out
            to a database.)

    '''

    def __init__(self, m_array, d_array, srcpoints, trgpoints,
                 settings=None, context=120,
                 db_name=None, db_host=None, db_col_name=None):
        if db_name is not None or db_host is not None:
            if db_host is None or db_name is None:
                logger.warning(
                    'If synapses are supposed to be written out to database, '
                    'both db_name and db_host must be provided')

        self.m_array = m_array
        self.d_array = d_array
        self.srcpoints = srcpoints
        self.trgpoints = trgpoints
        self.settings = settings
        if type(context) == tuple:
            context = list(context)
        if not type(context) == list:
            context = [context]
        self.context = context
        self.db_name = db_name
        self.db_host = db_host
        self.db_col_name = db_col_name
        self.pre_to_post = False

    def setup(self):

        self.spec_src = gp.PointsSpec()
        self.spec_trg = gp.PointsSpec()

        self.provides(self.srcpoints, self.spec_src)
        self.provides(self.trgpoints, self.spec_trg)

        self.enable_autoskip()

    def prepare(self, request):

        context = self.context
        dims = request[self.srcpoints].roi.dims()

        assert type(context) == list
        if len(context) == 1:
            context = context * dims

        # request array in a larger area to get predictions from outside
        # write roi
        m_roi = request[self.srcpoints].roi.grow(
            gp.Coordinate(context),
            gp.Coordinate(context))

        # however, restrict the request to the array actually provided
        # m_roi = m_roi.intersect(self.spec[self.m_array].roi)
        request[self.m_array] = gp.ArraySpec(roi=m_roi)

        # Do the same for the direction vector array.
        request[self.d_array] = gp.ArraySpec(roi=m_roi)

    def process(self, batch, request):

        srcpoints, trgpoints = self.__extract_synapses(batch, request)

        points_spec = self.spec[self.srcpoints].copy()
        points_spec.roi = request[self.srcpoints].roi
        batch.points[self.srcpoints] = gp.Points(data=srcpoints,
                                                 spec=points_spec)
        batch.points[self.trgpoints] = gp.Points(data=trgpoints,
                                                 spec=points_spec.copy())

        # restore requested arrays
        if self.m_array in request:
            batch.arrays[self.m_array] = batch.arrays[self.m_array].crop(
                request[self.m_array].roi)
        if self.d_array in request:
            batch.arrays[self.d_array] = batch.arrays[self.d_array].crop(
                request[self.d_array].roi)

    def __extract_synapses(self, batch, request):
        mchannel = batch[self.m_array]
        dchannel = batch[self.d_array]
        start_time = time.time()
        predicted_syns, scores = detection.find_locations(mchannel.data,
                                                          self.settings,
                                                          mchannel.spec.voxel_size)
        logger.debug('find locations %0.2f' % (time.time() - start_time))
        # Filter synapses for scores.
        new_scorelist = []
        if self.settings.score_thr is not None:
            filtered_list = []
            for ii, loc in enumerate(predicted_syns):
                score = scores[ii]
                if score > self.settings.score_thr:
                    filtered_list.append(loc)
                    new_scorelist.append(score)

            logger.debug(
                'filtered out %i' % (len(predicted_syns) - len(filtered_list)))
            predicted_syns = filtered_list
            scores = new_scorelist
        start_time = time.time()
        target_sites = detection.find_targets(predicted_syns, dchannel.data,
                                              voxel_size=dchannel.spec.voxel_size)
        logger.debug('find targets %0.2f' % (time.time() - start_time))

        # Synapses need to be shifted to the global ROI
        # (currently aligned with arrayroi)
        for loc in predicted_syns:
            loc += np.array(mchannel.spec.roi.get_begin())
        for loc in target_sites:
            loc += np.array(dchannel.spec.roi.get_begin())

        if self.pre_to_post:
            synapses = synapse.create_synapses(predicted_syns, target_sites,
                                               scores=scores)
        else:
            synapses = synapse.create_synapses(target_sites, predicted_syns,
                                               scores=scores)

        srcroi = request[self.srcpoints].roi

        if self.db_name is not None and self.db_host is not None:
            db_col_name = 'syn' if self.db_col_name is None else self.db_col_name
            nodes, edges = self.__from_synapses_to_nodes_and_edges(synapses,
                                                                   roi=srcroi)

            dag_db = database.DAGDatabase(self.db_name, self.db_host,
                                          db_col_name=db_col_name,
                                          mode='r+')
            dag_db.write_nodes(nodes)
            dag_db.write_edges(edges)

        # Bring into gunpowder format
        srcpoints = {}
        trgpoints = {}
        syn_id = 0
        for syn in synapses:
            loc = gp.Coordinate(syn.location_pre)
            if srcroi.contains(syn.location_pre) and srcroi.contains(
                    syn.location_post):  # TODO: currently, gunpowder complains
                # about points being outside ROI, thus can only provide synapses
                # where pre and point are inside ROI
                loc_index = syn_id * 2
                syn_point = PreSynPoint(location=loc,
                                        location_id=loc_index,
                                        synapse_id=syn_id,
                                        partner_ids=[loc_index + 1],
                                        props={'score': syn.score})
                srcpoints[loc_index] = syn_point
                loc = gp.Coordinate(syn.location_post)
                syn_point = PostSynPoint(location=loc,
                                         location_id=loc_index + 1,
                                         synapse_id=syn_id,
                                         partner_ids=[loc_index],
                                         props={'score': syn.score})
                trgpoints[loc_index + 1] = syn_point
                syn_id += 1
        return srcpoints, trgpoints

    def __from_synapse_to_node(self, synapse, id=None, pre=True):
        node = {'id': id}
        if pre:
            node['position'] = synapse.location_pre
        else:
            node['position'] = synapse.location_post
        node['score'] = np.double(synapse.score)

        return node

    def __from_synapses_to_nodes_and_edges(self, synapses, roi=None):
        nodes = []
        edges = []
        for synapse in synapses:
            post_node_inside = True
            if roi is not None:
                post_node_inside = roi.contains(synapse.location_post)

            if post_node_inside:
                id_bump = cantor_number(synapse.location_post)
                node_pre = self.__from_synapse_to_node(synapse,
                                                       id=int(-id_bump),
                                                       pre=True)
                node_post = self.__from_synapse_to_node(synapse,
                                                        id=int(id_bump),
                                                        pre=False)
                edge = {'source': int(-id_bump)}
                edge['target'] = int(id_bump)
                edges.append(edge)
                nodes.extend([node_pre, node_post])
        return nodes, edges
