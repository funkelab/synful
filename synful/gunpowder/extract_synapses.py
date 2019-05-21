import logging
import time

import gunpowder as gp
import numpy as np
from gunpowder import BatchFilter
from gunpowder.contrib.points import PreSynPoint, PostSynPoint

from .. import detection, synapse

logger = logging.getLogger(__name__)


class ExtractSynapses(BatchFilter):
    '''Extract synaptic partners from 2 prediction channels. One prediction map
    indicates the location (m_channel), the second map (z_channel) indicates the
    direction to its synaptic partner.

    Args:

        array (:class:``ArrayKey``):
            The key of the array to extract points from.

        array (:class:``ArrayKey``):
            The key of the array to extract vectors from.

        points (:class:``PointsKey``):
            The key of the presynaptic points to create.

        points (:class:``PointsKey``):
            The key of the postsynaptic points to create.

        settings (:class:``SynapseExtractionParameters``):
            Which settings to use to extract synapses.
    '''

    def __init__(self, m_array, d_array, srcpoints, trgpoints,
                 settings=None, context=120):

        self.m_array = m_array
        self.d_array = d_array
        self.srcpoints = srcpoints
        self.trgpoints = trgpoints
        self.settings = settings
        self.context = [context]

    def setup(self):

        m_roi = self.spec[self.m_array].roi
        # self.spec_src = gp.PointsSpec(roi=m_roi.copy())
        # self.spec_trg = gp.PointsSpec(roi=m_roi.copy())
        self.spec_src = gp.PointsSpec()
        self.spec_trg = gp.PointsSpec()

        self.provides(self.srcpoints, self.spec_src)
        self.provides(self.trgpoints, self.spec_trg)

        self.enable_autoskip()

    def prepare(self, request):

        context = self.context
        # dims = self.array_spec.roi.dims()
        # dims = self.spec_src.roi.dims()
        # dims = request[self.m_array].spec.roi.dims()
        dims = 3

        # if len(context) == 1:
        #     context = context.repeat(dims)
        assert type(context) == list
        if len(context) == 1:
            context = context * dims

        # request array in a larger area to get rasterization from outside
        # points
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
        logger.info('find lcoations %0.2f' % (time.time() - start_time))
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
        # np.stack([dchannel.data]*3)
        target_sites = detection.find_targets(predicted_syns, dchannel.data * 0,
                                              voxel_size=dchannel.spec.voxel_size)
        logger.info('find targets %0.2f' % (time.time() - start_time))

        # Synapses need to be shifted to the global ROI
        # (currently aligned with arrayroi)
        start_time = time.time()
        for loc in predicted_syns:
            loc += np.array(mchannel.spec.roi.get_begin())
        for loc in target_sites:
            loc += np.array(dchannel.spec.roi.get_begin())
        print(time.time() - start_time, 'forloop')

        synapses = synapse.create_synapses(predicted_syns, target_sites,
                                           scores=scores)

        srcroi = request[self.srcpoints].roi

        # Bring into gunpowder format
        srcpoints = {}
        trgpoints = {}
        syn_id = 0
        for syn in synapses:
            loc = gp.Coordinate(syn.location_pre)
            if srcroi.contains(syn.location_pre):
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
