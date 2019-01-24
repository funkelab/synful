import unittest

# from syntist.gunpowder import ProviderTest
from synful.gunpowder import AddPartnerVectorMap

from gunpowder import *
from gunpowder.contrib.points import PreSynPoint, PostSynPoint
from gunpowder.points import Points

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('synful.gunpowder').setLevel(logging.DEBUG)


class PointTestSource3D(BatchProvider):
    def __init__(self, points, partners, labels=None, voxel_size=None):
        self.voxel_size = voxel_size

        self.points = points

        self.partners = partners
        self.labels = labels

    def setup(self):

        self.provides(
            PointsKeys.PRESYN,
            PointsSpec(roi=Roi((0, 0, 0), (200, 200, 200))))

        self.provides(
            PointsKeys.POSTSYN,
            PointsSpec(roi=Roi((0, 0, 0), (200, 200, 200))))

        # self.provides(
        #     ArrayKeys.GT_LABELS,
        #     ArraySpec(
        #         roi=Roi((0, 0, 0), (200, 200, 200)),
        #         voxel_size=self.voxel_size))

    def provide(self, request):

        batch = Batch()

        roi_points = request[PointsKeys.PRESYN].roi
        trg_points = request[PointsKeys.POSTSYN].roi

        # get all pre points inside the requested ROI
        pre_points = {}
        post_points = {}
        syn_id = 0
        for pre_id, post_id in self.partners:
            loc = self.points[pre_id]
            if roi_points.contains(loc):
                pre_point = PreSynPoint(location=loc,
                                        partner_ids=[post_id],
                                        location_id=pre_id, synapse_id=syn_id)
                pre_points[pre_id] = pre_point
            loc = self.points[post_id]
            if trg_points.contains(loc):
                post_point = PostSynPoint(location=loc,
                                          partner_ids=[pre_id],
                                          location_id=post_id,
                                          synapse_id=syn_id)

                post_points[post_id] = post_point
            syn_id += 1

        batch.points[PointsKeys.PRESYN] = Points(
            pre_points,
            PointsSpec(roi=roi_points))
        batch.points[PointsKeys.POSTSYN] = Points(
            post_points,
            PointsSpec(roi=trg_points))

        # if ArrayKeys.GT_LABELS in request:
        #     roi_array = request[ArrayKeys.GT_LABELS].roi
        #
        #     spec = self.spec[ArrayKeys.GT_LABELS].copy()
        #
        #     spec.roi = roi_array
        #     batch.arrays[ArrayKeys.GT_LABELS] = Array(
        #         self.labels,
        #         spec=spec)

        return batch


class TestAddPartnerVectorMap(unittest.TestCase):
    def test_output_basics(self):

        vectormap = ArrayKey('VECTOR_MAP')
        presyn = PointsKey('PRESYN')
        postsyn = PointsKey('POSTSYN')

        voxel_size = Coordinate((10, 10, 10))
        spec = ArraySpec(voxel_size=voxel_size)

        add_vector_map = AddPartnerVectorMap(
            src_points=presyn,
            trg_points=postsyn,
            array=vectormap,
            radius=1, # 10 voxels,
            trg_context=20, # 10 voxels,,
            array_spec=spec
        )

        points = {
            1: Coordinate((40, 40, 40)),
            2: Coordinate((90, 90, 40)),
            3: Coordinate((70, 70, 70)), # should be ignored, as partner is far outside
            4: Coordinate((300, 300, 300)),
        }


        pipeline = (
            PointTestSource3D(points=points, partners=[(1, 2), (3, 4)], voxel_size=voxel_size) +
            add_vector_map
        )

        request = BatchRequest()

        roi = Roi((40, 40, 40), (80, 80, 80))

        request[presyn] = PointsSpec(roi=roi)
        request[postsyn] = PointsSpec(roi=roi)
        request[vectormap] = ArraySpec(roi=roi)


        with build(pipeline):
            batch = pipeline.request_batch(request)

        res = batch[vectormap]
        self.assertTrue((res.data[:, 0, 0, 0] == np.array([50, 50, 0])).all())
        # The rest should be all zero.
        self.assertEqual(np.count_nonzero(res.data[:, 1:, 1:, 1:]), 0)

    def test_output_outside_roi(self):

        vectormap = ArrayKey('VECTOR_MAP')
        presyn = PointsKey('PRESYN')
        postsyn = PointsKey('POSTSYN')

        voxel_size = Coordinate((5, 5, 5))
        spec = ArraySpec(voxel_size=voxel_size)

        add_vector_map = AddPartnerVectorMap(
            src_points=presyn,
            trg_points=postsyn,
            array=vectormap,
            radius=30,  # 10 voxels,
            trg_context=20,  # 10 voxels,,
            array_spec=spec
        )

        points = {
            1: Coordinate((30, 30, 30)), # should be included, since radius is large enough to reach into ROI.
            2: Coordinate((90, 90, 40)),
            3: Coordinate((70, 70, 70)), # should be included, since partner is within context
            4: Coordinate((90, 90, 50)),
        }

        pipeline = (
                PointTestSource3D(points=points, partners=[(1, 2), (3, 4)]) +
                add_vector_map
        )

        request = BatchRequest()

        roi = Roi((40, 40, 40), (80, 80, 80))

        request[presyn] = PointsSpec(roi=roi)
        request[postsyn] = PointsSpec(roi=roi)
        request[vectormap] = ArraySpec(roi=roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        res = batch[vectormap]
        self.assertTrue((res.data[:, 0, 0, 0] == np.array([50, 50, 0])).all())
        self.assertTrue((res.data[:, 6, 6, 6] == np.array([20, 20, -20])).all())


if __name__ == '__main__':
    unittest.main()
