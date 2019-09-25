import csv
import json
import logging
import multiprocessing as mp
import os
import sys

import daisy
import numpy as np
from lsd import local_segmentation
from pymongo import MongoClient
from scipy.spatial import KDTree

from . import database, synapse, evaluation

logger = logging.getLogger(__name__)


def csv_to_list(csvfilename, column):
    with open(csvfilename) as csvfile:
        data = list(csv.reader(csvfile))
    col_list = []
    for ii in range(1, len(data)):
        row = data[ii]
        col_list.append(int(row[column]))
    return col_list


class EvaluateAnnotations():

    def __init__(self, pred_db_name, pred_db_host, pred_db_col,
                 gt_db_name, gt_db_host, gt_db_col,
                 distance_upper_bound=None, skel_db_name=None,
                 skel_db_host=None, skel_db_col=None,
                 multiprocess=True, matching_threshold=400,
                 matching_threshold_only_post=False,
                 matching_threshold_only_pre=False,
                 skeleton_ids=None, res_db_host=None,
                 res_db_name=None, res_db_col=None,
                 res_db_col_summary=None,
                 filter_same_id=True, filter_same_id_type='seg',
                 filter_redundant=False,
                 filter_redundant_dist_thr=None, filter_redundant_id_type='seg',
                 only_input_synapses=False,
                 only_output_synapses=False, overwrite_summary=False,
                 seg_agglomeration_json=None,
                 roi_file=None, syn_dir=None):
        assert filter_redundant_id_type == 'seg' or filter_redundant_id_type == 'skel'
        assert filter_same_id_type == 'seg' or filter_same_id_type == 'skel'
        self.pred_db = pred_db_name
        self.pred_db_host = pred_db_host
        self.pred_db_col = pred_db_col
        self.gt_db_name = gt_db_name
        self.gt_db_host = gt_db_host
        self.gt_db_col = gt_db_col
        self.synapses = []
        self.seg_id_to_skel = {}
        self.seg_skel_to_nodes = {}
        self.distance_upper_bound = distance_upper_bound
        self.matching_threshold = matching_threshold
        self.skeleton_ids = skeleton_ids
        self.output_db_col = self.pred_db_col + '_skel_{}'.format('inf' if
                                                                  distance_upper_bound is None else distance_upper_bound)

        self.multiprocess = multiprocess

        assert not (
                only_input_synapses is True and only_output_synapses is True), 'both only_input_synapses and only_output_synapses is set to True, unclear what to do'
        # Evaluation settings
        self.filter_same_id = filter_same_id
        self.filter_redundant = filter_redundant
        self.filter_redundant_dist_thr = filter_redundant_dist_thr
        self.only_input_synapses = only_input_synapses
        self.only_output_synapses = only_output_synapses
        self.matching_threshold_only_post = matching_threshold_only_post
        self.matching_threshold_only_pre = matching_threshold_only_pre
        # Where to write out results to
        self.res_db_host = res_db_host
        self.res_db_name = res_db_name
        self.res_db_col = res_db_col
        self.res_db_col_summary = res_db_col_summary
        self.overwrite_summary = overwrite_summary
        self.skel_db_name = skel_db_name
        self.skel_db_host = skel_db_host
        self.skel_db_col = skel_db_col
        self.seg_agglomeration_json = seg_agglomeration_json
        self.roi_file = roi_file
        self.syn_dir = syn_dir
        self.filter_same_id_type = filter_same_id_type
        self.filter_redundant_id_type = filter_redundant_id_type

    def __match_position_to_closest_skeleton(self, position, seg_id, skel_ids):
        distances = []
        for skel_id in skel_ids:
            locations = [np.array(node['position']) for node in
                         self.seg_skel_to_nodes[(seg_id, skel_id)]]
            tree = KDTree(locations)
            dist = tree.query(x=np.array(position), k=1, eps=0, p=2,
                              distance_upper_bound=np.inf)[0]
            distances.append(dist)
        indexmin = np.argmin(np.array(distances))
        logger.debug('matching node to skeleton with distance: %0.2f '
                     'compared to average distance: %0.2f' % (
                         distances[indexmin], np.mean(distances)))
        if self.distance_upper_bound is not None:
            if distances[indexmin] > self.distance_upper_bound:
                logger.debug(
                    'synapse not mapped because distance {:0.2} '
                    'bigger than {:}'.format(
                        distances[indexmin], self.distance_upper_bound))
                return -2

        return skel_ids[indexmin]

    def match_synapses_to_skeleton(self, synapses):

        for ii, syn in enumerate(synapses):
            logger.debug('{}/{}'.format(ii, len(synapses)))
            # Allow to filter out synapses based on distance.
            skel_ids = np.unique(self.seg_id_to_skel.get(syn.id_segm_pre, []))
            if len(skel_ids) > 0:
                skel_ids = [
                    self.__match_position_to_closest_skeleton(syn.location_pre,
                                                              syn.id_segm_pre,
                                                              skel_ids)]
            syn.id_skel_pre = skel_ids[0] if len(skel_ids) > 0 else None

            skel_ids = np.unique(self.seg_id_to_skel.get(syn.id_segm_post, []))
            if len(skel_ids) > 0:
                skel_ids = [
                    self.__match_position_to_closest_skeleton(syn.location_post,
                                                              syn.id_segm_post,
                                                              skel_ids)]

            syn.id_skel_post = skel_ids[0] if len(skel_ids) > 0 else None
        syn_db = database.SynapseDatabase(self.pred_db,
                                          db_host=self.pred_db_host,
                                          db_col_name=self.output_db_col,
                                          mode='r+')
        syn_db.write_synapses(synapses)

    def get_cremi_score(self, score_thr=0):
        gt_db = database.SynapseDatabase(self.gt_db_name,
                                         db_host=self.gt_db_host,
                                         db_col_name=self.gt_db_col,
                                         mode='r')

        pred_db = database.SynapseDatabase(self.pred_db,
                                           db_host=self.pred_db_host,
                                           db_col_name=self.output_db_col,
                                           mode='r')

        client_out = MongoClient(self.res_db_host)
        db_out = client_out[self.res_db_name]
        db_out.drop_collection(
            self.res_db_col + '.thr{}'.format(1000 * score_thr))

        skel_ids = csv_to_list(self.skeleton_ids, 0)
        print(skel_ids)

        fpcountall, fncountall, predall, gtall, tpcountall, num_clustered_synapsesall = 0, 0, 0, 0, 0, 0

        pred_synapses_all = []
        for skel_id in skel_ids:
            logger.debug('evaluating skeleton {}'.format(skel_id))
            if not self.only_output_synapses and not self.only_input_synapses:
                pred_synapses = pred_db.synapses.find(
                    {'$or': [{'pre_skel_id': skel_id},
                             {'post_skel_id': skel_id}]})
                gt_synapses = gt_db.synapses.find(
                    {'$or': [{'pre_skel_id': skel_id},
                             {'post_skel_id': skel_id}]})

            elif self.only_input_synapses:
                pred_synapses = pred_db.synapses.find({'post_skel_id': skel_id})
                gt_synapses = gt_db.synapses.find({'post_skel_id': skel_id})
            elif self.only_output_synapses:
                pred_synapses = pred_db.synapses.find({'pre_skel_id': skel_id})
                gt_synapses = gt_db.synapses.find({'pre_skel_id': skel_id})
            else:
                raise Exception(
                    'Unclear parameter configuration: {}, {}'.format(
                        self.only_output_synapses, self.only_input_synapses))

            pred_synapses = synapse.create_synapses_from_db(pred_synapses)
            pred_synapses = [syn for syn in pred_synapses if
                             syn.score > score_thr]
            if self.filter_same_id:
                if self.filter_same_id_type == 'seg':
                    pred_synapses = [syn for syn in pred_synapses if
                                     syn.id_segm_pre != syn.id_segm_post]
                elif self.filter_same_id_type == 'skel':
                    pred_synapses = [syn for syn in pred_synapses if
                                     syn.id_skel_pre != syn.id_skel_post]

            if self.filter_redundant:
                assert self.filter_redundant_dist_thr is not None
                num_synapses = len(pred_synapses)
                __, removed_ids = synapse.cluster_synapses(pred_synapses,
                                                           self.filter_redundant_dist_thr,
                                                           fuse_strategy='max_score',
                                                           id_type=self.filter_redundant_id_type)
                pred_synapses = [syn for syn in pred_synapses if
                                 not syn.id in removed_ids]
                num_clustered_synapses = num_synapses - len(pred_synapses)
                logger.debug(
                    'num of clustered synapses: {}, skel id: {}'.format(
                        num_clustered_synapses, skel_id))
            else:
                num_clustered_synapses = 0

            logger.debug(
                'found {} predicted synapses'.format(len(pred_synapses)))

            gt_synapses = synapse.create_synapses_from_db(gt_synapses)
            stats = evaluation.synaptic_partners_fscore(pred_synapses,
                                                        gt_synapses,
                                                        matching_threshold=self.matching_threshold,
                                                        all_stats=True,
                                                        use_only_pre=self.matching_threshold_only_pre,
                                                        use_only_post=self.matching_threshold_only_post)
            fscore, precision, recall, fpcount, fncount, matches = stats

            tp_syns, fp_syns, fn_syns_gt, tp_syns_gt = evaluation.from_synapsematches_to_syns(
                matches, pred_synapses, gt_synapses)
            fpcountall += fpcount
            fncountall += fncount
            tpcountall += len(matches)
            predall += len(pred_synapses)
            gtall += len(gt_synapses)
            num_clustered_synapsesall += num_clustered_synapses
            db_dic = {
                'skel_id': skel_id,
                'tp_pred': [syn.id for syn in tp_syns],
                'tp_gt': [syn.id for syn in tp_syns_gt],
                'fp_pred': [syn.id for syn in fp_syns],
                'fn_gt': [syn.id for syn in fn_syns_gt],
                'fscore': stats[0],
                'precision': stats[1],
                'recall': stats[2],
                'fpcount': stats[3],
                'fncount': stats[4],
                'removed_ids': removed_ids,
            }

            db_out[self.res_db_col + '.thr{}'.format(1000 * score_thr)].insert(
                db_dic)
            pred_synapses_all.extend(pred_synapses)

        # Alsow write out synapses:
        syn_out = database.SynapseDatabase(self.res_db_name,
                                           db_host=self.res_db_host,
                                           db_col_name=self.res_db_col + '.syn_thr{}'.format(
                                               1000 * score_thr),
                                           mode='w')
        syn_out.write_synapses(pred_synapses_all)

        precision = float(tpcountall) / (tpcountall + fpcountall) if (
                                                                             tpcountall + fpcountall) > 0 else 0.
        recall = float(tpcountall) / (tpcountall + fncountall) if (
                                                                          tpcountall + fncountall) > 0 else 0.
        if (precision + recall) > 0:
            fscore = 2.0 * precision * recall / (precision + recall)
        else:
            fscore = 0.0

        # Collect all in a single document in order to enable quick queries.
        result_dic = {}
        result_dic['fscore'] = fscore
        result_dic['precision'] = precision
        result_dic['recall'] = recall
        result_dic['fpcount'] = fpcountall
        result_dic['fncount'] = fncountall
        result_dic['tpcount'] = tpcountall
        result_dic['predcount'] = predall
        result_dic['gtcount'] = gtall
        result_dic['score_thr'] = score_thr

        settings = {}
        settings['pred_db_col'] = self.pred_db_col
        settings['pred_db_name'] = self.pred_db_col
        settings['gt_db_col'] = self.gt_db_col
        settings['gt_db_name'] = self.gt_db_name
        settings['filter_same_id'] = self.filter_same_id
        settings['filter_same_id_type'] = self.filter_same_id_type
        settings['filter_redundant'] = self.filter_redundant
        settings['filter_redundant_id_type'] = self.filter_redundant_id_type
        settings['dist_thr'] = self.filter_redundant_dist_thr
        settings['skel_ids'] = self.skeleton_ids
        settings['matching_threshold'] = self.matching_threshold
        settings[
            'matching_threshold_only_post'] = self.matching_threshold_only_post
        settings[
            'matching_threshold_only_pre'] = self.matching_threshold_only_pre
        settings['only_output_synapses'] = self.only_output_synapses
        settings['only_input_synapses'] = self.only_input_synapses
        settings['num_clustered_synapses'] = num_clustered_synapsesall
        result_dic.update(settings)

        db_out[self.res_db_col_summary].insert_one(result_dic)

        print('final fscore {:0.2}'.format(fscore))
        print('final precision {:0.2}, recall {:0.2}'.format(precision, recall))

    def evaluate_synapse_complete(self, score_thresholds):
        if self.overwrite_summary:
            client_out = MongoClient(self.res_db_host)
            db_out = client_out[self.res_db_name]
            db_out.drop_collection(self.res_db_col_summary)
            client_out.drop_database(self.res_db_name)

        if self.multiprocess:
            pool = mp.Pool(10)
            pool.map(self.get_cremi_score, score_thresholds)
            pool.close()
            pool.join()
        else:
            for score_thr in score_thresholds:
                self.get_cremi_score(score_thr)

    def add_skel_ids_daisy(self, roi_core, roi_context, seg_thr,
                           seg_ids_ignore):
        """Maps synapses to ground truth skeletons and writes them into a
        database.
        This function creates a local segmentation in order to map synapses to
        skeletons. It uses euclidean distance to cope with ambigious cases, eg. if one
        segment contains multiple skeletons, the pre/post site is mapped to
        the closest skeleton in euclidean space. If distance_upper_bound is set,
        only synapses are mapped to a skeleton if it closer than
        distance_upper_bound. Synapses that have been not mapped because of
        distance_upper_bound are marked with a skel_id of -2.

        Args:
            roi_core: (``daisy.ROI``): The ROI that is used to read in the
                synapses.
            roi_context (``daisy.ROI``): The ROI that is used to read in
                skeletons and ground truth synapses, that are used for mapping.
            seg_thr (``float``): Edge score threshold used for agglomerating
                fragments to produce a local segmentation used for mapping.
            seg_ids_ignore (``list`` of ``int``):
                List of ids that are not used for mapping. Eg. all skeletons
                whose seg id are in seg_ids_ignore are removed and not used
                for mapping.

        """
        with open(self.seg_agglomeration_json) as f:
            seg_config = json.load(f)

        # Get actual segmentation ROI
        seg = daisy.open_ds(seg_config['fragments_file'],
                            seg_config['fragments_dataset'])

        # This reads in all synapses, where postsynaptic site is in ROI, but
        # it is not guaranteed, that presynaptic site is also in ROI.
        synapses = synapse.read_synapses_in_roi(self.syn_dir,
                                                roi_core)

        # Make sure to only look at synapses that are inside segmentation ROI.
        synapses = [syn for syn in synapses if
                    seg.roi.contains(syn.location_pre)
                    and seg.roi.contains(syn.location_post)]

        if len(synapses) == 0:
            logger.debug('no synapse in roi')
            return 0

        pre_locations = [daisy.Coordinate(syn.location_pre) for syn in synapses]
        post_locations = [daisy.Coordinate(syn.location_post) for syn in
                          synapses]
        # Compute Bounding box for pre_locations
        z_min, y_min, x_min = np.min(np.array(pre_locations), axis=0)
        z_max, y_max, x_max = np.min(np.array(pre_locations), axis=0)

        roi_big = daisy.Roi((z_min, y_min, x_min),
                            (z_max - z_min, y_max - y_min, x_max - x_min))
        roi_big = roi_big.union(roi_context)

        roi_big = roi_big.snap_to_grid(seg.voxel_size)
        roi_big = seg.roi.intersect(roi_big)

        # Load skeletons.
        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')
        nodes = gt_db.read_nodes(roi_context)
        logger.info('number of skel nodes {}'.format(len(nodes)))
        if len(nodes) == 0:
            return 0

        logger.debug('creating a local segmentation')
        locseg = local_segmentation.LocalSegmentationExtractor(**seg_config)
        seg = locseg.get_local_segmentation(roi_big, seg_thr)

        seg_id_to_skel = {}
        seg_skel_to_nodes = {}
        for node in nodes:
            seg_id = seg[daisy.Coordinate(node['position'])]
            seg_id_to_skel.setdefault(seg_id, [])
            seg_id_to_skel[seg_id].append(node['neuron_id'])

            seg_skel_to_nodes.setdefault((seg_id, node['neuron_id']), [])
            seg_skel_to_nodes[(seg_id, node['neuron_id'])].append(node)

        # Also add ground truth connectors.
        gt_db = database.SynapseDatabase(self.gt_db_name,
                                         db_host=self.gt_db_host,
                                         db_col_name=self.gt_db_col,
                                         mode='r')
        gt_synapses = gt_db.read_synapses(roi=roi_big)
        gt_synapses = synapse.create_synapses_from_db(gt_synapses)
        gt_synapses = [syn for syn in gt_synapses if
                       seg.roi.contains(syn.location_pre)
                       and seg.roi.contains(syn.location_post)]
        logger.debug('number of catmaid synapses: {}'.format(len(gt_synapses)))
        for gt_syn in gt_synapses:
            seg_id = seg[daisy.Coordinate(gt_syn.location_pre)]
            seg_id_to_skel.setdefault(seg_id, [])
            seg_id_to_skel[seg_id].append(gt_syn.id_skel_pre)
            seg_skel_to_nodes.setdefault((seg_id, gt_syn.id_skel_pre), [])
            seg_skel_to_nodes[(seg_id,
                               gt_syn.id_skel_pre)].append(
                {'position': gt_syn.location_pre})

            seg_id = seg[daisy.Coordinate(gt_syn.location_post)]
            seg_id_to_skel.setdefault(seg_id, [])
            seg_id_to_skel[seg_id].append(gt_syn.id_skel_post)
            seg_skel_to_nodes.setdefault((seg_id, gt_syn.id_skel_post), [])
            seg_skel_to_nodes[(seg_id,
                               gt_syn.id_skel_post)].append(
                {'position': gt_syn.location_post})

        for seg_id in seg_ids_ignore:
            if seg_id in seg_id_to_skel:
                del seg_id_to_skel[seg_id]
        self.seg_id_to_skel = seg_id_to_skel
        self.seg_skel_to_nodes = seg_skel_to_nodes

        pre_ids = [seg[pre_loc] for pre_loc in pre_locations]
        post_ids = [seg[post_loc] for post_loc in post_locations]
        syn_on_skels = []
        for ii, syn in enumerate(synapses):
            pre_id = pre_ids[ii]
            post_id = post_ids[ii]
            # Test whether segment intersects with a skeleton
            skel_syn_pre = seg_id_to_skel.get(pre_id, False)
            skel_syn_post = seg_id_to_skel.get(post_id, False)
            if skel_syn_pre or skel_syn_post:
                syn.id_segm_pre = pre_id
                syn.id_segm_post = post_id
                syn_on_skels.append(syn)

        logger.debug(
            'matching {} synapses to skeletons, original number of synapses {}'.format(
                len(syn_on_skels), len(synapses)))
        self.match_synapses_to_skeleton(syn_on_skels)

    def add_skel_ids(self, seg_ids_ignore=[]):
        """Maps synapses to ground truth skeletons and writes them into a
        database.
        It is assumend that each ground truth neuron and each pre and
        postsynaptic site has already a segmentation ID assigned. This function
        uses euclidean distance to cope with ambigious cases, eg. if one
        segment contains multiple skeletons, the pre/post site is mapped to
        the closest skeleton in euclidean space. If distance_upper_bound is set,
        only synapses are mapped to a skeleton if it closer than
        distance_upper_bound.

        Args:
            seg_ids_ignore (``list`` of ``int``):
                List of ids that are not used for mapping. Eg. all skeletons
                whose seg id are in seg_ids_ignore are removed from further
                analysis and synapses with seg_id in seg_ids_ignore are not
                written out to database.

        """

        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')

        pred_db = database.SynapseDatabase(self.pred_db,
                                           db_host=self.pred_db_host,
                                           db_col_name=self.pred_db_col,
                                           mode='r')

        nodes = gt_db.read_nodes()
        seg_id_to_skel = {}
        seg_skel_to_nodes = {}
        for node in nodes:
            seg_id_to_skel.setdefault(node['seg_id'], [])
            seg_id_to_skel[node['seg_id']].append(node['neuron_id'])

            seg_skel_to_nodes.setdefault((node['seg_id'], node['neuron_id']),
                                         [])
            seg_skel_to_nodes[(node['seg_id'], node['neuron_id'])].append(node)

        for seg_id in seg_ids_ignore:
            if seg_id in seg_id_to_skel:
                del seg_id_to_skel[seg_id]
        self.seg_id_to_skel = seg_id_to_skel
        self.seg_skel_to_nodes = seg_skel_to_nodes

        synapses = pred_db.synapses.find({'$or': [
            {'pre_seg_id': {'$in': list(seg_id_to_skel.keys())}},
            {'post_seg_id': {'$in': list(seg_id_to_skel.keys())}},
        ]})
        synapses = synapse.create_synapses_from_db(synapses)

        logger.debug('found {} synapses '.format(len(synapses)))
        logger.info('Overwriting {}/{}/{}'.format(self.pred_db,
                                                  self.pred_db_host,
                                                  self.output_db_col))
        syn_db = database.SynapseDatabase(self.pred_db,
                                          db_host=self.pred_db_host,
                                          db_col_name=self.output_db_col,
                                          mode='w')

        batch_size = 100
        procs = []
        for ii in range(0, len(synapses), batch_size):
            if self.multiprocess:
                proc = mp.Process(
                    target=lambda: self.match_synapses_to_skeleton(
                        synapses[ii:ii + batch_size])
                )
                procs.append(proc)
                proc.start()
            else:
                self.match_synapses_to_skeleton(synapses[ii:ii + batch_size])

        if self.multiprocess:
            for proc in procs:
                proc.join()

    def add_skel_ids_majority_vote(self):
        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')

        pred_db = database.SynapseDatabase(self.pred_db,
                                           db_host=self.pred_db_host,
                                           db_col_name=self.pred_db_col,
                                           mode='r')
        nodes = gt_db.read_nodes()
        seg_id_to_skel = {}

        for node in nodes:
            seg_id_to_skel.setdefault(node['seg_id'], [])
            seg_id_to_skel[node['seg_id']].append(node['neuron_id'])

        # If we had a perfect segmentation, we could create a simple mapping
        # from segmentation id to skeleton id. Here, we use majority voting for
        # cases in which multiple skeletons are mapped to the same
        # segmentation id (corresponds to a merger).
        counter = 0
        for seg_id, skel_ids in seg_id_to_skel.items():
            # Find merges.
            unique_skels = np.unique(skel_ids)
            if len(unique_skels) > 1:
                counter += 1
                num_nodes_max = 0
                skel_id_winner = None
                for skel_id in unique_skels:
                    num_nodes = skel_ids.count(skel_id)
                    if num_nodes > num_nodes_max:
                        skel_id_winner = skel_id
                        num_nodes_max = num_nodes

                seg_id_to_skel[seg_id] = skel_id_winner
            else:
                seg_id_to_skel[seg_id] = skel_ids[0]

        print('num of merges {}/{}'.format(counter, len(seg_id_to_skel)))

        synapses = pred_db.synapses.find({'$or': [
            {'pre_seg_id': {'$in': list(seg_id_to_skel.keys())}},
            {'post_seg_id': {'$in': list(seg_id_to_skel.keys())}},
        ]})
        synapses = synapse.create_synapses_from_db(synapses)
        print('found {} synapses '.format(len(synapses)))

        for syn in synapses:
            syn.id_skel_pre = seg_id_to_skel.get(syn.id_segm_pre, None)
            syn.id_skel_post = seg_id_to_skel.get(syn.id_segm_post, None)

        self.synapses = synapses

    def write_synapses(self):
        syn_db = database.SynapseDatabase(self.pred_db,
                                          db_host=self.pred_db_host,
                                          db_col_name=self.pred_db_col + '_skel',
                                          mode='w')
        syn_db.write_synapses(self.synapses)
