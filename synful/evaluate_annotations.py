import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import scipy

import daisy
import numpy as np
from lsd import local_segmentation
from pymongo import MongoClient
from scipy.spatial import KDTree
import pymaid

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
                 roi_file=None, syn_dir=None,
                 filter_redundant_dist_type='euclidean'):
        assert filter_redundant_id_type == 'seg' or filter_redundant_id_type == 'skel'
        assert filter_same_id_type == 'seg' or filter_same_id_type == 'skel'
        assert filter_redundant_dist_type == 'euclidean' or \
               filter_redundant_dist_type == 'geodesic'
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
        self.filter_redundant_dist_type = filter_redundant_dist_type

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
                if self.filter_redundant_dist_type == 'geodesic':
                    # Get skeleton
                    skeleton = pymaid.get_neurons([skel_id])
                else:
                    skeleton = None
                __, removed_ids = synapse.cluster_synapses(pred_synapses,
                                                           self.filter_redundant_dist_thr,
                                                           fuse_strategy='max_score',
                                                           id_type=self.filter_redundant_id_type,
                                                           skeleton=skeleton)
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
        settings['filter_redundant_dist_type'] = self.filter_redundant_dist_type
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
