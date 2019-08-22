from scipy.optimize import linear_sum_assignment
import numpy as np
import logging

logger = logging.getLogger(__name__)


def synaptic_partners_fscore(rec_annotations, gt_annotations, matching_threshold=400,
                             all_stats=False, use_only_pre=False, use_only_post=False):
    """Compute the f-score of the found synaptic partners. Original function
    from: https://github.com/cremi/cremi_python. Modified, such that it
    works with synful.Synapse.

    Parameters
    ----------

    rec_annotations: List of predicted synapses (synful.Synapse)

    gt_annotations: List of ground truth synapses (synful.Synapse)

    matching_threshold: float, world units
        Euclidean distance threshold to consider two synapses a potential
        match. Synapses that are `matching_threshold` or more untis apart
        from each other are not considered as potential matches.

    use_only_pre: whether to only consider the distance of presites for
        applying the matching_threshold.

    use_only_post: whether to only consider the distance of postsites for
        applying the matching_threshold.

    all_stats: boolean, optional
        Whether to also return precision, recall, FP, FN, and matches as a 6-tuple with f-score

    Returns
    -------

    fscore: float
        The f-score of the found synaptic partners.
    precision: float, optional
    recall: float, optional
    fp: int, optional
    fn: int, optional
    filtered_matches: list of tuples, optional
        The indices of the matches with matching costs.
    """

    # get cost matrix
    costs = cost_matrix(rec_annotations, gt_annotations, matching_threshold,
                        use_only_pre, use_only_post)

    # match using Hungarian method
    logger.debug("Finding cost-minimal matches...")
    matches = linear_sum_assignment(costs - np.amax(costs) - 1)
    matches = zip(matches[0], matches[1])  # scipy returns matches as numpy arrays

    filtered_matches = [(i, j, costs[i][j]) for (i, j) in matches if costs[i][j] <= matching_threshold]
    logger.debug(str(len(filtered_matches)) + " matches found")

    # unmatched in rec = FP
    fp = len(rec_annotations) - len(filtered_matches)

    # unmatched in gt = FN
    fn = len(gt_annotations) - len(filtered_matches)

    # all ground truth elements - FN = TP
    tp = len(gt_annotations) - fn

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    if (precision + recall) > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    if all_stats:
        return (fscore, precision, recall, fp, fn, filtered_matches)
    else:
        return fscore

def from_synapsematches_to_syns(matches, pred_synapses, gt_synapses):

    # True Positives PRED
    tp_ids_test = list(zip(*matches))[0] if len(matches) > 0 else []
    tp_ids_truth = list(zip(*matches))[1] if len(matches) > 0 else []
    tp_syns = [pred_synapses[index1] for index1 in tp_ids_test]

    # False Positives.
    test_ids = set(range(len(pred_synapses)))
    false_positive_ids = test_ids - set(tp_ids_test)
    fp_syns = [pred_synapses[ii] for ii in list(false_positive_ids)]

    # False Negative.
    truth_ids = set(range(len(gt_synapses)))
    false_negative_ids = truth_ids - set(tp_ids_truth)
    fn_syns_gt = [gt_synapses[ii] for ii in list(false_negative_ids)]

    # True Positives GT
    tp_syns_gt = [gt_synapses[index1] for index1 in tp_ids_truth]

    return tp_syns, fp_syns, fn_syns_gt, tp_syns_gt

def cost_matrix(rec, gt, matching_threshold, use_only_pre=False, use_only_post=False):
    logger.debug("Computing matching costs...")

    rec_locations = [(syn.location_pre, syn.location_post) for syn in rec]
    gt_locations = [(syn.location_pre, syn.location_post) for syn in gt]

    size = max(len(rec_locations), len(gt_locations))
    costs = np.zeros((size, size), dtype=np.float)
    costs[:] = 2 * matching_threshold
    num_potential_matches = 0
    for i in range(len(rec_locations)):
        for j in range(len(gt_locations)):
            c = cost(rec_locations[i], gt_locations[j], rec[i], gt[j], matching_threshold, use_only_pre, use_only_post)
            costs[i, j] = c
            if c <= matching_threshold:
                num_potential_matches += 1

    logger.debug(str(num_potential_matches) + " potential matches found")

    return costs


def cost(pre_post_location1, pre_post_location2, syn1, syn2,
         matching_threshold, use_only_pre, use_only_post):
    max_cost = 2 * matching_threshold

    # First check of the nodes are part of the same segment

    pre_label_same = syn1.id_segm_pre == syn2.id_segm_pre or syn1.id_skel_pre == syn2.id_skel_pre
    post_label_same = syn1.id_segm_post == syn2.id_segm_post or syn1.id_skel_post == syn2.id_skel_post
    if syn1.id_segm_pre == None or syn1.id_segm_post == None:
        pre_label_same = False  # If segm is not known, it is considered a False Positive.

    # pairs do not link the same segments
    if not pre_label_same or not post_label_same:
        return max_cost

    pre_dist = distance(pre_post_location1[0], pre_post_location2[0])
    post_dist = distance(pre_post_location1[1], pre_post_location2[1])
    if use_only_pre:
        if pre_dist > matching_threshold:
            return max_cost
        dist = pre_dist
    elif use_only_post:
        if post_dist > matching_threshold:
            return max_cost
        dist = post_dist
    else:
        if pre_dist > matching_threshold or post_dist > matching_threshold:
            return max_cost
        dist = 0.5 * (pre_dist + post_dist)

    return dist


def distance(a, b):
    return np.linalg.norm(np.array(list(a)) - np.array(list(b)))


def add(a, b):
    return tuple([a[d] + b[d] for d in range(len(b))])


def sub(a, b):
    return tuple([a[d] - b[d] for d in range(len(b))])
