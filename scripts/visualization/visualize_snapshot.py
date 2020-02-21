import glob
import os
import sys

import numpy as np

import daisy
import neuroglancer
import math

neuroglancer.set_server_bind_address('0.0.0.0')


def add(s, a, name, shader=None, data=None, visible=True):
    if a is None:
        return

    if shader == 'rgb':
        shader = """void main() { emitRGB(vec3(toNormalized(getDataValue(0)), 
        toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}
    kwargs['visible'] = visible

    if shader is not None:
        kwargs['shader'] = shader
    data = a.data if data is None else data
    s.layers.append(
        name=name,
        layer=neuroglancer.LocalVolume(
            data=data,
            offset=a.roi.get_offset()[::-1],
            voxel_size=a.voxel_size[::-1]
        ),
        **kwargs)


def open_ds_wrapper(path, ds_name):
    """Returns None if ds_name does not exists """
    try:
        return daisy.open_ds(path, ds_name)
    except KeyError:
        print('dataset %s could not be loaded' % ds_name)
        return None


if __name__ == '__main__':
    """
    Script to display synapse predictions gunpowder snapshots in neuroglancer.
    Argument 1: iteration (put -1, to selected most recent snapshot.
    Argument 2: experiment name.
    Example Usage: python visualize_snapshot.py -1 setup04
    """

    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
        iteration = iteration if iteration > 0 else None
    else:
        iteration = None

    if len(sys.argv) > 2:
        experiment_name = sys.argv[2]
    else:
        experiment_name = 'setup00'

    print(iteration)
    base_dir_exp = '../train/'

    if iteration is None:
        list_of_files = glob.glob(
            os.path.join(base_dir_exp, experiment_name, 'snapshots', 'batch*'))
        snapshot_path = max(list_of_files, key=os.path.getctime)
        print(snapshot_path, 'selecting most recent')
    else:
        snapshot_path = os.path.join(base_dir_exp, experiment_name, 'snapshots',
                                     'batch_%i.hdf' % iteration)
        print(snapshot_path)

    raw = open_ds_wrapper(snapshot_path, 'volumes/raw')
    gt_postsynapse = open_ds_wrapper(snapshot_path, 'volumes/gt_post_indicator')
    gt_vectormap = open_ds_wrapper(snapshot_path, 'volumes/gt_postpre_vectors')
    pred_indicator = open_ds_wrapper(snapshot_path,
                                     'volumes/pred_post_indicator')
    pred_dirvector = open_ds_wrapper(snapshot_path,
                                     'volumes/pred_postpre_vectors')
    post_loss_weight = open_ds_wrapper(snapshot_path,
                                       'volumes/post_loss_weight')
    vectors_mask = open_ds_wrapper(snapshot_path, 'volumes/vectors_mask')
    indicator_gradients = open_ds_wrapper(snapshot_path,
                                          'volumes/post_indicator_gradients')
    dirvector_gradients = open_ds_wrapper(snapshot_path,
                                          'volumes/partner_vectors_gradients')
    print(
        np.count_nonzero(gt_postsynapse.data), 'number of postsynaptic voxels')

    if gt_postsynapse is not None:
        gt_postsynapse.data = np.array(gt_postsynapse.data, dtype=np.uint32)
    pred_shader, gt_shader = None, None
    if pred_dirvector is not None:
        nmfactor = 1
        clipvalue = 100
        # clamp: values are clipped, otherwise not enough signal
        pred_shader = """void main() {{ emitRGB(vec3((
            clamp(getDataValue(0)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
            clamp(getDataValue(1)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
            clamp(getDataValue(2)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f})); }}""".format(
            str(nmfactor), clipvalue, clipvalue * 2)

    if gt_vectormap is not None:
        gt_vectormap.data = np.array(gt_vectormap.data, dtype=np.float32)
        nmfactor = 1
        clipvalue = 100
        # clamp: values are clipped, otherwise not enough signal
        gt_shader = """void main() {{ emitRGB(vec3((
            clamp(getDataValue(0)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
            clamp(getDataValue(1)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
            clamp(getDataValue(2)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f})); }}""".format(
            str(nmfactor), clipvalue, clipvalue * 2)
    if pred_indicator is not None:
        # Apply sigmoid.
        pred_indicator.data = np.array(
            1 / (1 + np.exp(-np.array(pred_indicator.data))))

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add(s, raw, 'raw')
        add(s, pred_dirvector, 'pred_dirvec', shader=pred_shader)
        add(s, pred_indicator, 'pred_post')
        add(s, gt_vectormap, 'gt_dirvec', shader=gt_shader)
        add(s, gt_postsynapse, 'gt_post')
        add(s, post_loss_weight, 'post_loss_weight', visible=False)
        add(s, vectors_mask, 'vectors_mask', visible=False)
        add(s, indicator_gradients, 'indicator_gradients', visible=False)
        add(s, dirvector_gradients, 'dirvector_gradients', shader='rgb',
            visible=False)

    print(viewer.__str__())
