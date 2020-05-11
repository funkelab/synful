import logging
import sqlite3

import matplotlib.pyplot as plt
import networkx as nx
import neuroglancer
import pandas as pd
from IPython.core.display import display, HTML

logger = logging.getLogger(__name__)

NG_BASE_LINK = 'https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B132267.5%2C63857.5%2C5229.5%5D%2C%22crossSectionScale%22:420.8181871611291%2C%22crossSectionDepth%22:-21.817278068177124%2C%22projectionScale%22:355033.5184786787%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig%22%2C%22name%22:%22fafb_v14%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe%22%2C%22name%22:%22fafb_v14_clahe%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://fafb-ffn1-20190805/segmentation%22%2C%22tab%22:%22segments%22%2C%22objectAlpha%22:0.87%2C%22colorSeed%22:449353638%2C%22segments%22:%5B%221%22%2C%227889893021%22%5D%2C%22name%22:%22fafb-ffn1-20190805%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://gs://neuroglancer-20191211_fafbv14_buhmann2019_li20190805%22%2C%22annotationColor%22:%22#e726da%22%2C%22crossSectionAnnotationSpacing%22:23.290450925474985%2C%22projectionAnnotationSpacing%22:10.657470105892225%2C%22shader%22:%22#uicontrol%20vec3%20preColor%20color%28default=%5C%22blue%5C%22%29%5Cn#uicontrol%20vec3%20postColor%20color%28default=%5C%22red%5C%22%29%5Cn#uicontrol%20float%20scorethr%20slider%28min=0%2C%20max=1000%29%5Cn#uicontrol%20int%20showautapse%20slider%28min=0%2C%20max=1%29%5Cn%5Cnvoid%20main%28%29%20%7B%5Cn%20%20setColor%28defaultColor%28%29%29%3B%5Cn%20%20setEndpointMarkerColor%28%5Cn%20%20%20%20vec4%28preColor%2C%201.0%29%2C%5Cn%20%20%20%20vec4%28postColor%2C%201.0%29%29%3B%5Cn%20%20setEndpointMarkerSize%285.0%2C%205.0%29%3B%5Cn%20%20setLineWidth%282.0%29%3B%5Cn%20%20if%20%28int%28prop_autapse%28%29%29%20%3E%20showautapse%29%20discard%3B%5Cn%20%20if%20%28prop_score%28%29%3Cscorethr%29%20discard%3B%5Cn%7D%5Cn%5Cn%22%2C%22shaderControls%22:%7B%22preColor%22:%22#c82fe8%22%2C%22postColor%22:%22#00e2b8%22%2C%22scorethr%22:80%7D%2C%22linkedSegmentationLayer%22:%7B%22pre_segment%22:%22fafb-ffn1-20190805%22%2C%22post_segment%22:%22fafb-ffn1-20190805%22%7D%2C%22filterBySegmentation%22:%5B%22post_segment%22%2C%22pre_segment%22%5D%2C%22name%22:%22synapses_buhmann2019%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22fafb-ffn1-20190805%22%7D%2C%22layout%22:%22xy-3d%22%7D'

""" Collection of tools to query predicted synapses intersected with a neuron segmentation"""


class QueryCircuit():
    def __init__(self, sqlite_path, sqltable='synlinks',
                 score_thr=60, filter_autapses=True):
        conn = sqlite3.connect(sqlite_path)
        self.cursor = conn.cursor()
        self.df = pd.DataFrame()
        self.sqltable = sqltable
        self.score_thr = score_thr
        self.filter_autapses = filter_autapses

    def init_with_seg_ids(self, seg_ids):
        self.__get_links(seg_ids)
        logger.info(f'Loaded {len(self.links)} links')

    def __get_links(self, seg_ids):
        seg_ids = [str(seg_id) for seg_id in seg_ids]
        cols = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y',
                'post_z', 'scores', 'segmentid_pre', 'segmentid_post',
                'cleft_scores']
        command = 'SELECT {} from {} WHERE (segmentid_pre IN ({})) OR (segmentid_post IN ({}));'.format(
            ','.join(cols), self.sqltable,
            ','.join(seg_ids), ','.join(seg_ids))
        logger.info(f'sql query: {command}')
        self.cursor.execute(command)
        pre_links = self.cursor.fetchall()
        links = pd.DataFrame.from_records(pre_links, columns=cols)
        if self.filter_autapses:
            links = links[links.segmentid_pre != links.segmentid_post]
        if self.score_thr > 0:
            links = links[links.scores >= self.score_thr]
        self.links = links

    def plot_input_output_sites(self, seg_id, input_site_color='#72b9cb',
                                output_site_color='#c12430'):
        pre_links = self.links[self.links.segmentid_pre == seg_id]
        post_links = self.links[self.links.segmentid_post == seg_id]
        fig = plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 22})
        ax = fig.add_subplot(111)
        ax.scatter(pre_links.pre_x, pre_links.pre_y, color=output_site_color,
                   s=2.5, alpha=1.0, label='output site')
        ax.scatter(post_links.post_x, post_links.post_y, color=input_site_color,
                   s=2.5, alpha=1.0, label='input site')
        plt.xticks([])
        plt.yticks([])
        plt.title('Input and output sites of seg id {}'.format(seg_id))
        plt.legend(markerscale=5., scatterpoints=1, fontsize=20)
        plt.show()

    def ng_link(self, seg_ids, urlbase=None):
        if urlbase is None:
            url = NG_BASE_LINK
        state = neuroglancer.parse_url(url)
        state.layers['fafb-ffn1-20190805'].segments = seg_ids
        url = neuroglancer.url_state.to_url(state)
        seg_ids = [str(seg_id) for seg_id in seg_ids]
        display(HTML(
            """<a href="{}">Neuroglancer link with seg ids {}</a>""".format(url,
                                                                            ','.join(
                                                                                seg_ids))))

    def links2nx(self, weight_threshold=0):
        links = pd.DataFrame(self.links)  # copy
        links = links[(links.segmentid_pre != 0) & (links.segmentid_post != 0)]
        links['edges'] = pd.Series(
            list(zip(links.segmentid_pre, links.segmentid_post)))
        c_edges = links.edges.value_counts()
        c_edges = c_edges[c_edges >= weight_threshold]
        nxg = nx.DiGraph()
        for k, v in c_edges.to_dict().items():
            nxg.add_edge(k[0], k[1], weight=v)
        return nxg

    def get_upstream_partners(self, seg_id, topk=5, weight_threshold=0):
        nxg = self.links2nx(weight_threshold=weight_threshold)
        upstream_nodes = list(nxg.neighbors(seg_id))
        return upstream_nodes[:min(len(upstream_nodes), topk)]

    def get_downstream_partners(self, seg_id, topk=5, weight_threshold=0):
        nxg = self.links2nx(weight_threshold=weight_threshold)
        downstream_nodes = list(nxg.predecessors(seg_id))
        return downstream_nodes[:min(len(downstream_nodes), topk)]

    def plot_circuit(self, seg_ids=None, weight_threshold=5,
                     remove_orphan_nodes=True, add_node_ids=False):
        nxg = self.links2nx(weight_threshold=weight_threshold)
        if seg_ids is not None:
            nxg = nxg.subgraph(seg_ids)
            nxg = nx.DiGraph(nxg)  # copy, otherwise graph frozen
        if remove_orphan_nodes:
            nxg.remove_nodes_from(list(nx.isolates(nxg)))
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 22})
        pos = nx.layout.spring_layout(nxg, k=5, weight='weight')
        node_sizes = 100
        nodes = nx.draw_networkx_nodes(nxg, pos, node_size=node_sizes,
                                       node_color='blue',
                                       label='neuron segment')
        edges = nx.draw_networkx_edges(nxg, pos, node_size=node_sizes,
                                       arrowstyle='->',
                                       arrowsize=10, edge_color='black',
                                       edge_cmap=plt.cm.Blues, width=2,
                                       connectionstyle='arc3,rad=0.1',
                                       label='edge weight')
        labels = nx.get_edge_attributes(nxg, 'weight')
        nx.draw_networkx_edge_labels(nxg, pos, edge_labels=labels)
        if add_node_ids:
            res = nx.draw_networkx_labels(sub_g, pos, font_size=20)

        ax = plt.gca()
        ax.set_axis_off()
        plt.legend()
        plt.show()

    def plot_up_downstream_subcircuit(self, seg_id, weight_threshold=5,
                                      topk=5, add_node_ids=False):

        nxg = self.links2nx(weight_threshold=weight_threshold)
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 22})
        downstream_nodes = list(nxg.predecessors(seg_id))
        downstream_nodes = downstream_nodes[:min(len(downstream_nodes), topk)]
        upstream_nodes = list(nxg.neighbors(seg_id))
        upstream_nodes = upstream_nodes[:min(len(upstream_nodes), topk)]
        all_nodes = [seg_id] + downstream_nodes + upstream_nodes

        sub_g = nxg.subgraph(all_nodes)

        pos = nx.spring_layout(sub_g, k=5, weight='weight')
        nx.draw_networkx_nodes(sub_g, pos, nodelist=downstream_nodes,
                               node_color='#F2A431', label='downstream seg ids',
                               alpha=0.5)
        nx.draw_networkx_nodes(sub_g, pos, nodelist=upstream_nodes,
                               node_color="#55B849", label='upstream seg ids',
                               alpha=0.5)
        nx.draw_networkx_nodes(sub_g, pos, nodelist=[seg_id],
                               node_color='#834D9D')
        edges = nx.draw_networkx_edges(sub_g, pos, arrowstyle='->',
                                       arrowsize=10, edge_color='black',
                                       edge_cmap=plt.cm.Blues, width=2,
                                       connectionstyle='arc3,rad=0.1',
                                       label='edge weight')

        labels = nx.get_edge_attributes(sub_g, 'weight')
        nx.draw_networkx_edge_labels(sub_g, pos, edge_labels=labels)
        plt.title(
            'Upstream and downstram neuron partner of segmentation id {}'.format(
                seg_id))
        if add_node_ids:
            res = nx.draw_networkx_labels(sub_g, pos, font_size=20)
        plt.legend(bbox_to_anchor=(0.01, 0.01))
        plt.show()
