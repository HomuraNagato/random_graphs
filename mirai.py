
''' Mirai functions for creating, modifying, and evaluating graphs
'''
import networkx as nx
import numpy as np
import pandas as pd
import itertools
import os
import math
import time
import argparse
import scipy.stats
from io import StringIO
from csv import writer
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set(rc={'figure.figsize':(12,12)})
sns.set_style("white")

import matplotlib.pyplot as plt
from data.curved_edges import *
from matplotlib.collections import LineCollection
from fa2 import ForceAtlas2

class Graph:

    def __init__(self, n, neighbor_prob):
        self.G = nx.empty_graph(n)
        self.G_init = self.G.copy() # updated at end of initialize and incorporate
        self.neighbor_prob = neighbor_prob
        self.G_neighbors = {}
        self.communities = {} # predicted community labels

    def get_graph(self):
        return self.G

    def initialized_graph(self):
        return self.G_init

    def get_communities(self):
        return self.communities
    
    def initialize_graph(self, n, p, randomized=False):

        internal_edges = list(itertools.combinations(list(self.G.nodes), 2))

        if randomized:
            p = np.random.uniform(p,1)

        #print("initialize graph variables", n, p, p, randomized)
        nodes = np.arange(0,n)
        for i in nodes:
            randomized_label = np.random.uniform(0,1)
            self.G.add_node(i, community=1, random_label=randomized_label, initial_label = randomized_label, contiguous=False)

        for possible_edge in internal_edges:
            if np.random.uniform(0,1) <= p:
                self.G.add_edge(*possible_edge)

        #self.groups.append(np.arange(0,n))
        self.identify_leader(nodes, 1)
        self.G_init = self.G.copy()

    def incorporate_graph(self, n, p1, p2, k, randomized=False, background=False):

        max_label = max(list(self.G.nodes))
        G1_nodes = list(self.G.nodes)
        G2_nodes = list(range(max_label+1,max_label+1+n))

        # randomize community intra probability using uniform distribution
        if randomized and not background:
            p1 = np.random.uniform(p1,1)

        #print("incorporate graph variables", n, p1, p2, k, randomized, background)
        # add new nodes to cosmos network
        nodes = np.arange(max_label+1,max_label+1+n)
        for i in nodes:
            #print("adding new node to G", i)
            randomized_label = np.random.uniform(0,1)
            self.G.add_node(i, community=k, random_label=randomized_label, initial_label = randomized_label, contiguous=False)

        # get distinct edges to calculate different probability of connectivity
        external_edges = list(itertools.combinations(G1_nodes, 2))
        internal_edges = list(itertools.combinations(G2_nodes, 2))
        all_connected_edges = list(itertools.combinations(G1_nodes + G2_nodes, 2))
        new_connected_edges = list(set(all_connected_edges) - set(external_edges) - set(internal_edges))

        for possible_edge in internal_edges:
            if np.random.uniform(0,1) <= p1:
                self.G.add_edge(*possible_edge)
                #print("adding edge between internal nodes", possible_edge)

        for possible_edge in new_connected_edges:
            if np.random.uniform(0,1) <= p2:
                self.G.add_edge(*possible_edge)
                #print("adding edge between external nodes", possible_edge)

        # ignore background from nodes of interest
        if background == False:
            self.identify_leader(nodes, k)
        
        self.G_init = self.G.copy()

    
    def prune_graph(self, nodes, p2):
        # get list of nodes with zero connectivity and try to reconnect to network
        degree_zero_nodes = []
        internal_edges = list(itertools.combinations(list(self.G.nodes), 2))
        
        for node in self.G.nodes:
            if self.G.degree(node) == 0:
                #print("node", node, "has zero degree")
                for possible_edge in internal_edges:
                    if np.random.uniform(0,1) <= p2:
                        self.G.add_edge(*possible_edge)
            if self.G.degree(node) == 0:
                return self.prune_graph(nodes, p2)
        return None
                
    
    def neighbors(self, current_node):
        if len(self.G_neighbors) == 0:
            for node in self.G.nodes:
                self.G_neighbors[node] = nx.single_source_shortest_path_length(self.G, node, cutoff=1)
        return self.G_neighbors[current_node]


    def identify_leader(self, nodes, comm):
        max_node_id = 0
        max_node_label = 0
        
        for node in nodes:
            
            if self.G.node[node]['random_label'] > max_node_label:
                #print("\tnode", node, G.node[node]['random_label'], "is currently the largest label in community", group)
                max_node_id = node
                max_node_label = self.G.node[node]['random_label']
        
        self.communities[max_node_id] = nodes

        
    def init_leaders(self):
        '''
        Identify the max label in each group
        '''

        for group in self.groups:
            max_node_id = 0 # potentially '0'
            max_node_label = 0
            for node_id in group:
                #node = str(node)
                if self.G.node[node_id]['random_label'] > max_node_label:
                    #print("\tnode", node, G.node[node]['random_label'], "is currently the largest label in community", group)
                    max_node_id = node_id
                    max_node_label = self.G.node[node_id]['random_label']
            self.communities[max_node_id] = group

    def manual_control(self):
        '''
        manually increase nodes that should be respective leaders to more easily see which is leading each neighborhood
        '''
        manual_value = np.arange(1.5,len(self.communities.keys())+.51)
        borderColors = dict(zip(self.communities.keys(), manual_value))
        for key, value in borderColors.items():
            self.G.node[key]['initial_label'] = value
            self.G.node[key]['random_label'] = value

    def label_choice(self, axis, nodelist=[]):

        if len(nodelist) == 0 or len(nodelist) == len(self.G):
            nodelist = list(self.G.nodes)

        labels = [ self.G.node[node][axis] for node in self.G.nodes if int(node) in nodelist ]

        return labels

    def get_labels(self):
        labels = {}
        for node in self.G.nodes:
            labels[node] = round(self.G.node[node]['random_label'], 2)

        return labels
    
    def convergence(self, i):

        truth = [ 0 for i in range(len(self.communities))]
        j = 0

        for community_leader, community_list in self.communities.items():
            expected_label = self.G.node[community_leader]['initial_label']
            current_labels = self.label_choice('random_label', community_list)

            if current_labels[0] == expected_label and len(set(current_labels)) == 1:
                truth[j] = 1

            j = j + 1

        if truth[0] == 1 and len(set(truth)) == 1:
            #print("convergence reached at step", i)
            return True
        else:
            return False


    def label_propagation(self, neighbor_prob=1):
        '''
        Iterating through each node one at a time, get the labels of its neighbors
        Update label with the mode of the neighbors (including self) if mode > 1, else
        update label with the max label of the neighbors
        '''
        labels = {}
        G_clone = self.G.copy()
        for node in self.G.nodes:

            nearest_neighbors = self.neighbors(node)

            neighborhood_labels = [ G_clone.node[neighbor]['random_label'] for neighbor in nearest_neighbors ]
            #print("neighborhood_labels", neighborhood_labels)

            # theory
            # consider how a subset of neighbors used for label choice affects label propagation
            neighborhood_labels = np.random.choice(neighborhood_labels, int(round(len(neighborhood_labels) * neighbor_prob, 1)), replace=False)
            #print("choosing", round(len(neighborhood_labels) * edge_prob), "from", len(neighborhood_labels))

            mode_label = scipy.stats.mode(neighborhood_labels)
            #print("mode label", mode_label)

            updated = False

            # take the most common label of neighbors
            # or take max label of neighbors
            if mode_label.count > 1 and float(mode_label.mode) != self.G.node[node]['random_label']:
                refresh_label = float(mode_label.mode)
                #print(float(mode_label.mode), " ", end='')
                updated = True
            elif mode_label.count == 1 and max(neighborhood_labels) != self.G.node[node]['random_label']:
                refresh_label = max(neighborhood_labels)
                #print("max", max(neighborhood_labels))
                updated = True
            else:
                #print("label propagation for node", node, "has no neighbor")
                #refresh_label = G.node[node]['random_label']
                updated = False

            if updated == True:
                labels[node] = round(refresh_label, 2)
                self.G.node[node]['random_label'] = refresh_label
                self.G.node[node]['contiguous'] = True
            else:
                labels[node] = round(G_clone.node[node]['random_label'], 2)
                self.G.node[node]['contiguous'] = False

        return labels

    def animate(self, i):
        
        #if not converged and i >= 1:
        if i >= 1:
            labels = self.label_propagation()
            converged = self.convergence(i)

        else:
            #self.init_leaders()
            self.manual_control()
            converged = False

        return converged


    def draw_histogram(self, trials, i):


        output = StringIO()
        csv_writer = writer(output)
        csv_writer.writerow(['node', 'community', 'community_label', 'initial_label', 'random_label', 'height'])

        for community_leader, community_nodes in self.communities.items():
            for node in community_nodes:
                csv_writer.writerow([node, self.G.node[node]['community'], community_leader, 
                                     self.G.node[node]['initial_label'], self.G.node[node]['random_label'], 1])

        output.seek(0) # need to get back to the start of the BytesIO
        df = pd.read_csv(output)

        df = df.groupby(['community', 'random_label']).sum().reset_index()
        df = df[df['height'] > 1]
        
        num_unique_labels = len(df['random_label'].unique())

        if (num_unique_labels <= 10 or num_unique_labels > 10) and df.shape[0] > 0:

            plt.figure()
            ax = sns.barplot(data=df, x="community", y="height", hue="random_label")
            annotation = "trials:" + str(trials) + "   step:" + str(i)
            ax.text(0.95, 0.95, annotation)

            path = 'images/trial_' + str(format(trials, '02d')) + '/histogram' + format(i, '03d') + '.png'
            fig = ax.get_figure()
            fig.savefig(path)


    
# for parsing known argument without consuming it in parse_args()
class UserNamespace(object):
    pass
user_namespace = UserNamespace()

    
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='''Analyze communities of Erdos-Renyi networks. In all variable arguments, possible values are INT=integer for a constant, ARR=array for a space separated list of values to test, or 
    RANGE=two values to create a numpy array range between the two.''')
    parser.add_argument('--nodes', nargs='+', type=int, default=[100, 100], help='INT, ARR: Number of nodes in each community')
    parser.parse_known_args(namespace=user_namespace)
    n = user_namespace.nodes
    c1, c2, n_total, k = 0.00044, 1.2, sum(n), len(n)
    p1 = c2 * (1800 * c1 * math.log(n_total) / n[0])**(1/4)
    p2 = 1/c2 * (n[0] * p1**2) / (8 * n_total)
    #print(theorem_2(n_total, n[0], p1, p2, c1), "p", p1, "p'", p2)
    parser.add_argument('--communities', nargs='+', type=int, default=[k], help='INT, ARR: Number of communities')
    parser.add_argument('--trials', nargs='?', type=int, const=10, default=10, help='INT: Number of trials' )
    parser.add_argument('--draw', nargs='?', const=True, default=False, help='FLAG: Whether to draw each step of algorithm; significantly increases run time, best used for with trials=1 for quick visualizations')
    parser.add_argument('--histogram', nargs='?', const=True, default=False, help='FLAG: Whether to draw a histogram of unique labels for nodes in expected communities')
    parser.add_argument('--steps', nargs='?', type=int, const=50, default=50, help='INT: Number of steps to take before stopping the algorithm')
    parser.add_argument('--intra_prob', nargs='+', type=float, default=[p1], help='INT, RANGE: Probability for intra-community connectivity with default to satisfy Theorem II')
    parser.add_argument('--inter_prob', nargs='+', type=float, default=[p2], help='INT, RANGE: Probability for intra-community connectivity with default to satisfy Theorem II')
    parser.add_argument('--background_num', nargs='+', type=int, default=[100], help='INT: Number of nodes to add as background noise')
    parser.add_argument('--background_prob', nargs='+', type=float, default=[p2], help='INT, RANGE: Probability for noise in background-connectivity. Defaults to p2\'')
    parser.add_argument('--neighbor_prob', nargs='?', type=float, default=1.0, help='FLOAT: Probability of number of neighbors to randomize when collecting neighbors')
    parser.add_argument('--randomized', nargs='?', const=True, default=False, help='FLAG: Whether p value is randomized uniform(p,1) or held constant when adding each node in each community')
    
    args = vars(parser.parse_args())
    defaults = vars(parser.parse_args([]))

    # make range if analysis variable range is provided
    if len(args['intra_prob']) == 2:
        args['intra_prob'] = np.arange(args['intra_prob'][0], args['intra_prob'][1], 0.01)
    if len(args['inter_prob']) == 2:
        args['inter_prob'] = np.arange(args['inter_prob'][0], args['inter_prob'][1], 0.01)
    if len(args['background_prob']) == 2:
        args['background_prob'] = np.arange(args['background_prob'][0], args['background_prob'][1], 0.01)
    # if draw or histogram, typically only care for one trial when trials parameter is left as default
    # update steps only if parameter not defined
    if args['draw'] == True and args['trials'] == defaults['trials']:
        args['trials'] = 1
        if args['steps'] == defaults['steps']:
            args['steps'] = 10

    if args['histogram'] == True and args['trials'] == defaults['trials']:
        args['trials'] = 1
        if args['steps'] == defaults['steps']:
            args['steps'] = 10
        
    print("parser arguments", args)

    return args

def load_network(path):
    G = nx.read_graphml(path, node_type=int)
    return G

def theorem_2(n_total, n, p1, p2, c):
    i_a, i_b = n * p1**2, 8 * n_total * p2
    ii_a, ii_b = n * p1**4, 1800 * c * math.log(n_total)
    if i_a > i_b and ii_a > ii_b:
        return True
    else:
        print("(i)", i_a, "!>", i_b, "\tand/or\t", "(ii)", ii_a, "!>", ii_b)
        return False


def draw_graphS2(G, i, communities, labels, pos, curves, converged):
    
    # Produce the curves and axis
    fig, ax = plt.subplots(figsize=(20,20))
    
    # nodes
    #leader_keys = [ v for key, v in communities.items() ]
    leader_keys = communities.keys()
    # max 9
    borderColorsList = ['orange', 'olive', 'teal', 'cyan', 'navy', 'grey', 'apricot', 'lavender', 'pink'][:len(communities)]
    contiguous_nodes = [ j for j in G.nodes if G.nodes[j]['contiguous'] == True ]
    non_contiguous_nodes = [ node for node in G.nodes if node not in contiguous_nodes ]

    nx.draw_networkx_labels(G, pos, labels=labels)
    
    nx.draw_networkx_nodes(G, pos, nodelist = leader_keys,
                           node_size=500, edgecolors = borderColorsList, linewidths = 10, node_color='red', alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, nodelist = contiguous_nodes,
                           node_size=100, edgecolors = 'yellow', linewidths = 10, node_color='cyan', alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, nodelist = non_contiguous_nodes,
                            node_size=100, node_color='cyan', alpha=0.8)
    
    # edges
    lc = LineCollection(curves, color='grey', alpha=0.05)    
    plt.gca().add_collection(lc)
    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    
    # text
    ax.text(0.95, 0.95, "n: "+str(n)+"   communities: "+str(len(communities)), ha="right", va="top", transform=plt.gca().transAxes)
    ax.text(0.95, 0.90, "step: "+str(i), ha="right", va="top", transform=plt.gca().transAxes)
    if converged:
        ax.text(0.95, 0.85, "converged!", ha="right", va="top", transform=plt.gca().transAxes)

    path = 'images/trial_' + str(format(trials, '02d')) + '/img' + format(i, '03d') + '.png'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def activate_algorithm(n_array, p1, p2, k, nq, q, count, trials, randomized=False, draw=False, histogram=False, neighbor_prob=1, stopping_time=10):
    
    # build the graph class
    OperativeGraph = Graph(n_array[0], neighbor_prob)
    OperativeGraph.initialize_graph(n_array[0], p1, randomized=randomized)

    for index, n in enumerate(n_array[1:]):
        if k > 1:
            OperativeGraph.incorporate_graph(n, p1, p2, index+2, randomized=randomized)

    # add background noise where its inter and intra connectivity is the same
    # remove unconnected nodes from network
    OperativeGraph.incorporate_graph(nq, q, q, 0, background=True)
    OperativeGraph.prune_graph(np.arange(sum(n_array), sum(n_array)+nq), q)
    
    if draw == True:
        forceatlas2 = ForceAtlas2(verbose=False)
        pos = forceatlas2.forceatlas2_networkx_layout(OperativeGraph.get_graph(), pos=None, iterations=50)
        curves = curved_edges(OperativeGraph.get_graph(), pos)
        
    try:
        path_length = nx.average_shortest_path_length(OperativeGraph.get_graph())
    except Exception as e:
        path_length = None
        #print("error", e)
    
    # test convergence
    i = 0
    converged = False
    
    # allow additional steps past convergence around step three to see if it remains converged
    while not converged or i <= stopping_time:
        print('\rtrial %02d / %02d \t step %02d / %02d' % (count+1, trials, i+1, stopping_time), end='\r')
        converged = OperativeGraph.animate(i)
        i = i + 1
        
        if draw == True:
            G = OperativeGraph.get_graph()
            communities = OperativeGraph.get_communities()
            labels = OperativeGraph.get_labels()
            draw_graphS2(G, i, communities, labels, pos, curves, converged)
            nx.write_graphml(OperativeGraph.initialized_graph(), 'images/trial_' + str(format(trials, '02d')) + '/S2_graph.graphml')

        if histogram == True:
            OperativeGraph.draw_histogram(trials, i)

        if i >= stopping_time:
            #print("convergence failed after", i, "steps.")
            break
    
    return path_length, converged

                                                                                                                                                                                                    
if __name__ == "__main__":

    args = parse_arguments()

    S2 = pd.DataFrame()
    start_time = time.time()

    n, trials, draw, histogram, stopping_time, neighbor_prob, randomized = args['nodes'], args['trials'], args['draw'], args['histogram'], args['steps'], args['neighbor_prob'], args['randomized']
    
    for k in args['communities']:
        for p1 in args['intra_prob']:
            for p2 in args['inter_prob']:
                for nq in args['background_num']:
                    for q in args['background_prob']:

                        # if alternating community size, default to make a constant community size
                        if len(n) != k:
                            n = [n[0]] * k

                        print('algorithm n: %s p: %s p\':%s nq: %s q: %s k: %s' % (n, round(p1,3), round(p2,3), nq, q, k))

                        path_lengths = np.empty(trials)
                        amount_converged = np.empty(trials)

                        for count in range(trials):

                            path_length, converged = activate_algorithm(n, p1, p2, k, nq, q, count, trials, randomized=randomized, draw=draw, histogram=histogram, neighbor_prob=neighbor_prob, stopping_time=stopping_time)
                            path_lengths[count] = path_length
                            amount_converged[count] = converged

                        averaged_path_lengths = np.average(path_lengths)
                        averaged_amount_converged = np.sum(amount_converged) / trials

                        #print("average path length:", averaged_path_lengths)
                        #print("average convergence:", averaged_amount_converged)

                        temp = pd.DataFrame({'n': n[0], 'p1': p1, 'p2': p2, 'nq': nq, 'q': q, 'k': k, 'path_length': averaged_path_lengths, 'converged': averaged_amount_converged}, index=[0])
                        S2 = pd.concat([S2, temp], axis=0, ignore_index = True)

    print("\n*** final results ***\n")
    print(S2.to_string())
    S2.to_csv('results/dataframe_results.csv', encoding='utf-8', index=False)
    
        
    end_time = time.time()
    run_time = end_time - start_time
    print("The program took", run_time, "seconds to complete")
