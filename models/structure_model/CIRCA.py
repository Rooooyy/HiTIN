import math
import heapq
import numba as nb
import numpy as np
import copy
import random
from queue import Queue


def get_id():
    i = 0
    while True:
        yield i
        i += 1


def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]  # num of real nodes
    adj_table = {}  # 邻接表, 存放每个节点[i]的邻居
    VOL = 0  # VOL of Graph
    node_vol = []  # vol of each node
    for i in range(g_num_nodes):
        n_v = 0
        neighbour_set = set()
        for j in range(g_num_nodes):
            if adj_matrix[i, j] != 0:
                n_v += adj_matrix[i, j]
                VOL += adj_matrix[i, j]
                neighbour_set.add(j)
        adj_table[i] = neighbour_set
        node_vol.append(n_v)
    return g_num_nodes, VOL, node_vol, adj_table


@nb.jit(nopython=True)
def cut_volume(adj_matrix,p1,p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i],p2[j]]
            if c != 0:
                c12 += c
    return c12


def LayerFirst(node_dict, start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
    new_node = CodingTreeNode(ID=new_ID, partition=new_partition, children={id1, id2},
                              g=g, vol=v, child_h= child_h, child_cut = cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break


def child_tree_deepth(node_dict,nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth+=1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1, p_node):
    a = node1.child_cut
    v1 = node1.vol
    v2 = p_node.vol
    return a * math.log(v2 / v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2
    return ((v1 - g1) * math.log(v12 / v1, 2) + (v2 - g2) * math.log(v12 / v2, 2) - 2 * cut_v * math.log(g_vol / v12, 2)) / g_vol


class CodingTreeNode():
    def __init__(self, ID, partition, vol, g, children: set = None, parent=None, child_h=0, child_cut=0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h  # 不包括该节点的子树高度
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())


class CodingTree():

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(adj_matrix)  # total # of real nodes, Vol of Graph, 邻接表
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()


    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)  # 给叶节点依次赋予id
            n_v = self.node_vol[vertex]
            leaf_node = CodingTreeNode(ID=ID, partition=[vertex], g = n_v, vol=n_v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)


    def build_sub_leaves(self,node_list,p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                       * math.log2(self.tree_node[vertex].vol / p_vol)
            sub_n = set()
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex,vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            sub_leaf = CodingTreeNode(ID=vertex, partition=[vertex], g=vol, vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict,ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            if node.vol * g_vol != 0:
                ori_en += -(node.g / g_vol) * math.log2(node.vol / g_vol)
            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n

            new_node = CodingTreeNode(ID=node_id, partition=node.partition, vol=node.vol, g = node.g, children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en


    def entropy(self,node_dict = None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id,node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                if node_p_vol * node_vol != 0:
                    ent += - (node_g / self.VOL) * math.log2(node_vol / node_p_vol)
        return ent

    def __build_k_tree(self, g_vol, nodes_dict:dict, k = None):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:  # 避免重复
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        # 将叶子节点尝试两两组合起来
        # add by zc
        if len(nodes_dict.keys()) <= 2:
            leave_num = len(nodes_dict.keys())
            for i in range(0, k):
                new_id = leave_num + i
                if leave_num == 2:
                    if i == 0:
                        new_node = CodingTreeNode(ID=new_id, partition=list(nodes_ids), children={new_id - 1, 0},
                                                  vol=0, g=0, child_h=i + 1)
                        nodes_dict[0].parent = new_id
                    else:
                        new_node = CodingTreeNode(ID=new_id, partition=list(nodes_ids), children={new_id - 1},
                                                  vol=0, g=0, child_h=i+1)
                else:
                    new_node = CodingTreeNode(ID=new_id, partition=list(nodes_ids), children={new_id - 1},
                                              vol=0, g=0, child_h=i + 1)
                nodes_dict[new_id - 1].parent = new_id
                nodes_dict[new_id] = new_node
            self.root_id = new_id
            root = new_id
            return root

        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            # compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[id1],nodes_dict[new_id]), id1, new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[id2],nodes_dict[new_id]), id2, new_id])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix,np.array(n1.partition), np.array(n2.partition))

                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id

        if unmerged_count > 1:
            # combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            max_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes])
            new_child_h = max_child_h + 1

            new_id = next(self.id_g)
            new_node = CodingTreeNode(ID=new_id, partition=list(nodes_ids), children=unmerged_nodes,
                                      vol=g_vol, g = 0, child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[i], nodes_dict[new_id]), i, new_id])
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        return root


    def check_balance(self,node_dict,root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict,c)

    def single_up(self,node_dict,node_id):
        new_id = next(self.id_g)
        p_id = node_dict[node_id].parent
        grow_node = CodingTreeNode(ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
                                   children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        for i in self.adj_table[node_id].copy():
            self.adj_table[i].add(new_id)

    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0, None, None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol, nodes_dict=subgraph_node_dict, k=2)
        self.check_balance(subgraph_node_dict, new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self,sub_node_dict,sub_root_id,node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict,sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict,ori_ent = self.build_sub_leaves(sub_nodes,candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol,nodes_dict=subgraph_node_dict,k = 2)
                self.check_balance(subgraph_node_dict,sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict,sub_root,node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta,id_mapping,h1_new_child_tree

    def leaf_up_update(self,id_mapping,leaf_up_dict):
        for node_id,h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node, i)

            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1

    def root_down_update(self, new_id , root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, mode='v2', k=2):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=2)
            self.check_balance(self.tree_node, self.root_id)
            flag = 0
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id, root_down_dict = self.root_down_delta()

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                elif flag == 2:
                    root_down_delta, new_id, root_down_dict = self.root_down_delta()
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    # print('root down')
                    # root down update and recompute root down delta
                    flag = 2
                    self.root_down_update(new_id, root_down_dict)

                else:
                    # leaf up update
                    # print('leave up')
                    flag = 1
                    self.leaf_up_update(id_mapping, leaf_up_dict)

                    # update root down leave nodes' children
                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items():
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[root_down_id].children
        # count = 0
        # for _ in LayerFirst(self.tree_node, self.root_id):
        #     count += 1
        # assert len(self.tree_node) == count


def get_child_h(y, k):
    root = y.tree_node[y.root_id]
    queue = Queue()
    next = []
    queue.put(root)
    i = 0
    while i < k:
        if queue.empty():
            for n in next:
                queue.put(n)
            i += 1
            next = []
        node = queue.get()
        node.child_h = k - i
        if node.children is None:
            continue
        for ch in node.children:
            next.append(y.tree_node[ch])

    return y


def map_id(y, k):
    map = {}
    layer = {}
    for i in range(0, k + 1):
        layer[str(i)] = []
    for i, n in y.tree_node.items():
        layer[str(n.child_h)].append(n.ID)
    index = 0
    for k in layer.keys():
        for id in layer[k]:
            map[id] = index
            index += 1
    for i, n in y.tree_node.items():
        n.ID = map[n.ID]
        n.partition = [map[j] for j in n.partition if j in list(y.tree_node.keys())]
        if n.parent is not None:
            n.parent = map[n.parent]
        if n.children is not None:
            n.children = {map[j] for j in n.children}
    new_treenode_list = [0] * len(list(y.tree_node.keys()))
    for i, n in y.tree_node.items():
        new_treenode_list[n.ID] = n
    new_y = {}
    for i, n in enumerate(new_treenode_list):
        new_y[i] = n
    y.tree_node = new_y
    y.root_id = len(list(y.tree_node.keys())) - 1
    # new_y = sorted(sorted(new_y.keys()))
    return y


def update_node(tree):
    # 去掉因为压缩而消失的点，把idx排序
    ids = [v.ID for k, v in tree.items()]
    ids.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = ids.index(n.ID)
        if n.parent is not None:
            n.parent = ids.index(n.parent)
        if n.children is not None:
            n.children = [ids.index(c) for c in n.children]
        new_tree[n.ID] = n
    return new_tree


def build_random_tree(y, k):
    '''
    random tree
    :param y: Coding Tree
    :param k: the height of Coding Tree
    :return:
    '''
    new_id = len(y.leaves)
    new_tree_node = {}
    for nid, n in y.tree_node.items():
        if nid in y.leaves:
            new_tree_node[nid] = n
    for h in range(1, k):
        nodes_tobe_merged = [n for nid, n in new_tree_node.items() if n.child_h == (h-1)]
        while len(nodes_tobe_merged) > 0:
            if len(nodes_tobe_merged) == 1:
                new_node = CodingTreeNode(ID=new_id, partition=nodes_tobe_merged[0].partition, children={nodes_tobe_merged[0].ID},
                                             vol=0, g=0, child_h=h)
                nodes_tobe_merged.remove(nodes_tobe_merged[0])
            else:
                merge_nodes = random.sample(nodes_tobe_merged, 2)
                new_partition = []
                merge_ids = []
                for n in merge_nodes:
                    nodes_tobe_merged.remove(n)
                    new_partition.extend(n.partition)
                    merge_ids.append(n.ID)
                new_node = CodingTreeNode(ID=new_id, partition=new_partition, children={merge_ids[0], merge_ids[1]}, vol=0, g=0, child_h=h)
            new_tree_node[new_id] = new_node
            new_id += 1
    last_layer_nodes = [n for nid, n in new_tree_node.items() if n.child_h == (k-1)]
    new_partition = []
    merge_ids = []
    for n in last_layer_nodes:
        new_partition.extend(n.partition)
        merge_ids.append(n.ID)
    new_node = CodingTreeNode(ID=new_id, partition=new_partition, children=set(merge_ids), vol=0, g=0,
                                 child_h=k)
    new_tree_node[new_id] = new_node
    y.tree_node = new_tree_node
    y.root_id = new_id
    return y


if __name__ == '__main__':

    # adj = np.loadtxt("b.txt", delimiter=",")
    # adj = []
    # with open("a.txt", 'r') as f:
    #     for line in f:
    #         data = line.strip().split(',')
    #         adj.append(list(map(int,data)))
    adj = [[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
          [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
          [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1],
          [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
          [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 3, 0, 4, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 2, 0, 0, 3, 1, 0, 0, 0, 1],
          [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
          [1, 0, 1, 0, 0, 0, 0, 0, 1, 4, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]]
    # adj = [[0, 1, 1, 0, 0],
    #        [1, 0, 1, 1, 0],
    #        [1, 1, 0, 1, 1],
    #        [0, 1, 1, 0, 1],
    #        [0, 0, 1, 1, 0]]
    # adj = [[0]]
    # adj = [[0, 4],
    #        [4, 0]]
    height = 3
    adj = np.array(adj)
    y = CodingTree(adj)
    if adj.shape[0] <= 2:
        m = 'v1'
    else:
        m = 'v2'
    # print(m)
    y.build_coding_tree(mode=m, k=height)
    y = get_child_h(y, k=height)
    y = map_id(y, k=height)

    for i, j in y.tree_node.items():
        print(j)
    print(y.entropy())