from cgp.functionset import FunctionSet
from tkinter import *

# _function_set is used to grab the function descriptions
_function_set = FunctionSet()


class Node:
    """
    Simple class to convert genes into a graphable format.
    Node.description is either a function name or gene output index.
    Node.from_nodes is a list containing the nodes which represent gene.x and gene.y.
    """
    def __init__(self, description=None):
        self.description = description
        self.from_nodes = [None, None]

    def add_from_node(self, node, idx):
        self.from_nodes[idx] = node


class _GeneGraph:
    """
    _GeneGraph is responsible for tracing an output gene to the input genes.
    """
    def __init__(self, output_gene_index, genome):
        output_gene = genome.genes[output_gene_index]
        self.head = Node(description='OUT ' + str(output_gene_index))
        self.add_to_graph(parent_node=self.head, parent_gene=output_gene, genome=genome)

    def add_to_graph(self, parent_node, parent_gene, genome):
        from_genes = [parent_gene.getx(), parent_gene.gety()]

        for i in range(len(from_genes)):
            gene_index = from_genes[i]
            gene = genome.genes[gene_index]

            if not gene.active_in_functional_graph:
                gene.active_in_functional_graph = True
                description = _function_set.function_descriptions[gene.get_f()]
                node = Node(description=description)
                self.add_to_graph(parent_node=node, parent_gene=gene, genome=genome)
                parent_node.add_from_node(node=node, idx=i)

    def print(self):
        ptr = self.head
        while ptr.from_nodes[0] is not None:
            print(ptr.description)
            ptr = ptr.from_nodes[0]


class FunctionalGraph:
    def __init__(self, genome):
        self.gene_graphs = []
        self.WIDTH, self.HEIGHT = 1500, 1000
        self.window = None

        for i in genome.outputs:
            for gene in genome.genes:
                gene.active_in_functional_graph = False

            gene_graph = _GeneGraph(output_gene_index=i, genome=genome)
            self.gene_graphs.append(gene_graph)

    def save_to_image(self):
        if self.window is None:
            print("FunctionalGraph.window is None")
            return

    def draw(self, gene_graph_index):
        master = Tk()
        gene_graph = self.gene_graphs[gene_graph_index]
        title = 'CGP Functional Graph - ' + gene_graph.head.description
        master.title(title)
        self.window = Canvas(master, width=self.WIDTH, height=self.HEIGHT)
        self.window.pack()
        self.add_node_to_window(gene_graph=gene_graph,
                                node=gene_graph.head,
                                x=0,
                                y=0,
                                used_point_list=[])
        self.window.mainloop()

    @staticmethod
    def get_drawing_x_point(inx, window_width):
        return (inx * 100) + (window_width / 2)

    @staticmethod
    def get_drawing_y_point(iny):
        return (iny * 50) + 25

    def add_node_to_window(self, gene_graph, node, x, y, from_x=0, from_y=0, used_point_list=None):

        # if the (x, y) coordinate has already been used, increase y to avoid overlapping labels
        while (x, y) in used_point_list:
            y += 1

        draw_x = self.get_drawing_x_point(x, self.WIDTH)
        draw_y = self.get_drawing_y_point(y)
        used_point_list += [(x, y)]

        self.window.create_text(draw_x, draw_y, text=node.description)

        # if the node is not the output node, then we can draw an arrow from its parent to it
        if node != gene_graph.head:
            self.window.create_line(from_x, from_y + 7, draw_x, draw_y - 7, arrow=FIRST)

        # iterate over node.left and node.right
        for node_index in range(len(node.from_nodes)):
            from_node = node.from_nodes[node_index]

            if from_node is not None:
                # next_point places a node downward and left or right of parent node
                # next_x_point uses the node_index to determine left or right: '+ (1.0 * node_index)'
                next_x_point = x - 0.5 + (1.0 * node_index)
                next_y_point = y + 0.8
                self.add_node_to_window(gene_graph=gene_graph,
                                        node=from_node,
                                        x=next_x_point,
                                        y=next_y_point,
                                        from_x=draw_x,
                                        from_y=draw_y,
                                        used_point_list=used_point_list)
