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
    _GeneGraph is responsible for tracing an output gene to its input genes.
    """
    def __init__(self, output_gene_index, genome):
        output_gene = genome.genes[output_gene_index]
        self.head = Node(description='OUT ' + str(output_gene_index))

        # begin recursive trace on self.head
        self.add_to_graph(parent_node=self.head,
                          parent_gene=output_gene,
                          genome=genome)

    def add_to_graph(self, parent_node, parent_gene, genome):
        """
        Recursively adds nodes to a drawable graph
        :param parent_node:
        :param parent_gene:
        :param genome:
        :return:
        """
        # a list containing the indexes of gene.x and gene.y
        from_genes = [parent_gene.getx(), parent_gene.gety()]

        # iterate over from_list and recurse on a gene if it is not already part of the graph
        for i in range(len(from_genes)):
            gene_index = from_genes[i]
            gene = genome.genes[gene_index]

            if not gene.active_in_functional_graph:
                gene.active_in_functional_graph = True

                # the label text shown on the graph
                description = _function_set.function_descriptions[gene.get_f()]
                node = Node(description=description)

                self.add_to_graph(parent_node=node,
                                  parent_gene=gene,
                                  genome=genome)

                parent_node.add_from_node(node=node, idx=i)

    def print(self):
        """
        Prints a single direction given the head node. Used mostly for verification
        :return: None
        """
        ptr = self.head
        while ptr.from_nodes[0] is not None:
            print(ptr.description)
            ptr = ptr.from_nodes[0]


class FunctionalGraph:
    """
    FunctionalGraph is responsible for calling _GeneGraph to create multiple graphs
    then FunctionalGraph draws it
    """
    WIDTH = 1500  # window width
    HEIGHT = 800  # window height
    LABEL_MARGIN = 7  # the margin between the label (a node) and the arrow tip
    CHILD_NODE_X_OFFSET = 1.5  # how far left/right a child node is placed from parent. AKA index incrementation
    CHILD_NODE_Y_OFFSET = 0.7  # how far downward a child node is placed from parent. AKA index incrementation
    INDEX_TO_COORDINATE = 50  # multiply an index by this to get its drawing location
    HEAD_NODE_X_OFFSET = WIDTH / 1.5  # where the head node x coordinate is set. graph location is determined by this

    def __init__(self, genome):
        # constants:
        # gene_graphs contains a graph for each output gene
        self.gene_graphs = []

        # reset all gene activated flags for each output gene
        for i in genome.outputs:
            for gene in genome.genes:
                gene.active_in_functional_graph = False

            gene_graph = _GeneGraph(output_gene_index=i, genome=genome)
            self.gene_graphs.append(gene_graph)

    def draw(self, gene_graph_index):
        """
        Creates a window and draws the graph.
        :param gene_graph_index: determines which of the output genes we wish to draw
        :return: None
        """
        master = Tk()
        gene_graph = self.gene_graphs[gene_graph_index]
        title = 'CGP Functional Graph - ' + gene_graph.head.description
        master.title(title)
        window = Canvas(master, width=self.WIDTH, height=self.HEIGHT)
        window.pack()

        # begin recursion on gene_graph.head
        self.add_node_to_window(window=window,
                                gene_graph=gene_graph,
                                node=gene_graph.head,
                                x=0,
                                y=0,
                                used_point_list=[])
        window.mainloop()

    def get_drawing_x_point(self, in_x):
        """
        Determines the drawing location given some index x
        :param in_x:  an index x
        :return: an x coordinate on the window's canvas
        """
        return (in_x * self.INDEX_TO_COORDINATE) + self.HEAD_NODE_X_OFFSET

    def get_drawing_y_point(self, in_y):
        """
        Determines the drawing location given some index y
        :param in_y: an index y
        :return: an y coordinate on the window's canvas
        """
        return (in_y * self.INDEX_TO_COORDINATE) + (self.INDEX_TO_COORDINATE / 2)

    def add_node_to_window(self, window, gene_graph, node, x, y, from_x=0, from_y=0, used_point_list=None):
        """
        Recursively adds nodes to a drawable gene graph
        :param window: the target window
        :param gene_graph: the target graph we are adding a node to
        :param node: the node we are adding to the graph
        :param x: the x index. used to determine where this node belongs on the canvas
        :param y: the y index. used to determine where this node belongs on the canvas
        :param from_x: the drawing x coordinate of the previous node. this is used to draw the directional edge
        :param from_y: the drawing y coordinate of the previous node. this is used to draw the directional edge
        :param used_point_list: a list containing used (x, y) coordinates. Used to avoid overlapping labels
        :return: None
        """
        # if the (x, y) coordinate has already been used, increase y to avoid overlapping labels
        while (x, y) in used_point_list:
            y += self.CHILD_NODE_Y_OFFSET

        draw_x = self.get_drawing_x_point(x)
        draw_y = self.get_drawing_y_point(y)
        used_point_list += [(x, y)]

        window.create_text(draw_x, draw_y, text=node.description)

        # if the node is not the output node, then we can draw an arrow from its parent to it
        if node != gene_graph.head:
            window.create_line(from_x,
                               from_y + self.LABEL_MARGIN,
                               draw_x,
                               draw_y - self.LABEL_MARGIN,
                               arrow=FIRST)

        # iterate over node.left and node.right
        for node_index in range(len(node.from_nodes)):
            from_node = node.from_nodes[node_index]

            if from_node is not None:
                # next_point places a node downward and left or right of parent node
                # next_x_point uses the node_index to determine left or right
                next_x_point = x - self.CHILD_NODE_X_OFFSET + (2 * node_index * self.CHILD_NODE_X_OFFSET)
                next_y_point = y + self.CHILD_NODE_Y_OFFSET

                self.add_node_to_window(window=window,
                                        gene_graph=gene_graph,
                                        node=from_node,
                                        x=next_x_point,
                                        y=next_y_point,
                                        from_x=draw_x,
                                        from_y=draw_y,
                                        used_point_list=used_point_list)
