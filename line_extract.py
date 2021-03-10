import cv2
import numpy as np
from tqdm import tqdm
import sys
THRESHOLD = 10000
THRESHOLD_y = 20
THRESHOLD_x_diff = 5
NEIBOR_THREHOLD = 10
class Coordinate:
    """
    点数据结构
    """
    def __init__(self, x, y, id):
        self.id = id
        self.x = x
        self.y = y


    def __repr__(self):
        res ='coord: (%s, %s)' % (self.x, self.y)
        return res

    def set_visited(self, visited):
        self.visited = visited

    def coord2list(self):
        return [self.x, self.y]



class Node(Coordinate):
    def __init__(self,x ,y, id):
        super(Node, self).__init__(x,y,id)
        self.visited = False
        self.neighbors = {} # key: node.id, value: edge.weight

    def add_neighbor(self, coord):
        if coord.id in self.neighbors:
            print("INFO: %s has been connected to %s" % (coord, self))
        self.neighbors[coord.id] = self.distance_from_node(coord.x, coord.y)

    def get_clostest_node(self):
        return self.neighbors[0] #return the closest node id

    def get_neighbors(self):
        return self.neighbors

    def sort_neighbors_by_weights(self):
        self.neighbors = sorted(self.neighbors.items(), key=lambda t:t[1], reverse=False)

    def distance_from_node(self, x, y):
        """
        计算该节点到另一节点的距离
        :param x:
        :param y:
        :return:
        """
        result = np.abs(self.y-y) + np.abs(self.x-x) # simple try!
        if self.y == y:
            result += THRESHOLD
        # tmp = np.power(self.x - x, 2) + np.power(self.y - y, 2)
        return result

class Edge():
    """
    边数据结构
    """
    def __init__(self, coord1: Node, coord2: Node, id=0):
        self.start_node = coord1
        self.end_node = coord2
        self.weight = self.start_node.distance_from_node(self.end_node.x, self.end_node.y)
        self.id = id


    def __repr__(self):
        return 'edge: (%s -> %s)' % (self.start_node, self.end_node)


class Graph():
    def __init__(self):
        self.edges = []
        self.nodes = []
        self.id2node = {}
        self.id2edge = {}
        self.min_y = None
        self.max_y = None

    def get_nodes(self):
        return self.nodes

    def get_egdes(self):
        return self.edges

    def add_edge(self, edge):
        """
        按照边的id添加
        :param edge:
        :return:
        """
        self.edges.append(edge)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def find_neighbors(self):
        """
        TODO: 预计算每个节点的邻居
        :return:
        """
        for node in self.nodes:
            pass

    def __repr__(self):
        return """
            Graph has %s nodes and %s edges
        """% (len(self.nodes), len(self.edges))

class Data:
    """
    结果数据结构
    """
    def __init__(self):
        self.coordinates = [] # 最后保留的线的坐标
        self.xs = []
        self.ys = []
        self.id2coord = {}
        self.start_id = 0
        self.length = 0

    def add_coord(self, x: int, y: int):
        """
        添加坐标
        :param x:
        :param y:
        :return:
        """

        self.coordinates.append(Node(x, y, id=(self.start_id)))
        self.start_id += 1
        self.id2coord[self.start_id] = [x, y]

    def sort_coordinate(self):
        print("**对坐标按照y维度排序**")
        self.coordinates = sorted(self.coordinates, key=lambda t: t.y)

    def get_length(self):
        pass
    def coord2array(self):
        pass



class PathSearch():
    def __init__(self, data):
        self.graph = None # 图
        self.data = data # 原坐标数据
        self.result = None # 结果
        self.visited = None # 节点是否已访问
        self.start_points = []


    def get_init_points(self):
        """
        寻找搜索起点： 如将y<THRESHOLD_y(default 20)的坐标中x相差超过5的节点加入起点列表
        TODO: 待优化，目前会将
        :return:
        """
        import math
        print("**GET INIT POINTS**")

        last_node = None
        for node in self.graph.nodes[:THRESHOLD_y]:
            if last_node is None:
                last_node = node
                self.start_points.append(last_node)
            else:
                # if last_node.y == node.y:
                #     last_node = node
                #     continue
                if math.fabs(last_node.x - node.x) > THRESHOLD_x_diff:
                    self.start_points.append(node)
            last_node = node

        return self.start_points



    def run(self):
        """
        TODO: 主搜索流程
         流程：从最左边的第一个点开始，向右进行搜索，贪心地连接到离当前点最近的点
        :return:
        """
        # 寻找搜索起点
        self.get_init_points()

        # 开始搜索
        self.search()

        # 整理搜索结果
        pass

    def build_graph(self) -> Graph:
        """
         添加节点之后计算两两之间的距离构建图
        :return:
        """
        print("***构建图****")
        self.graph = Graph()
        for i, coord in enumerate(self.data.coordinates):
            self.graph.add_node(coord)
            self.graph.id2node[coord.id] = coord

        edge_id = 0
        # 计算节点间距离：构建边
        for node in tqdm(self.graph.get_nodes()):
            for node2 in self.graph.get_nodes():
                if node.id != node2.id and (node2.id not in node.neighbors):
                    edge = Edge(node, node2, id=edge_id)
                    self.graph.add_edge(edge)
                    self.graph.id2edge[edge_id] = edge
                    node.add_neighbor(node2) # 内存占用极大，代优化
                    edge_id += 1
            node.sort_neighbors_by_weights()
        print(self.graph)
        return self.graph

    def search(self):
        """
        给定起始点，进行搜索
        :return:
        """
        print("**SEARCH: the size of start_points: %s**" % len(self.start_points))
        result_paths = {}

        for start_point in tqdm(self.start_points): # 对于每个待选起点都执行一次搜索
            path = self.search_from_one_point(start_point)
            result_paths[path] = path.length

        # 返回长度最长的一条路径
        pass

    def search_from_one_point(self, start_point: Node)-> Data:
        """
        从单个起始点开始搜索
        1. 找到离当前点current_p最近的点clostest_p，加入路径中，将当前点设置为clostest_p
        2. 搜索结束的条件：找到图中位于最右边的节点或者下一个最近的节点超过某个阈值（突变）为止
        :param start_point:
        :return:
        """

        current_node = start_point
        clostest_node = None
        path = Data()
        path.add_coord(current_node.x, current_node.y)
        while not (current_node.y >= self.graph.nodes[-1].y):
            clostest_node = self.graph.id2node[current_node.get_clostest_node()[0]]
            distance = current_node.get_clostest_node()[1]
            if distance > 100: # 判定为"突变"的阈值
                break
            else:
                path.add_coord(clostest_node.x, clostest_node.y)
                path.length += distance
            current_node = clostest_node
        return path





    def distance(self, coord1: Node, coord2: Node) -> float:
        """
        计算两个坐标的距离/成本函数
        :param coord1:
        :param coord2:
        :return:
        """
        return coord1.distance_from_node(coord2.x, coord2.y)

    def get_result(self) -> Data:
        return Data()



class LineExtract():
    def __init__(self):
        self.images = []
        self.datas = []
        self.data = None


    def find_shortest_path(self) -> Data:
        """
        TODO 寻找最短路径
        :return:
        """
        print("****寻找最短路径****")
        helper = PathSearch(self.data)

        # 构建图
        helper.build_graph()

        # 执行最短路径算法
        helper.run()

        # 返回结果(处理成Data结构，方便最后通过restore_from_corrdinate方法出图)
        return helper.get_result()


    def extract_corrdinate(self, image) -> Data:
        """
        从图片中提取坐标数据结构
        :param image:
        :return:
        """
        print("****数据简化：从标签图片提取坐标*****")
        self.data = Data()
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (384,288))
        print(img.shape)
        img = np.array((img))
        non_zero_pixels = np.where(img > 0)
        self.data.xs = non_zero_pixels[0]
        self.data.ys = non_zero_pixels[1]
        coords =  list(zip(non_zero_pixels[0], non_zero_pixels[1]))
        for coord in coords:
            self.data.add_coord(coord[0], coord[1])

        self.data.sort_coordinate()
        print(len(self.data.coordinates))
        print(self.data.coordinates)

        print("start_id:", self.data.start_id)

        return self.data


    def restore_from_corrdinate(self, data, save_path = "test.png"):
        print("****从坐标恢复成图片****")
        img = np.zeros((288,384))
        # img[255,23] = 255
        img[data.xs, data.ys] = 255
        # img[[255,255, 123], [23,50,53]] = 255
        cv2.imwrite(save_path, img)





if __name__ == '__main__':
    # test_image_path = "/Users/zhuge/CSUClass/GIS/外包山脊线检测/高层图数据/ridge_fine/train/label/ex_8.png"
    test_image_path = "/Users/zhuge/CSUClass/GIS/外包山脊线检测/高层图数据/ridge_fine/train/label/ex_28.png"
    import os
    # os.path.basename(test_image_path)

    instance = LineExtract()

    # 数据简化
    res = instance.extract_corrdinate(test_image_path)

    # 主算法
    shorest_path = instance.find_shortest_path()

    # 恢复原图
    restore_res = instance.restore_from_corrdinate(res, os.path.basename(test_image_path))

