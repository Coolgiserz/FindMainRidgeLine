import unittest
import sys
import os
# sys.path.append(os.path.join(os.getcwd(), "..", "..", "utils"))
import line_extract
class TestGraph(unittest.TestCase):


    def test_add_edge(self):
        self.graph = line_extract.Graph()
        self.graph.add_node()

if __name__ == '__main__':

    unittest.main()