import unittest
import sys
import os
import pathlib
target_path = pathlib.Path(os.path.abspath(__file__)).parents
# sys.path.append(target_path)
# print (target_path)
# sys.path.append(os.path.abspath('..'))
# print(sys.path)
sys.path.append(os.path.join(os.getcwd(), "..", "..", "utils"))
from .. import line_extract
class TestCoord(unittest.TestCase):


    def test_distance_from_node(self):
        self.coord = line_extract.Node(10, 2)

        print("**TEST distance from node**")
        print(self.coord.distance_from_node(10, 1))
        print(self.coord.distance_from_node(11, 1))
        print(self.coord.distance_from_node(10, 4))
        print(self.coord.distance_from_node(120, 1))

if __name__ == '__main__':

    unittest.main()