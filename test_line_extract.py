import unittest
import sys
import os
import line_extract
class TestPathSearch(unittest.TestCase):

    def test_run(self):
        print("===RUN===")
        test_image_path = "/Users/zhuge/CSUClass/GIS/外包山脊线检测/高层图数据/ridge_fine/train/label/newgp3k_269.png"#newgp3k_269,ex_28

        instance = line_extract.LineExtract()

        # 数据简化
        res = instance.extract_corrdinate(test_image_path)

        # 搜索
        self.pathsearch = line_extract.PathSearch(data=res)
        self.pathsearch.build_graph()
        self.pathsearch.run()
        pass


    # def test_build_graph(self):
    #     print("===test_build_graph===")
    #     test_image_path = "/Users/zhuge/CSUClass/GIS/外包山脊线检测/高层图数据/ridge_fine/train/label/ex_28.png"
    #
    #     instance = line_extract.LineExtract()
    #
    #     # 数据简化
    #     res = instance.extract_corrdinate(test_image_path)
    #
    #     # 搜索
    #     self.pathsearch = line_extract.PathSearch(data=res)
    #
    #     self.pathsearch.build_graph()
    #
    #     pass


if __name__ == '__main__':

    unittest.main()