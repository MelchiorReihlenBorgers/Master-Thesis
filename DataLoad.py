import os
import cv2
import pandas as pd
import numpy as np


class DataLoad(object):

    """
    Class to load images, depths and metadata
    """

    def __init__(self, path):
        """
        Parameters:
        -----------
        path:   string
                - Path to the project


        Attributes:
        -----------
        path
        """

        self.path = path

    def extract_paths(self):
        """
        Create list of paths if the structure is more complicated.
        Here the directory structure is as follows:

        Thesis
        |
        |______ images_and_dephts_large
                |
                |_____  Cow_Image_T
                        |
                        |_____  1
                                |
                                |___ image1_1.jpg
                                |___ image1_2.jpg
                                |___ image1_3.jpg
                                |___ depth1_1.csv
                                |___ depth1_2.csv
                                |___ depth1_3.csv
                                |___ metafile1.txt
                        |_____  2
                                |
                                |___ image2_1.jpg
                                |___ image2_2.jpg
                                |___ image2_3.jpg
                                |___ depth2_1.csv
                                |___ depth2_2.csv
                                |___ depth2_3.csv
                                |___ metafile2.txt
                        |____ ...
                        |___  ...
                        |
                        |_____  72
                                |
                                |___ image72_1.jpg
                                |___ image72_2.jpg
                                |___ image72_3.jpg
                                |___ depth72_1.csv
                                |___ depth72_2.csv
                                |___ depth72_3.csv
                                |___ metafile72.txt

        Note that the same image is repeated three times.
        The current implementation takes the first element of the list of images loaded.

        Description:
        ------------
        - Create the list of all the subfolders of Cow_Images_T
        - Create list of paths to all files
        - Save every third file path that ends with .csv as depth_path files
        - Save every third file path that ends with .jpg as image_path
        - Save every third file path that ends with .txt as metadata

        Returns:
        --------
        image_paths:    list
                        - list of paths to images

        depth_paths:    list
                        - list of paths to depths

        txt_paths:      list
                        - list of paths to metadata
        """

        path_data = os.path.join(self.path, "Data/images_and_depths_large/Cow_Images_T")

        # Create all the paths.
        paths = [i for i in os.listdir(path_data)]

        n = len(paths)

        # Extract all the file names from the pats.
        file_list = [os.listdir(os.path.join(path_data, paths[i])) for i in range(n)]

        m = len(file_list)

        # Combine all the paths and filenames.
        file_paths = [os.path.join(path_data, paths[i], file_list[i][j])
                      for i in range(m)
                      for j in range(len(file_list[i]))
                      ]

        k = len(file_paths)

        # Select all the depths (.csv files) and keep only every third.
        depth_paths = [file_paths[i] for i in range(k) if file_paths[i].endswith(".csv")]
        depth_paths = depth_paths[::3]

        # Select all the images (.jpg files) and keep only every third.
        image_paths = [file_paths[i] for i in range(k) if file_paths[i].endswith(".jpg")]
        image_paths = image_paths[::3]

        # Select all the metafiles
        txt_paths = [file_paths[i] for i in range(k) if file_paths[i].endswith(".txt")]

        return image_paths, depth_paths, txt_paths

    def read_txt(self, path):
        with open(path) as MetaDataFile:
            metaData = MetaDataFile.read().split(',')
            radian = float(metaData[2])

        return radian


    def load_data(self, N):
        """
        Load the data itself.

        Description:
        ------------
        Using complex_path method, load the paths to the images, depths and metadata files (latter read in using
        the read_txt method)

        Control the number of images loaded using N


        Parameters:
        -----------
        N:  int
            - Number of images to load

        Returns:
        --------
        images: list
                - list of images

        depths: list
                - list of depths

        radian: list
                - list of radians extracted from the metadata files
        """

        image_paths, depth_paths, txt_paths = self.extract_paths()

        depths = [ np.asarray(pd.read_csv(depth_paths[ i ])) for i in range(N)]

        images = [ cv2.cvtColor(cv2.imread(image_paths[ i ]), cv2.COLOR_BGR2RGB) for i in range(N)]

        radian = [ self.read_txt(path=txt_paths[ i ]) for i in range(N)]

        print("{} Images, {} Depths and {} Radians measures were loaded".format(len(images), len(depths), len(radian)))
        return images, depths, radian



if __name__ == "__main__":
    data = DataLoad(path = os.getcwd())

    n_total = len(data.extract_paths()[0]) # If you want to load all images.

    images, depths, radians = data.load_data(N = 10)

    paths = data.extract_paths()
    print(paths[0][:2])