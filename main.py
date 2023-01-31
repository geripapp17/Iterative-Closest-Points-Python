import sys
import time
import numpy as np
import open3d

from typing import List
from nptyping import NDArray


class Runtime():
    def __init__(self, name:str) -> None:
        self.name = name

    def __enter__(self) -> None:
        self.begin = time.time()

    def __exit__(self, *args) -> None:
        print(f'Runtime of {self.name}: {time.time() - self.begin}ms')

def main(args):

    cloud1 = open3d.io.read_point_cloud(args[1])
    cloud2 = open3d.io.read_point_cloud(args[2])

    # open3d.visualization.draw_geometries([cloud1])

    with Runtime('ICP') as _:
        icp(cloud_src=cloud2, cloud_dst=cloud1)


def icp(cloud_src:open3d.geometry.Geometry.Type.PointCloud, cloud_dst:open3d.geometry.Geometry.Type.PointCloud, max_iter:int=1000):
    
    tree_dst = open3d.geometry.KDTreeFlann(cloud_dst)

    prev_error = 0
    for i in range(max_iter):

        mean_src, _ = open3d.geometry.compute_point_cloud_mean_and_covariance(cloud_src)
        mean_dst, _ = open3d.geometry.compute_point_cloud_mean_and_covariance(cloud_dst)

        points_src, points_dst = associate(cloud_src=cloud_src, cloud_dst=cloud_dst, tree_dst=tree_dst)

        points_src_shifted = points_src - mean_src
        points_dst_shifted = points_dst - mean_dst

        T = get_transformation(mean_src, mean_dst, points_src_shifted, points_dst_shifted)

        # error = 
        print()
        # if error >= prev_error:
        #     break

        # prev_error = error

def tr_icp(cloud_src, cloud_dst):
    pass

def associate(cloud_src:open3d.geometry.Geometry.Type.PointCloud, cloud_dst:open3d.geometry.Geometry.Type.PointCloud, tree_dst) -> List[NDArray]:

    array_src = np.asarray(cloud_src.points)
    array_dst = np.asarray(cloud_dst.points)
    array_dst_ordered = np.zeros(shape=array_src.shape, dtype=float)

    for i in range(array_src.shape[0]):
        _, idx, _ = tree_dst.search_knn_vector_3d(cloud_src.points[i], 1)
        array_dst_ordered[i,:] = array_dst[idx, :]

    return array_src, array_dst_ordered

def get_transformation(mean_src:NDArray, mean_dst:NDArray, points_src_shifted:NDArray, points_dst_shifted:NDArray) -> NDArray[float]:

    H = np.dot(points_dst_shifted, points_src_shifted.T)
    U, S, VT = open3d.core.svd(H)
    
    T = np.eye(N=4, dtype=float)
    R = np.dot(VT.t, U.t)
    T[:3,:3] = R
    T[:3,3] = mean_dst - np.dot(R, mean_src)

    return T

def to_homogeneous(cloud:NDArray) -> NDArray:
    
    dim = cloud.shape[1]
    ret_val = np.ones(shape=(cloud.shape[0], dim+1), dtype=float)
    ret_val[:,:-1] = cloud

    return ret_val

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        raise Exception('Bad arguments')

    main(sys.argv)