from fileinput import filename
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
    cloud1.colors = open3d.utility.Vector3dVector(np.array([255, 0, 0], dtype=int) * np.ones((len(cloud1.points), 3), dtype=int))

    cloud2 = open3d.io.read_point_cloud(args[2])
    cloud2.colors = open3d.utility.Vector3dVector(np.array([0, 0, 255], dtype=int) * np.ones((len(cloud2.points), 3), dtype=int))

    with Runtime('ICP') as _:
        icp(cloud_src=cloud2, cloud_dst=cloud1, visualize=True)

    with Runtime('TR-ICP') as _:
        tr_icp(cloud_src=cloud2, cloud_dst=cloud1, visualize=True)

def icp(cloud_src:open3d.geometry.Geometry.Type.PointCloud, cloud_dst:open3d.geometry.Geometry.Type.PointCloud, max_iter:int=50, epsilon:float=0.00001, visualize:bool=False):

    if visualize:
        open3d.visualization.draw_geometries([cloud_dst + cloud_src])

    tree_dst = open3d.geometry.KDTreeFlann(cloud_dst)

    prev_error = float("inf")
    for i in range(max_iter):

        points_src, points_dst, distances = associate(cloud_src=cloud_src, cloud_dst=cloud_dst, tree_dst=tree_dst)

        T = get_transformation(cloud_src, cloud_dst, points_src, points_dst)
        cloud_src.transform(T)

        error = np.mean(distances)
        print(f'Iteration: {i+1}\t-\tError: {error}')

        if prev_error - error < epsilon:
            break

        prev_error = error
    
    if visualize:
        open3d.visualization.draw_geometries([cloud_dst + cloud_src])

def tr_icp(cloud_src:open3d.geometry.Geometry.Type.PointCloud, cloud_dst:open3d.geometry.Geometry.Type.PointCloud, max_iter:int=50, epsilon:float=0.00001, percentage:float=0.5, visualize:bool=False):

    if visualize:
        open3d.visualization.draw_geometries([cloud_dst + cloud_src])

    tree_dst = open3d.geometry.KDTreeFlann(cloud_dst)

    prev_error = float("inf")
    for i in range(max_iter):

        points_src, points_dst, distances = associate(cloud_src=cloud_src, cloud_dst=cloud_dst, tree_dst=tree_dst)
        s = distances.argsort(axis=0)
        points_src = points_src[s][:, 0 ,:3]
        points_dst = points_dst[s][:, 0 ,:3]

        T = get_transformation(cloud_src, cloud_dst, points_src[:int(points_src.shape[0]*percentage),:], points_dst[:int(points_dst.shape[0]*percentage),:])
        cloud_src.transform(T)

        error = np.mean(distances)
        print(f'Iteration: {i+1}\t-\tError: {error}')

        if prev_error - error < epsilon:
            break

        prev_error = error
    
    if visualize:
        open3d.visualization.draw_geometries([cloud_dst + cloud_src])

def associate(cloud_src:open3d.geometry.Geometry.Type.PointCloud, cloud_dst:open3d.geometry.Geometry.Type.PointCloud, tree_dst) -> List[NDArray]:

    array_src = np.asarray(cloud_src.points)
    distances = np.zeros(shape=(array_src.shape[0], 1), dtype=float)
    array_dst = np.asarray(cloud_dst.points)
    array_dst_ordered = np.zeros(shape=array_src.shape, dtype=float)

    for i in range(array_src.shape[0]):
        _, idx, _ = tree_dst.search_knn_vector_3d(cloud_src.points[i], 1)
        array_dst_ordered[i,:] = array_dst[idx, :]
        distances[i] = np.linalg.norm(array_src[i,:] - array_dst_ordered[i,:])

    return array_src, array_dst_ordered, distances

def get_transformation(cloud_src:open3d.geometry.Geometry.Type.PointCloud, cloud_dst:open3d.geometry.Geometry.Type.PointCloud, points_src:NDArray, points_dst:NDArray) -> NDArray[float]:

    mean_src, _ = open3d.geometry.compute_point_cloud_mean_and_covariance(cloud_src)
    mean_dst, _ = open3d.geometry.compute_point_cloud_mean_and_covariance(cloud_dst)

    points_src_shifted = points_src - mean_src
    points_dst_shifted = points_dst - mean_dst

    H = np.dot(points_src_shifted.T, points_dst_shifted)
    U, _, VT = np.linalg.svd(H)
    
    T = np.eye(N=4, dtype=float)
    R = np.dot(VT.T, U.T)
    T[:3,:3] = R
    T[:3,3] = mean_dst - np.dot(R, mean_src)

    return T


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        raise Exception('Bad arguments')

    main(sys.argv)