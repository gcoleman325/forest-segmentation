import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import KMeans
import random
import os
import glob
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from open3d.web_visualizer import draw

def preprocess_point_cloud(pcd):
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    radius_normal = 0.2
    print("estimating normals")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = 0.5
    print("computing fpfh")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh = np.asarray(pcd_fpfh.data).T
    return pcd, fpfh

def cov_metrics(pcd):
    print("estimating covariances")
    covs = pcd.estimate_covariances()
    covs = np.asarray(pcd.covariances)

    print("computing eigen-based metrics")
    metrics = []
    for pt in covs:
        eigenvalues = np.linalg.eigvals(pt)
        e1, e2, e3 = eigenvalues
        
        linearity = (e1 - e2) / e1
        planarity = (e2 - e3) / e1
        scattering = e3 / e1
        omnivariance = (e1 * e2 * e3) ** (1 / 3)
        anisotropy = (e1 - e3) / e1
        eigentropy = -(e1 * np.log(e1) + e2 * np.log(e2) + e3 * np.log(e3))
        curvature = e3 / (e1 + e2 + e3)

        metrics.append((linearity, planarity, scattering, omnivariance, anisotropy, eigentropy, curvature))

    dtype = [('linearity', 'f8'), ('planarity', 'f8'), ('scattering', 'f8'), 
            ('omnivariance', 'f8'), ('anisotropy', 'f8'), ('eigentropy', 'f8'), 
            ('curvature', 'f8')]
    
    metrics_array = np.array(metrics, dtype=dtype)
  
    return np.array([tuple(row) for row in metrics_array])

def prep_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd, fpfh = preprocess_point_cloud(pcd)
    cov = cov_metrics(pcd)
    
    cov_headers = ['linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy', 'eigentropy', 'curvature']
    header = ['x', 'y', 'z'] + [f'feature{i}' for i in range(fpfh.shape[1])] + cov_headers
    all_metrics = np.hstack([np.asarray(pcd.points), fpfh, cov])

    metrics = pd.DataFrame(all_metrics, columns=header)
    metrics = metrics[metrics['z'] <= 0.25]
    return pcd, metrics

def process_multiple(folder_path):
    all_metrics = []

    paths = glob.glob(f"{folder_path}/**/*.pcd", recursive=True)
    paths = [os.path.normpath(path).replace(os.sep, '/') for path in paths]

    for file_ct, path in enumerate(paths, start=1):
        pcd, metrics = prep_pcd(path)
        metrics["scan_num"] = file_ct
        all_metrics.append(metrics)

    return all_metrics

df = process_multiple("C:/Users/ellie/OneDrive/Desktop/lidar_local/ccb_preprocessed")
all_metrics = pd.concat(df, ignore_index=True)
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(all_metrics)
all_metrics['cluster'] = kmeans.labels_

pcd = all_metrics[all_metrics['scan_num'] == 3]
just_clusters = pcd[["x", "y", "z", "cluster"]]
cluster_0 = just_clusters[just_clusters['cluster'] == 0]
cluster_1 = just_clusters[just_clusters['cluster'] == 1]
cluster_2 = just_clusters[just_clusters['cluster'] == 2]

zero = cluster_0[['x', 'y', 'z']].values
one = cluster_1[['x', 'y', 'z']].values
two = cluster_2[['x', 'y', 'z']].values


pcd0 = o3d.geometry.PointCloud()
pcd0.points = o3d.utility.Vector3dVector(zero)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(one)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(two)

o3d.visualization.draw_geometries([pcd0])
o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd2])