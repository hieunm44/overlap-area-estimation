import cv2
import os
import glob
import numpy as np
from pathlib import Path
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
import torch
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

WIDTH = 1920
HEIGHT = 1080
NEW_WIDTH = 640
NEW_HEIGHT = 480

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_frame(video_folder, frame_folder):
    videos = glob.glob(os.path.join(video_folder, '*.mp4'))
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        cnt = 0
        while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                if cnt % 1 == 0:
                    out_folder_frame = os.path.join(frame_folder, str(cnt))
                    create_folder(out_folder_frame)
                    cv2.imwrite(os.path.join(out_folder_frame,
                                             os.path.basename(video_path).split('.')[0] + '_frame_' + str(
                                                 cnt) + '.jpg'),
                                frame)
            else:
                break
            cnt += 1
    # print('Extracted frames.')


def gen_points(opt, frame_folder):
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    pairs = []
    frame_path = glob.glob(os.path.join(frame_folder, '*.jpg'))
    
    for i in range(len(frame_path)):
        for j in range(i+1, len(frame_path)):
            pairs.append([frame_path[i], frame_path[j]])
    points_folder = os.path.join(frame_folder, 'points')
    create_folder(points_folder)
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = points_folder + '/{}_{}_matches.npz'.format(stem0, stem1)

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            name1, device, opt.resize, rot1, opt.resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                name0, frame_folder + '/' + name1))
            exit(1)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']


        # Keep the matching keypoints.
        # print(matches)
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        # print('a', mkpts0.shape)
        # print('b', mkpts1.shape)
        
        out_matches = {'mkpts0': mkpts0, 'mkpts1': mkpts1}
        np.savez(str(matches_path), **out_matches)

    # print('Extracted points.')


def create_points_dict(points_folder):
    files_paths = glob.glob(os.path.join(points_folder, '*.npz'))
    files_paths = sorted(files_paths)

    points_dict = dict()
    for files_path in files_paths:
        npz = np.load(files_path)
        points_1, points_2 = npz['mkpts0'], npz['mkpts1']
        file_name = os.path.basename(files_path)
        tmp = file_name.split('_')
        cam1 = '_'.join(tmp[0:2])
        cam2 = '_'.join(tmp[4:6])
        # print('cam1', cam1)
        # print('cam2', cam2)
        # print(tmp)
        if cam1 not in points_dict.keys():
            points_dict[cam1] = []
        if cam2 not in points_dict.keys():
            points_dict[cam2] = []
        points_dict[cam1].append(points_1)
        points_dict[cam2].append(points_2)

    return points_dict


def get_convex(points):
    hull = ConvexHull(points)
    return hull.vertices


def plot_overlap_for_one_camera(img, pts_list):
    # print(pts_list)
    Poly_pts1 = Polygon(pts_list[0][get_convex(pts_list[0])].tolist())
    for pt in pts_list[1:]:
        pt = Polygon(pt[get_convex(pt)].tolist())
        Poly_pts1 = Poly_pts1.intersection(pt)
    Poly_pts1 = np.array(list(Poly_pts1.exterior.coords))
    cv2.polylines(img, [Poly_pts1.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


def plot_overlap_for_all_camera(img_folder, points_dict):
    files_paths = glob.glob(os.path.join(img_folder, '*.jpg'))
    for file_path in files_paths:
        img = cv2.imread(file_path)
        img = cv2.resize(img, dsize=(NEW_WIDTH, NEW_HEIGHT))
        file_name = os.path.basename(file_path)
        camera_name = '_'.join(file_name.split('_')[0:2])
        point_list = points_dict[camera_name]
        # print(len(point_list))
        plot_overlap_for_one_camera(img, point_list)
        cv2.imwrite(file_path.split('.')[0] + '_final.jpg', img)
        # cv2.imshow(camera_name, img)
        # cv2.waitKey(0)
    
    # print('Plotted overlap for all cameras.')


def get_overlap_for_camera(points_list):
    # Create an empty mask with the same shape as the original image
    Poly_points = []
    for points in points_list:
        points_cvx = points[get_convex(points)].tolist()
        Poly_points.append(points_cvx)

    if len(Poly_points) > 2:
        inter = Polygon(Poly_points[0]).intersection(Polygon(Poly_points[1]))
        for i in range(2, len(Poly_points), 1):
            inter = inter.intersection(Polygon(Poly_points[i]))
    else:
        inter = Polygon(Poly_points[0])          

    return np.array(list(inter.exterior.coords))


def reconstruct_points(img_path, points_path):
    img = cv2.imread(img_path)
    npz, t = np.load(points_path)['arr_0'], np.load(points_path)['arr_1']
    # print(points_path, npz.shape)
    reconstructed_npz = np.zeros_like(npz)
    reconstructed_npz[:, 0] = npz[:, 0] / NEW_WIDTH * WIDTH
    reconstructed_npz[:, 1] = npz[:, 1] / NEW_HEIGHT * HEIGHT
    cv2.polylines(img, [reconstructed_npz.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(img_path.split('.')[0] + '_final_reconstructed.jpg', img)
    npz_file_path = img_path.split('.')[0] + '_final_reconstructed.npz'
    np.savez(npz_file_path, reconstructed_npz, t)
    # cv2.waitKey(0)

    # print('Reconstructed images and points.')



