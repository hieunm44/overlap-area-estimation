from scipy.spatial import ConvexHull
import cv2
import numpy as np
import glob
import os
from shapely.geometry import Polygon


def get_convex(points):
    hull = ConvexHull(points)
    return hull.vertices


def crop_image(img, pts):
    pts = pts.astype(np.int32)
    # Create an empty mask with the same shape as the original image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw the polyline on the mask
    cv2.fillPoly(mask, [pts], color=(255))

    # Extract the region of interest (ROI) from the original image using the mask
    cropped_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('img', cropped_img)
    cv2.waitKey(0)
    # Display the cropped image
    return cropped_img


def plot(img_path, corrdinate_path, cnt):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(320, 240))
    corrdinate = np.load(corrdinate_path)
    corrdinate = np.squeeze(corrdinate, 0)
    idx = get_convex(corrdinate)

    selected_corr = corrdinate[idx]
    cv2.polylines(img, [selected_corr], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('img' + str(cnt), img)
    cv2.waitKey(0)


def get_overlap(img_path, corrdinate_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 240))
    corrdinate = np.load(corrdinate_path)
    corrdinate = np.squeeze(corrdinate, 0)
    corrdinate = corrdinate[get_convex(corrdinate)]
    return crop_image(img, corrdinate)


# crop_1_2_1 = get_overlap(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Anh1_2\img1.jpg",
#                          r"E:\juntion2023_2\SuperGluePretrainedNetwork\1_2\corr_1.npy")
# crop_3_4_3 = get_overlap(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Anh3_4\img1.jpg",
#                          r"E:\juntion2023_2\SuperGluePretrainedNetwork\3_4\corr_1.npy")
# crop_1_2_2 = get_overlap(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Anh1_2\img2.jpg",
#                          r"E:\juntion2023_2\SuperGluePretrainedNetwork\1_2\corr_2.npy")
# crop_3_4_4 = get_overlap(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Anh3_4\img2.jpg",
#                          r"E:\juntion2023_2\SuperGluePretrainedNetwork\3_4\corr_2.npy")
# cv2.imwrite("assets/Cross_1_3/crop_1_2_1.jpg", crop_1_2_1)
# cv2.imwrite("assets/Cross_1_3/croqp_3_4_3.jpg", crop_3_4_3)

# plot(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Anh1_2\img1.jpg",
#      r"E:\juntion2023_2\SuperGluePretrainedNetwork\1_2\corr_1.npy", 1)


# plot(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Cross_1_3\crop_1_2_1.jpg",
#      r"E:\juntion2023_2\SuperGluePretrainedNetwork\Cross_1_3\corr_1.npy", 1)
# plot(r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Cross_1_3\crop_3_4_3.jpg",
#      r"E:\juntion2023_2\SuperGluePretrainedNetwork\Cross_1_3\corr_2.npy", 2)

def crop_image_2(img, pts_list):
    img = cv2.resize(img, (640, 480))
    print(img.shape)
    pts1, pts2, pts3 = pts_list[0], pts_list[1], pts_list[2]
    pts1, pts2, pts3 = pts1.astype(np.int32), pts2.astype(np.int32), pts3.astype(np.int32)
    # Create an empty mask with the same shape as the original image
    mask1 = np.zeros(shape=img.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(shape=img.shape[:2], dtype=np.uint8)
    mask3 = np.zeros(shape=img.shape[:2], dtype=np.uint8)

    # pts1 = np.squeeze(pts1, 0)
    pts1 = pts1[get_convex(pts1)]
    pts2 = pts2[get_convex(pts2)]
    pts3 = pts3[get_convex(pts3)]
    # Draw the polyline on the mask
    cv2.fillPoly(mask1, [pts1], color=(255))
    cv2.fillPoly(mask2, [pts2], color=(255))
    cv2.fillPoly(mask3, [pts3], color=(255))

    # Extract the region of interest (ROI) from the original image using the mask
    cropped_img1 = cv2.bitwise_and(img, img, mask=mask1)
    cropped_img2 = cv2.bitwise_and(img, img, mask=mask2)
    cropped_img3 = cv2.bitwise_and(img, img, mask=mask3)

    cv2.imshow('img1', cropped_img1)
    cv2.imshow('img2', cropped_img2)
    cv2.imshow('img3', cropped_img3)
    cv2.waitKey(0)
    # Display the cropped image
    # return cropped_img


# Todo:
def get_overlap_for_camera(pts_list):
    pts1, pts2, pts3 = pts_list[0], pts_list[1], pts_list[2]
    pts1, pts2, pts3 = pts1.astype(np.int32), pts2.astype(np.int32), pts3.astype(np.int32)
    # Create an empty mask with the same shape as the original image
    # pts1 = np.squeeze(pts1, 0)
    pts1 = pts1[get_convex(pts1)].tolist()
    pts2 = pts2[get_convex(pts2)].tolist()
    pts3 = pts3[get_convex(pts3)].tolist()
    # Draw the polyline on the mask
    Poly_pts1 = Polygon(pts1)
    Poly_pts2 = Polygon(pts2)
    Poly_pts3 = Polygon(pts3)

    inter1_2 = Poly_pts1.intersection(Poly_pts2)
    return np.array(list(inter1_2.intersection(Poly_pts3).exterior.coords))


def plot_overlap_for_one_camera(img, pts_list):
    Poly_pts1 = Polygon(pts_list[0][get_convex(pts_list[0])].tolist())
    for pt in pts_list[1:]:
        pt = Polygon(pt[get_convex(pt)].tolist())
        Poly_pts1 = Poly_pts1.intersection(pt)
    Poly_pts1 = np.array(list(Poly_pts1.exterior.coords)).astype(np.int32)
    cv2.polylines(img, [Poly_pts1], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


def plot_overlap_for_all_camera(img_folder, points_dict):
    try:
        files_name = os.listdir(img_folder)
        for file_name in files_name:
            img_path = os.path.join(img_folder, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, dsize=(640, 480))
            camera_name = '_'.join(file_name.split('_')[0:4])
            point_list = points_dict[camera_name]
            img = plot_overlap_for_one_camera(img, point_list)
            cv2.imshow(camera_name, img)
            cv2.waitKey(0)
    except:
        return


def create_points_dict(points_folder):
    files_name = os.listdir(points_folder)
    files_name = sorted(files_name)

    points_dict = dict()
    for file_name in files_name:
        npz = np.load(os.path.join(points_folder, file_name))
        points_1, points_2 = npz['mkpts0'], npz['mkpts1']
        tmp = file_name.split('_')
        cam1 = '_'.join(tmp[0:4])
        cam2 = '_'.join(tmp[6:10])
        if cam1 not in points_dict.keys():
            points_dict[cam1] = []
        if cam2 not in points_dict.keys():
            points_dict[cam2] = []
        points_dict[cam1].append(points_1)
        points_dict[cam2].append(points_2)

    return points_dict


path = r'E:\juntion2023_2\SuperGluePretrainedNetwork\out_frame\30\Points'
points_dict = create_points_dict(path)
# print(points_dict['192_168_5_102'][0].shape)
img_path = r'E:\juntion2023_2\SuperGluePretrainedNetwork\out_frame\30\192_168_5_101_frame_30.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 480))

plot_overlap_for_all_camera(r"E:\juntion2023_2\SuperGluePretrainedNetwork\out_frame\30",
                            points_dict)
# cam1_overlap = get_overlap_for_camera(points_dict['192_168_5_103']).astype(np.int32)
# cv2.polylines(img, [cam1_overlap], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# npz_name = '192_168_5_101_frame_30_192_168_5_102_frame_30_matches.npz'
# npz = np.load(os.path.join(path, npz_name))
# pts1 = npz['mkpts0']
# pts2 = npz['mkpts1']
# print(pts2.shape)
#
# pts1 = pts1[get_convex(pts1)].astype(np.int32)
# pts2 = pts2[get_convex(pts2)].astype(np.int32)
# img = cv2.imread(r"E:\juntion2023_2\SuperGluePretrainedNetwork\out_frame\30\192_168_5_102_frame_30.jpg")
# img = cv2.resize(img, (640, 480))
#
# img2 = cv2.imread(r"E:\juntion2023_2\SuperGluePretrainedNetwork\out_frame\30\192_168_5_101_frame_30.jpg")
# img2 = cv2.resize(img2, (640, 480))
# cv2.polylines(img,
#               [pts2], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(img2,
#               [pts1], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cv2.imshow('img', img)
# cv2.imshow('img2', img2)
# cv2.waitKey(0)

# path = r'E:\juntion2023_2\SuperGluePretrainedNetwork'
# img_name = 'assets\Anh1_2\img1.jpg'
# points_name = '1_2/corr_1.npy'
# points = np.load(os.path.join(path, points_name))
# img2 = cv2.imread(os.path.join(path, img_name))
# img2 = cv2.resize(img2, (320, 240))
# crop_image_2(img2, points, points, points)
