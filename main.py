import argparse
import os
import numpy as np
import glob
import time

from utils import create_folder, get_frame, gen_points, create_points_dict, plot_overlap_for_all_camera, get_overlap_for_camera, reconstruct_points

WIDTH = 1920
HEIGHT = 1080
NEW_WIDTH = 640
NEW_HEIGHT = 480


def run_one_video_folder(opt, video_folder):
    frame_folder = os.path.join(video_folder, 'frames')
    points_folder = os.path.join(frame_folder, 'points')
    get_frame(video_folder, frame_folder)

    frame_subfolders = glob.glob(os.path.join(frame_folder, '*'))
    frame_subfolders = sorted(frame_subfolders)
    for frame_subfolder in frame_subfolders:
        print(frame_subfolder)
        t1 = time.time()
        gen_points(opt, frame_subfolder)
        t2 = time.time()
        t = t2-t1
        points_folder = os.path.join(frame_subfolder, 'points')
        points_dict = create_points_dict(points_folder)
        plot_overlap_for_all_camera(frame_subfolder, points_dict)
        for key in sorted(list(points_dict.keys())):
            coor_list = get_overlap_for_camera(points_dict[key])
            npz_file_path = os.path.join(frame_subfolder, key + '_coor_list')
            np.savez(npz_file_path, coor_list, t)
    
    for frame_subfolder in frame_subfolders:
        img_paths = glob.glob(os.path.join(frame_subfolder, '*[!_final].jpg'))
        img_paths = sorted(img_paths)
        npz_paths = glob.glob(os.path.join(frame_subfolder, '*.npz'))
        npz_paths = sorted(npz_paths)
        for img_path, npz_path in zip(img_paths, npz_paths):
            reconstruct_points(img_path, npz_path)

    print(f'Done for {video_folder}\n')


def write_txt(test_video_folder, team_folder):
    create_folder(team_folder)
    video_subfolders = glob.glob(os.path.join(test_video_folder, 'videos/*'))
    video_subfolders = sorted(video_subfolders)
    for video_subfolder in video_subfolders:
        video_sub_name = os.path.basename(video_subfolder)
        create_folder(team_folder + '/' + video_sub_name)
        videos = glob.glob(os.path.join(video_subfolder, '*.mp4'))
        for cam in videos:
            cam_name = os.path.basename(cam).split('.')[0]
            txt_path = team_folder + '/' + video_sub_name + '/' + cam_name + '.txt'
            with open(txt_path, 'w') as f:
                frame_folder = os.path.join(video_subfolder, 'frames')
                frame_subfolders = glob.glob(os.path.join(frame_folder, '*'))
                frame_subfolders = sorted(frame_subfolders)
                for frame_subfolder in frame_subfolders:
                    frame_name = 'frame_' + os.path.basename(frame_subfolder)
                    npz = np.load(frame_subfolder + '/' + cam_name + '_' + frame_name + '_final_reconstructed.npz')
                    coor = [str(x) for x in npz['arr_0'].reshape(-1).tolist()]
                    t = npz['arr_1']
                    f.write(frame_name + ', \n')
                    f.write('(' + ','.join(coor) + '), ' + str(1.0/t))
                    f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--video_folder', type=str, default='data_btc/data')
    
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')

    opt = parser.parse_args()

    test_video_folder = 'data_btc/Private_Test'
    video_subfolders = glob.glob(os.path.join(test_video_folder, 'videos/*'))
    video_subfolders = sorted(video_subfolders)
    for video_subfolder in video_subfolders:
        run_one_video_folder(opt, video_subfolder)

    write_txt(test_video_folder, 'Next Gen AI')
    

    # frame_subfolder = 'data_btc/Public_Test (copy)/videos/scene4cam_01/frames/9'
    # gen_points(opt, frame_subfolder)
    # points_folder = os.path.join(frame_subfolder, 'points')
    # points_dict = create_points_dict(points_folder)
    # plot_overlap_for_all_camera(frame_subfolder, points_dict)

    # for key, value in points_dict.items():
    #     coor_list = get_overlap_for_camera(value)
    #     print(key)
    #     print(coor_list.shape)
    #     npz_file_path = os.path.join(frame_subfolder, key + '_coor_list')
    #     np.savez(npz_file_path, coor_list)
    # img_paths = glob.glob(os.path.join(frame_subfolder, '*[!_final].jpg'))
    # img_paths = sorted(img_paths)
    # npz_paths = glob.glob(os.path.join(frame_subfolder, '*.npz'))
    # npz_paths = sorted(npz_paths)
    # # print(img_paths)
    # # print(npz_paths)
    # for img_path, npz_path in zip(img_paths, npz_paths):
    #     reconstruct_points(img_path, npz_path)    