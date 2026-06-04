import torch
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

import sys
sys.path.append('./common/')
import models
from loss.inverse_warp import pose_vec2mat

from thop import clever_format
from thop import profile
import time

def profile_pose(encoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)

    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")

    return flops_e, params_e

def profile_posed(decoder, v):
    flops_d, params_d = profile(decoder, inputs=(v[0,:].unsqueeze(0),), verbose=False)
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")
    return flops_d, params_d
    
def profile_imu(encoder, x):
    flops_e, params_e = profile(encoder, inputs=(x[0, :, :].unsqueeze(0), ), verbose=False)
    flops_ie, params_ie = clever_format([flops_e, params_e], "%.3f")
    return flops_ie, params_ie
    
def profile_fusion(encoder, x, y):
    flops_e, params_e = profile(encoder, inputs=(x[0,:].unsqueeze(0), y[0,:].unsqueeze(0),), verbose=False)
    flops_ie, params_ie = clever_format([flops_e, params_e], "%.3f")
    return flops_ie, params_ie

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", type=str, default='', help="pretrained PoseNet path")
parser.add_argument("--pretrained-pose", type=str, default='', help="pretrained PoseNet path")
parser.add_argument("--pretrained-imu", type=str, default='')
parser.add_argument("--pretrained-fusion", type=str, default='')
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width") 
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, default= '', help="Dataset directory")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=5)
parser.add_argument("--sequences", default=['outdoor_robust_night1'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(img, args):
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float16)
    img = np.transpose(img, (2, 0, 1))
    img = (torch.from_numpy(img).float() / 2**14)
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img

@torch.no_grad()
def main():
    args = parser.parse_args()

    # load models
    pose_encoder = models.PoseEncoder(drop_path_rate=0.2).to(device)
    pose_decoder = models.PoseDecoder().to(device)
    imu_encoder = models.InertialEncoder(drop_path_rate=0.2).to(device)
    fusion = models.ModalityGraphFusion(feat_dim=256).to(device)

    weights = torch.load(args.pretrained_posenet)
    pose_encoder.load_state_dict(weights['state_dict'], strict=False)
    pose_encoder.cuda()
    pose_encoder.eval()

    weights_decoder = torch.load(args.pretrained_pose)
    pose_decoder.load_state_dict(weights_decoder['state_dict'], strict=False)
    pose_decoder.cuda()
    pose_decoder.eval()

    weights_imu = torch.load(args.pretrained_imu)
    imu_encoder.load_state_dict(weights_imu['state_dict'], strict=False)
    imu_encoder.cuda()
    imu_encoder.eval()

    weights_fusion = torch.load(args.pretrained_fusion)
    fusion.load_state_dict(weights_fusion['state_dict'], strict=False)
    fusion.cuda()
    fusion.eval()

    seq_length = 3

    # load data loader
    from eval_vivid.pose_evaluation_utils import test_framework_VIVID as test_framework
    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length=seq_length, step=1)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))
    all_final_poses = []
    all_gt_poses = []

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']
        imus = sample['imus']
        squence_imgs = []
        squence_imus = []

        for i, img in enumerate(imgs):
            img = load_tensor_image(img, args)
            squence_imgs.append(img)

        for i, imu in enumerate(imus):
            imu = np.genfromtxt(imu).astype(np.float16)[:6]
            imu = torch.from_numpy(imu)
            imu = imu.unsqueeze(0).unsqueeze(0).to(device)
            squence_imus.append(imu)

        global_pose = np.eye(4)
        poses = []
        poses.append(global_pose[0:3, :])

        for iter in range(seq_length - 1):
            images = torch.cat([squence_imgs[iter], squence_imgs[iter + 1]], dim=1)
            imu_inputs = torch.cat([squence_imus[iter], squence_imus[iter + 1]], dim=1)

            # t1 = time.time()
            imu_feature = imu_encoder(imu_inputs)
            pose_feature = pose_encoder(images)
            fused = fusion(pose_feature, imu_feature)
            pose = pose_decoder(fused)
            # t2 = time.time()
            # print((t2-t1)*1000)
            # flops_e, params_e = profile_pose(pose_encoder, images)
            # print("visual_encoder",flops_e, params_e)
            # flops_i, params_i = profile_imu(imu_encoder, imu_inputs)
            # print("imu_encoder",flops_i, params_i)
            # flops_f, params_f = profile_fusion(fusion, pose_feature, imu_feature)
            # print("fusion",flops_f, params_f)
            # flops_d, params_d = profile_posed(pose_decoder, fused)
            # print("pose_decoder", flops_d, params_d)
            # pose = pose_net(squence_imgs[iter], squence_imgs[iter + 1])
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()

            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :])

        final_poses = np.stack(poses, axis=0)
        all_final_poses.append(final_poses[-1:])
        all_gt_poses.append(sample['poses'][-1:])

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1])/np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])     
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


def compute_pose(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        pose = pose_net(tgt_img, ref_img).unsqueeze(1)
        poses.append(pose)
    poses = torch.cat(poses, 1)
    return poses


if __name__ == '__main__':
    main()
