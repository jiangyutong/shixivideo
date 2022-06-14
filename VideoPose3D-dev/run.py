# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import transformations
import numpy as np
import matplotlib.gridspec as gridspec
from common.arguments import parse_args
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import matplotlib.pyplot as plt
from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
import evaluate
args = parse_args()
print(args)
import cv2
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_' + args.dataset + '.npz'
# npz_path = "/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/data/data_egopose.npz"
#
# # words.npy lables.npy
# data = np.load(npz_path, allow_pickle=True)
if args.dataset == 'egopose':
    from common.egopose_dataset import EgoposeDataset

    dataset = EgoposeDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset

    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
def show3Dpose_hm36(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor='b'
    rcolor='r'

    I = np.array( [0, 7, 8, 9, 8, 11, 12, 8, 14,  15, 0, 4, 5, 0,  1, 2])
    J = np.array( [7, 8, 9, 10,11, 12,13,14, 15, 16, 4, 5, 6, 1, 2, 3])

    LR = np.array([0, 0, 0, 0,
                    0, 0, 0,
                    1, 1, 1,
                    0, 0, 0,
                    1, 1, 1], dtype=bool)
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = rcolor if LR[i] else lcolor)
    ax.set_xlim3d([1, -1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([2, 0])
    ax.set_aspect('auto')
    # ax.set_xlim3d([2, -2])
    # ax.set_ylim3d([-1, 2])
    # ax.set_zlim3d([3, -1])
    # ax.view_init(-69, 90)

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = True)
    ax.tick_params('y', labelleft = True)
    ax.tick_params('z', labelleft = True)
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        # for idx, pos_3d in enumerate(anim['positions_3d']):
        #     #
        #     pos_3d[0:10] -= pos_3d[10]  # Remove global offset, but keep trajectory in first position
        #     pos_3d[11:] -= pos_3d[10]
            # pos_3d -= pos_3d[0]
            # pos_3d[1:] -= pos_3d[:1]
        #     anim['positions_3d'][idx] = pos_3d
            # ax = plt.subplot(111, projection='3d')
            # show3Dpose_hm36(pos_3d, ax)
            # plt.show()
            # # plt.savefig(output_dir + str(i).zfill(5) + '_3Dh.png', dpi=200, format='png', bbox_inches='tight')
            # plt.close()

        for idx, pos_2d in enumerate(anim['positions']):
            pos_2d = normalize_screen_coordinates(pos_2d, w=256, h=256)
            anim['positions'][idx] = pos_2d

keypoints_metadata = {
    'layout_name': 'egopose',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())


def fetch(subjects, subset=1):
    out_poses_3d = []
    out_poses_2d = []
    out_imag_path = []
    out_camera_params=[]
    out_camera_rot=[]
    out_camera_trans = []
    for subject in subjects:
        for env in dataset[subject].keys():
            poses_2d = dataset[subject][env]['positions']
            out_poses_2d.append(np.array(poses_2d))
            poses_3d = dataset[subject][env]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            out_poses_3d.append(np.array(poses_3d))
            image_path=dataset[subject][env]['image_path']
            rot=dataset[subject][env]['rot']
            trans=dataset[subject][env]['trans']
            out_imag_path.append(np.array(image_path))
            out_camera_rot.append(np.array(rot))
            out_camera_trans.append(np.array(trans))
    # out_camera_params.append(out_camera_rot)
    # out_camera_params.append(out_camera_trans)
    out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d,out_imag_path,out_camera_rot,out_camera_trans


cameras_test, poses_test, poses_test_2d,img_path_test,rot_test,trans_test = fetch(["male_008_a_a"])
cameras_valid, poses_valid, poses_valid_2d,img_path_vail,rot_vail,trans_vail = fetch(["female_004_a_a","female_008_a_a","female_010_a_a","female_012_a_a","female_012_f_s","male_001_a_a","male_002_a_a","female_004_a_a","male_004_f_s","male_006_a_a","male_007_f_s","male_010_a_a","male_014_f_s"])

filter_widths = [int(x) for x in args.architecture.split(',')]
if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                               dataset.skeleton().num_joints(),
                                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                               channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                    dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                    channels=args.channels,
                                    dense=args.dense)

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                          filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

    if args.evaluate and 'model_traj' in checkpoint:
        # Load trajectory model if it contained in the checkpoint (e.g. for inference in the wild)
        model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                   filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                   channels=args.channels,
                                   dense=args.dense)
        if torch.cuda.is_available():
            model_traj = model_traj.cuda()
        model_traj.load_state_dict(checkpoint['model_traj'])
    else:
        model_traj = None
eval_body = evaluate.EvalBody()
eval_upper = evaluate.EvalUpperBody()
eval_lower = evaluate.EvalLowerBody()
test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                    joints_right=joints_right,rot=rot_vail,trans=trans_vail,)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d,img_path_train ,rot,trans= fetch(
        ['female_001_a_a', 'female_002_f_s', 'female_003_a_a', 'female_007_a_a', 'female_009_a_a', 'female_011_a_a',
         'female_014_a_a', 'female_015_a_a', 'male_003_f_s', 'male_004_a_a', 'male_005_a_a', 'male_006_f_s',
         'male_007_a_a', 'male_008_a_a', 'male_009_a_a', 'male_010_f_s', 'male_011_f_s', 'male_014_a_a'])

    lr = args.learning_rate
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train,poses_train, poses_train_2d,
                                       args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                       joints_right=joints_right,rot=rot,trans=trans)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False,rot=rot,trans=trans)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()

        for _, batch_3d, batch_2d,rot,trans in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            # inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()
            # if semi_supervised:
            #     model_traj.load_state_dict(model_traj_train.state_dict())
            #     model_traj.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0

            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d,rot,trans in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    # inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_traj = inputs_3d[:, :, 0].clone()
                    # inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]



                losses_3d_valid.append(epoch_loss_3d_valid / N)
                # if semi_supervised:
                #     losses_traj_valid.append(epoch_loss_traj_valid / N)
                #     losses_2d_valid.append(epoch_loss_2d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d ,rot,trans in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    # inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_traj = inputs_3d[:, :, :0].clone()
                    # inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                epoch_loss_2d_train_unlabeled_eval = 0

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        model_pos_train.set_bn_momentum(momentum)
        # if semi_supervised:
        #     model_traj_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')

checkpoint = torch.load('/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/checkpointNoneCentered/epoch_200.bin', map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos.load_state_dict(checkpoint['model_pos'])
# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
        output_dir = "egopose_hm36_trainval/"
        os.makedirs(output_dir, exist_ok=True)
        for cam, batch, batch_2d ,rot,trans in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
                                                                         joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            rot=rot.squeeze(0)
            trans=trans.squeeze(0)
            preworlddataall=[]
            y_output=predicted_3d_pos.squeeze(0).cpu().numpy()
            y_target = inputs_3d.squeeze(0).numpy()
            # f = open("/media/zlz422/jyt/egpose/XR-EGOPOSE/data/ValSet/male_008_a_a/env_002/cam_down/json/male_008_a_a_000008.json")
            # data = json.load(f)


            Mmaya = np.array([[1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])

            # 相机外参
            # translation = np.array(trans[0])
            # rotation = np.array(rot[0]) * np.pi / 180.0
            # Mf = transformations.euler_matrix(rotation[0],
            #                                   rotation[1],
            #                                   rotation[2],
            #                                   'sxyz')
            # Mf[0:3, 3] = translation
            # Mf = np.linalg.inv(Mf)
            # # M为相机的旋转和平移矩阵
            # M = Mmaya.T.dot(Mf)
            #
            # # 世界坐标系的位置
            # joints = np.vstack([j['trans'] for j in data['joints']]).T
            #
            # # 世界坐标到相机坐标系
            # Xj = M[0:3, 0:3].dot(joints) + M[0:3, 3:4]
            #
            # # 相机坐标系下的3d坐标位置
            # pts3d_json = data["pts3d_fisheye"]
            # print(np.allclose(Xj, pts3d_json))
            #
            # # 从相机坐标系返回世界坐标系
            # M_inv = np.linalg.inv(M[0:3, 0:3])
            # joints_ = M_inv.dot((pts3d_json - M[0:3, 3:4])).T
            # joints2_ = M_inv.dot((y_target[0].T - M[0:3, 3:4]/100)).T
            # joints3_ = M_inv.dot((y_output[0].T - M[0:3, 3:4] / 100)).T
            # y_output[:, 10, :] = 0
            # y_target[:, 10, :] = 0
            for i in range(len(y_output)):
                if i>2000: break
                ax = plt.subplot(111, projection='3d')
                translation = np.array(trans[i])
                rotation = np.array(rot[i]) * np.pi / 180.0
                Mf = transformations.euler_matrix(rotation[0],
                                                  rotation[1],
                                                  rotation[2],
                                                  'sxyz')
                Mf[0:3, 3] = translation
                Mf = np.linalg.inv(Mf)
                # M为相机的旋转和平移矩阵
                M = Mmaya.T.dot(Mf)



                # 从相机坐标系返回世界坐标系
                M_inv = np.linalg.inv(M[0:3, 0:3])
                predictionw = M_inv.dot((y_output[i].T - M[0:3, 3:4]/100)).T
                show3Dpose_hm36(y_output[i], ax)
                # plt.show()
                preworlddataall.append(predictionw)
                plt.savefig(output_dir + str(i).zfill(5) + '_3DHeadCentered2.png', dpi=200, format='png', bbox_inches='tight')
                plt.close()
            #
            #
            # # predictionw = camera_to_world(y_output, R=cam['orientation'], t=cam['translation'])
            # # predictionw = camera_to_world(y_output, R=rot, t=trans)
            # np.save("/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/reslutdata/preworlddataall.npy",predictionw)
            # np.save("/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/reslutdata/predata.npy", y_output)
            # for i in range(len(y_output)):
            #     if i > 800: break
            #     ax = plt.subplot(111, projection='3d')
            #     show3Dpose_hm36(y_output[i], ax)
            #     # plt.show()
            #     plt.savefig(output_dir + str(i).zfill(5) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            #     plt.close()

            if return_predictions:
                # y_target[0:10] += y_target[10]  # Remove global offset, but keep trajectory in first position
                # y_target[11:] += y_target[10]
                # y_output[0:10] += y_output[10]  # Remove global offset, but keep trajectory in first position
                # y_output[11:] += y_output[10]
                eval_body.eval(y_output, y_target, action)
                eval_upper.eval(y_output, y_target, action)
                eval_lower.eval(y_output, y_target, action)
                res = {'FullBody': eval_body.get_results(),
                       'UpperBody': eval_upper.get_results(),
                       'LowerBody': eval_lower.get_results()}
                print(res)
                # return predicted_3d_pos.squeeze(0).cpu().numpy()

    #         inputs_3d = torch.from_numpy(batch.astype('float32'))
    #         if torch.cuda.is_available():
    #             inputs_3d = inputs_3d.cuda()
    #
    #         if test_generator.augment_enabled():
    #             inputs_3d = inputs_3d[:1]
    #
    #         error = mpjpe(predicted_3d_pos, inputs_3d)
    #         epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
    #                                                                                      inputs_3d).item()
    #
    #         epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
    #         N += inputs_3d.shape[0] * inputs_3d.shape[1]
    #
    #         inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
    #         predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
    #
    #         epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
    #
    #         # Compute velocity error
    #         epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
    #
    # if action is None:
    #     print('----------')
    # else:
    #     print('----' + action + '----')
    # e1 = (epoch_loss_3d_pos / N) * 1000
    # e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    # e3 = (epoch_loss_3d_pos_scale / N) * 1000
    # ev = (epoch_loss_3d_vel / N) * 1000
    # print('Test time augmentation:', test_generator.augment_enabled())
    # print('Protocol #1 Error (MPJPE):', e1, 'mm')
    # print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    # print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    # print('Velocity Error (MPJVE):', ev, 'mm')
    # print('----------')
    #
    # return e1, e2, e3, ev


def img2video(video_path, output_dir):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    names = sorted(glob.glob(os.path.join(output_dir, '*_3DHeadCentered2.png')),key=lambda s:int(s[-23:-20]))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    videoWrite = cv2.VideoWriter(video_path, fourcc, 5,size)
    i=0
    for name in names:
        if i>900: break
        i+=1
        print(name)
        img = cv2.imread(name)
        videoWrite.write(img)
    videoWrite.release()
output_dir = "egopose_hm36_trainval/"
predictionw=np.load("/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/reslutdata/predata.npy")

# for i in range(len(predictionw)):
#     if i > 800: break
#     ax = plt.subplot(111, projection='3d')
#     show3Dpose_hm36(predictionw[i], ax)
#     # plt.show()
#     plt.savefig(output_dir + str(i).zfill(5) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
#     plt.close()
if args.render:
    print('Rendering...')

    # input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    # cameras_valid, poses_valid, poses_valid_2d ,img_path_vail,rot_vail,trans_vail= fetch(["female_004_a_a","female_008_a_a","female_010_a_a","female_012_a_a","female_012_f_s","male_001_a_a","male_002_a_a","female_004_a_a","male_004_f_s","male_006_a_a","male_007_f_s","male_010_a_a","male_014_f_s"])
    cameras_valid, poses_valid, poses_valid_2d, img_path_vail, rot_vail, trans_vail = fetch(["male_008_a_a"])
        # fetch(["female_004_a_a","female_008_a_a","female_010_a_a","female_012_a_a","female_012_f_s","male_001_a_a","male_002_a_a","female_004_a_a","male_004_f_s","male_006_a_a","male_007_f_s","male_010_a_a","male_014_f_s"])

    vail_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,rot_vail,trans_vail,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right)
    prediction = evaluate(vail_generator, return_predictions=True)
#     # predictionw = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
#     # for i in range(len(predictionw)):
#     #     if i>800: break
#     #     ax = plt.subplot(111, projection='3d')
#     #     show3Dpose_hm36(predictionw[i], ax)
#     #     # plt.show()
#     #     plt.savefig(predictionw + str(i).zfill(5) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
#     #     plt.close()
#     output_dir = "egopose_hm36_trainval/"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # for i in range(len(prediction)):
#     #     if i>500: break
#     #     ax = plt.subplot(111, projection='3d')
#     #     show3Dpose_hm36(prediction[i], ax)
#     #     # plt.show()
#     #     print('save:{}'.format(output_dir + str(i)))
#     #     plt.savefig(output_dir + str(i) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
#     #     plt.close()
#     # if model_traj is not None and ground_truth is None:
#     #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
#     #     prediction += prediction_traj
#
#     if args.viz_export is not None:
#         print('Exporting joint positions to', args.viz_export)
#         # Predictions are in camera space
#         np.save(args.viz_export, prediction)
#
#     if args.viz_output is not None:
#         if ground_truth is not None:
#             # Reapply trajectory
#             trajectory = ground_truth[:, :1]
#             ground_truth[:, 1:] += trajectory
#             prediction += trajectory
#
#         # Invert camera transformation
#         cam = dataset.cameras()
#         if ground_truth is not None:
#             prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
#             ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
#         else:
#
#             def camera_to_world2(prediction):
#                 translation = np.array([1.7888190728441975, 151.6691129267569, 14.097993796871524])
#                 rotation = np.array([270.1195067877059, 0.4091768313486091, -0.10697654719674095]) * np.pi / 180.0
#                 Mmaya = np.array([[1, 0, 0, 0],
#                                   [0, -1, 0, 0],
#                                   [0, 0, -1, 0],
#                                   [0, 0, 0, 1]])
#                 import transformations
#                 Mf = transformations.euler_matrix(rotation[0], rotation[1], rotation[2], 'sxyz')
#                 Mf[0:3, 3] = translation
#                 Mf = np.linalg.inv(Mf)
#                 # M为相机的旋转和平移矩阵
#                 M = Mmaya.T.dot(Mf)
#                 M_inv = np.linalg.inv(M[0:3, 0:3])
#                 joints_ = M_inv.dot((prediction.transpose(1, 0) - M[0:3, 3:4])) / 100
#                 joints_ = joints_.transpose(1, 0)
#                 return joints_
#
#
#             for idx, pos_3d in enumerate(prediction):
#                 # pos_3d[1:] += pos_3d[:1]
#                 pos_3d[:, 0] = -pos_3d[:, 0]
#
#                 pos_3d[:, 2] = -pos_3d[:, 2]
#                 pos_3d[:, 2] -= np.min(pos_3d[:, 2])
#                 prediction[idx] = pos_3d
#
#         anim_output = {'Reconstruction': prediction}
#         if ground_truth is not None and not args.viz_no_ground_truth:
#             anim_output['Ground truth'] = ground_truth
#
#         input_keypoints = poses_valid_2d[0][..., :2]
#         input_keypoints = image_coordinates(input_keypoints, w=256, h=256)
#
#         from common.visualization import render_animation
#         #
#         # render_animation(input_keypoints, keypoints_metadata, anim_output,
#         #                  dataset.skeleton(), dataset.fps(), args.viz_bitrate, 40, args.viz_output,
#         #                  limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
#         #                  input_video_path=args.viz_video, viewport=(256, 256),
#         #                  input_video_skip=args.viz_skip)
#img2video("/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/egopose_hm36_trainval/resultprePelvisCentered2.mp4", "/media/zlz422/jyt/VideoPose3D-dev/VideoPose3D-dev/egopose_hm36_trainval/")
