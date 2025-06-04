import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from tasks.protopnet.receptive_field import compute_rf_prototype
from tasks.protopnet.helpers import makedir, find_high_activation_crop


def push_prototypes(dataloader,
                    prototype_network_parallel,
                    class_specific=True,
                    preprocess_input_function=None,
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,
                    epoch_number=None,
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,
                    log=print,
                    prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = getattr(prototype_network_parallel, 'module', prototype_network_parallel).prototype_shape

    n_prototypes = getattr(prototype_network_parallel, 'module', prototype_network_parallel).num_prototypes

    global_min_proto_dist = np.full(n_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-' + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = getattr(prototype_network_parallel, 'module', prototype_network_parallel).num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):

        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir,
                             proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    getattr(prototype_network_parallel, 'module', prototype_network_parallel).prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    end = time.time()
    log('\tpush time: \t{0}'.format(end - start))


def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist,
                               global_min_fmap_patches,
                               proto_rf_boxes,
                               proto_bound_boxes,
                               class_specific=True,
                               search_y=None,
                               num_classes=None,
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        protoL_input_torch, proto_dist_torch = getattr(prototype_network_parallel, 'module', prototype_network_parallel).push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = getattr(prototype_network_parallel, 'module', prototype_network_parallel).prototype_shape

    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(getattr(prototype_network_parallel, 'module', prototype_network_parallel).prototype_class_identity[j]).item()
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]
        else:
            proto_dist_j = proto_dist_[:, j, :, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                     :,
                                     fmap_height_start_index:fmap_height_end_index,
                                     fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            protoL_rf_info = getattr(prototype_network_parallel, 'module', prototype_network_parallel).proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                       rf_prototype_j[3]:rf_prototype_j[4], :]

            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if getattr(prototype_network_parallel, 'module', prototype_network_parallel).prototype_activation_function == 'log':
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1) / (proto_dist_img_j + getattr(prototype_network_parallel, 'module', prototype_network_parallel).epsilon))
            elif getattr(prototype_network_parallel, 'module', prototype_network_parallel).prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                          proto_bound_j[2]:proto_bound_j[3], :]

            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(
                                                j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                             rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(
                                                    j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)

    if class_specific:
        del class_to_img_index_dict