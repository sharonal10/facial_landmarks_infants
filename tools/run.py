# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import numpy as np
import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import sys
import pandas as pd
import cv2  # For video processing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
from lib.core.evaluation import decode_preds
from collections import namedtuple

class InfantFacialLandmarksModel(nn.Module):
    def __init__(self):
        super().__init__()
        Args = namedtuple('Args', ['cfg', 'model_file'])

        # Usage
        base_dir = rf"/viscam/projects/infants/sharonal/infants-sharon/third_party/facial_landmarks_infants"
        args = Args(cfg=rf"{base_dir}/experiments/300w/hrnet-r90jt.yaml", model_file=rf"{base_dir}/infanface_pretrained/hrnet-r90jt.pth")

        update_config(config, args)

        logger, final_output_dir, tb_log_dir = \
            utils.create_logger(config, args.cfg, 'test')
        
        # print(config)
        logger.info(pprint.pformat(config))

        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.determinstic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        self.model = models.get_face_alignment_net(config)

        gpus = list(config.GPUS)
        self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()

        # load model
        state_dict = torch.load(args.model_file)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict)
        else:
            self.model.module.load_state_dict(state_dict)
        

    def forward(self, x):
        # x: (b, 3, h, w)
        # x = (x - self.pixel_mean[None, :, None, None]) / self.pixel_std[None, :, None, None]

        b, _, h, w = x.shape
        assert h == w == 256, (h, w)

        centers = torch.tensor([[128.0, 128.0]] * b)
        scales =  torch.tensor([1.28] * b)
        print('batch.shape', x.shape, centers.shape, scales.shape)
        output = self.model(x)
        score_map = output.data.cpu()
        preds = decode_preds(score_map, centers, scales, [64, 64]) # 32's are just half of the image size. maybe need resize image bigger (256)
        print('preds size:', preds.shape)


        output = preds.reshape(b, -1, 1, 1)  # we want shape b, 136, 1, 1
        assert False, (x.shape, output.shape)
        return output

        # # with torch.no_grad():
        # intermediate_output = self.model.get_intermediate_layers(x, self.n_last_blocks)
        # output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        # if self.avgpool:
        #     output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
        #     output = output.reshape(output.shape[0], -1)
        # output = output[:, :, None, None]  # (b, c, h=1, w=1)
        # return output


class VideoFrameDataset(Dataset):
    def __init__(self, npz_path):
        npz_data = np.load(npz_path)['frames']
        print(npz_data.shape)
        self.length = npz_data.shape[0]

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        self.data = (npz_data - self.mean) / self.std

        print('preprocessing done.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


def run_model_once():
    
    Args = namedtuple('Args', ['cfg', 'model_file'])

    # Usage
    args = Args(cfg=rf"experiments/300w/hrnet-r90jt.yaml", model_file="infanface_pretrained/hrnet-r90jt.pth")

    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')
    
    # print(config)
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    # dataset_type = get_dataset(config)

    # test_loader = DataLoader(
    #     dataset=dataset_type(config,
    #                          is_train=False),
    #     batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=config.PIN_MEMORY
    # )
    # npz_path = rf"/viscam/projects/infants/sharonal/infants-sharon/data/sharonal_ManyBabies/melanie/melanie_rgb_12_09_256/ManyBabies_melanie_2065_10.npz"
    npz_path = rf"/viscam/projects/infants/sharonal/google_video_rgb/ManyBabies_melanie_2065_10.npz"
    dataset = VideoFrameDataset(npz_path)
    dataloader = DataLoader(dataset, 
                             batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
                             shuffle=False,
                             num_workers=config.WORKERS,
                             pin_memory=config.PIN_MEMORY)
    
    flattened_outputs = []
    i = 0
    for batch in dataloader:
        centers = torch.tensor([[128.0, 128.0]] * config.TEST.BATCH_SIZE_PER_GPU*len(gpus))
        scales =  torch.tensor([1.28] * config.TEST.BATCH_SIZE_PER_GPU*len(gpus))
        print('batch.shape', batch.shape, centers.shape, scales.shape)
        print(f'processing batch {i} of {len(dataloader)}, batch size {config.TEST.BATCH_SIZE_PER_GPU*len(gpus)}')
        output = model(batch)
        score_map = output.data.cpu()
        preds = decode_preds(score_map, centers, scales, [64, 64]) # 32's are just half of the image size. maybe need resize image bigger (256)
        print('preds size:', preds.shape)
        i += 1


        flattened_output = preds.view(preds.shape[0], -1)  # Flatten to shape (8, 136)
        print('output shape', flattened_output.shape)
        flattened_outputs.append(flattened_output)

    final_output = torch.cat(flattened_outputs, dim=0)
    df = pd.DataFrame(final_output.numpy())

    # Save the DataFrame as a CSV file
    csv_file_path = 'facial_landmarks_output.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"Saved tensor to {csv_file_path}")
    print(final_output[0])
    print('------------')
    print(final_output[1])
    print(final_output.shape)

# run_model()


#     filenames, nme, predictions = function.inference(config, test_loader, model)
    
#     # Save coordinate predictions externally in an ad hoc manner.
#     output_dir = os.path.join(final_output_dir, 'predictions')
#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir) 
    
#     pred_array = predictions.numpy()
    
#     for i in range(len(filenames)):
#         with open(os.path.join(output_dir, os.path.splitext(filenames[i])[0] + '.txt'), 'w') as output_text:
#             output_text.write('file: ' + filenames[i] + '\n')
#             output_text.write('x: ' + str(list(pred_array[i, :, 0])) + '\n')
#             output_text.write('y: ' + str(list(pred_array[i, :, 1])) + '\n') 
#             output_text.write('min_box: ' + str([np.min(pred_array[i, :, 0]),
#                                                  np.min(pred_array[i, :, 1]),
#                                                  np.max(pred_array[i, :, 0]),
#                                                  np.max(pred_array[i, :, 1])]))
                                                 
#     print('Predictions saved to', output_dir)

#     torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


# if __name__ == '__main__':
#     main()

