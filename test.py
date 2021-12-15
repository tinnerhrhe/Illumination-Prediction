import argparse
import os
from os.path import join
import pyexr
import torch
from torch.autograd import Variable
import numpy as np
from model import G
import pytorch_ssim
from util import is_exr, load_data, save_image
ssim_loss = pytorch_ssim.SSIM()
l1loss = torch.nn.L1Loss()
def frame_loader(path):
    normals = []
    depths = []
    directs = []
    albedos = []
    gts = []
    #path = join(path, "path")
    path = [path]
    for image in path:
        #print("---------------",path)
        image_gbuffer = join(image, "gbuffer.exr")
        image_albedo = join(image, "albedo.exr")
        current_frame = pyexr.open(image_gbuffer)
        current_albedo = pyexr.open(image_albedo)
        #current_frame.describe_channels()
        #current_albedo.describe_channels()
        key = list(current_frame.root_channels)[0]
        gbuffer = current_frame.get(key)
        gt = gbuffer[:, :, 0:3]  # return (r,g,b) 3 channel matrix
        depth_1 = gbuffer[:,:,4].reshape((576, 1024, 1))
        depth = np.concatenate((depth_1, depth_1, depth_1), axis=2)
        direct = gbuffer[:, :, 5:8]
        normal = gbuffer[:, :, -3:]
        albedo_buffer = current_albedo.get("ViewLayer")
        albedo = albedo_buffer[:, :, 0:3]
        gt = np.transpose(gt, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        gts.append(gt)
        gts = np.concatenate(gts, axis=0)
        albedo = np.transpose(albedo, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        albedos.append(albedo)
        albedos = np.concatenate(albedos, axis=0)
        direct = np.transpose(direct, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        directs.append(direct)
        directs = np.concatenate(directs, axis=0)
        normal = np.transpose(normal, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        normals.append(normal)
        normals = np.concatenate(normals, axis=0)
        depth = np.transpose(depth, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        depths.append(depth)
        depths = np.concatenate(depths, axis=0)

    return albedos, directs, normals, depths, gts
parser = argparse.ArgumentParser(description='DeepRendering-implementation')
parser.add_argument('--dataset', required=False, help='unity')
parser.add_argument('--model', type=str, required=True, help='model file')
parser.add_argument('--n_channel_input', type=int, default=3, help='input channel')
parser.add_argument('--n_channel_output', type=int, default=3, help='output channel')
parser.add_argument('--n_generator_filters', type=int, default=64, help="number of generator filters")
opt = parser.parse_args()

netG_model = torch.load(opt.model)
netG = G(opt.n_channel_input * 4, opt.n_channel_output, opt.n_generator_filters)
netG.load_state_dict(netG_model['state_dict_G'])

root_dir = 'dataset/test/'
#image_dir = 'dataset/{}/test/albedo'.format(opt.dataset)
image_filenames = [x for x in os.listdir(root_dir) if is_exr(x)]

for image_name in image_filenames:
    albedo, direct, normal, depth, gts = frame_loader(root_dir + image_name)
    albedo = load_data(torch.from_numpy(albedo).float())
    direct = load_data(torch.from_numpy(direct).float())  # * randScalar
    normal = load_data(torch.from_numpy(normal).float())
    depth = load_data(torch.from_numpy(depth).float())
    gts = load_data(torch.from_numpy(gts).float())  # * randScalar

    albedo = Variable(albedo).view(1, -1, 576, 1024).cuda()
    direct = Variable(direct).view(1, -1, 576, 1024).cuda()
    normal = Variable(normal).view(1, -1, 576, 1024).cuda()
    depth = Variable(depth).view(1, -1, 576, 1024).cuda()
    gts = Variable(gts).view(1, -1, 576, 1024).cuda()
    netG = netG.cuda()


    out = netG(torch.cat((albedo, direct, normal, depth), 1))
    shading_loss = 0
    l1_shading_loss = 0
    laplacian_loss = 0
    # laplacian_loss += l1loss(laplacian_warp(albedo_var[:, frame, :, :, :].cuda() * output_var[:, frame, :, :, :]),
    #                        laplacian_warp(target[:, frame, :, :, :].cuda()))
    l1_shading_loss += l1loss(out[:, :, :, :], gts[:, :, :, :].cuda())
    shading_loss += ssim_loss(out[:, :, :, :], gts[:, :, :, :].cuda())
    print('=> shading_loss: {:.4f} L1_loss: {:.4f}'.format(
        shading_loss,
        l1_shading_loss
    ))
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", "Final")):
        os.mkdir(os.path.join("result", "Final"))
    save_image(out_img, "result/Final/{}.png".format(image_name))
