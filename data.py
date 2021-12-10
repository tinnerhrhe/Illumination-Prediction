from os import listdir
from os.path import join
import numpy as np
import torch.utils.data as data
import pyexr
import torch
from util import is_exr, load_data
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
        gbuffer = current_frame.get("View Layer")
        gt = gbuffer[:, :, 0:3]  # return (r,g,b) 3 channel matrix
        depth_1 = gbuffer[:,:,4].reshape((576, 1024, 1))
        depth = np.concatenate((depth_1, depth_1, depth_1), axis=2)
        direct = gbuffer[:, :, 5:8]
        normal = gbuffer[:, :, 8:11]
        albedo_buffer = current_albedo.get("View Layer")
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
        '''
        albedos = torch.from_numpy(albedos).float()
        directs = torch.from_numpy(directs).float()
        normals = torch.from_numpy(normals).float()
        depths = torch.from_numpy(depths).float()
        gts = torch.from_numpy(gts).float()

        min = image.min()
        max = image.max()
        image = torch.FloatTensor(image.size()).copy_(image)
        image.add_(-min).mul_(1.0 / (max - min))
        image = image.mul_(2).add_(-1)
        '''
    '''
    print("gts------", gts.shape)
    print("albedos------", albedos.shape)
    print("directs------", directs.shape)
    print("normals------", normals.shape)
    print("depths------", depths.shape)
    '''
    return albedos, directs, normals, depths, gts
class DataLoaderHelper(data.Dataset):
    def __init__(self, image_dir, loader=frame_loader):
        super(DataLoaderHelper, self).__init__()
        '''
        self.albedo_path = join(image_dir, "albedo")
        self.depth_path = join(image_dir, "depth")
        self.direct_path = join(image_dir, "direct")
        self.normal_path = join(image_dir, "normal")
        self.gt_path = join(image_dir, "gt")
        '''
        self.loader = loader
        self.root = join(image_dir, "test")
        #print("-------------",image_dir)
        self.image_filenames = [x for x in listdir(self.root) if is_exr(x)]


    def __getitem__(self, index):
        '''
        albedo = load_image(join(self.albedo_path, self.image_filenames[index]))
        depth = load_image(join(self.depth_path, self.image_filenames[index]))
        direct = load_image(join(self.direct_path, self.image_filenames[index]))
        normal = load_image(join(self.normal_path, self.image_filenames[index]))
        gt = load_image(join(self.gt_path, self.image_filenames[index]))
        '''
        albedo, direct, normal, depth, gts = self.loader(join(self.root, self.image_filenames[index]))
        randScalar = torch.ones(3)
        randScalar[int(torch.randint(0, 3, (1,))[0])] = torch.rand(1)[0]
        randScalar = randScalar.view(1, 3, 1, 1)
        albedo = torch.from_numpy(albedo).float()
        direct = torch.from_numpy(direct).float() #* randScalar
        normal = torch.from_numpy(normal).float()
        depth = torch.from_numpy(depth).float()
        gts = torch.from_numpy(gts).float()  #* randScalar
        #flows = torch.from_numpy(flows).float()
        return load_data(albedo), load_data(direct), load_data(normal), load_data(depth), load_data(gts)

    def __len__(self):
        return len(self.image_filenames)
