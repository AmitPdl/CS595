import numpy as np
import torch
import os
import cv2
from models.GT import GT_Model
from models.GR import FeaturePyramidVGG
from models.encoder_build import encoder
from models.encoder_new import encoder as inpenc
from torch.utils.data.dataset import Dataset
#import data.new_dataset as datasets
# from data.new_dataset import paired_data_transforms
import torchvision.transforms as transforms
from models.vgg import Vgg19
from utils import index
import argparse
from collections import OrderedDict as odict
import torch




print(torch.cuda.is_available())

device = torch.device("cuda")


print("Device", device)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'





# Define a set of valid image file extensions for faster lookups
valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

def str2bool(v):
    return v.lower() in ('y', 'yes', 't', 'true', '1')
    
def resize_and_crop(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute the scaling factor and resize the image
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate the cropping coordinates
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped_image = resized_image[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped_image

parser = argparse.ArgumentParser('test')
parser.add_argument('--save_result_path',type=str2bool,default=True,help="if save result")

parser.add_argument('--sir_wild',type=str2bool,default=True,help="if sir_wild test")
parser.add_argument('--sir_solid',type=str2bool,default=True,help="if sir_solid test")
parser.add_argument('--sir_postcard',type=str2bool,default=True,help="if sir_postcard test")

parser.add_argument('--nature',type=str2bool,default=True,help="if real45 test")
parser.add_argument('--newdata',type=str2bool,default=True,help="if real45 test")

parser.add_argument('--real20',type=str2bool,default=True,help="if real20 test")
parser.add_argument('--num_workers',type=int,default=1,help="num_workers")
parser.add_argument('--batch_size',default=1,type=int,help="batch size")
args = parser.parse_args()
    
#dataset without GT
path_real45 = './testsets/real45/'

#datasets with GT

# The following datasets are not provided due to their policy
# You can apply for the SIR^2 dataset from https://sir2data.github.io/
path_sir_wild = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/testsets/SIR2/WildSceneDataset/'
path_sir_postcard = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/testsets/SIR2/PostcardDataset/' 
path_sir_solid = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/testsets/SIR2/SolidObjectDataset/'
path_nature = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/testsets/Nature/'
path_newdata = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/testsets/newdata/'
path_real20 = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/testsets/real20/'


encoder_I = encoder()
encoder_R = encoder()
inpenc = inpenc()

encoder_I.cuda()
encoder_R.cuda()
inpenc.cuda()

gt_model = GT_Model(encoder_I,encoder_R, inpenc)
gt_model.cuda()
gt_model.eval()


gr_model = FeaturePyramidVGG(out_channels=64)
gr_model.cuda()
gr_model.eval()

vgg = Vgg19(requires_grad=False).cuda()

def creat_list(path,if_gt=True):
    gt_list = []
    image_list = []
    
    blended_path = path + 'blended/'
                   
    if if_gt:

        trans_path = path + 'transmission_layer/'

            
        for _,_,fnames in sorted(os.walk(blended_path)):
            # Iterate through filenames
            for fname in fnames:

                # Get the file extension
                ext = os.path.splitext(fname)[1].lower()  # Ensure the extension is in lowercase
            
                # Check if the file has a valid image extension
                if ext in valid_extensions:
                    blended_file = os.path.join(blended_path, fname)
                    trans_file = os.path.join(trans_path, fname)
            
                    image_list.append(blended_file)
                    gt_list.append(trans_file)
                else:
                    continue
                
    else:
        for _,_,fnames in sorted(os.walk(blended_path)):
            for fname in fnames:

                image_list.append(path+fname)
    
    print(image_list)           
    return image_list,gt_list


class TestDataset(Dataset):
    def __init__(self,blended_list,trans_list,transform=True,if_GT=True):
        self.to_tensor = transforms.ToTensor()            
        self.blended_list = blended_list
        self.trans_list = trans_list
        self.transform = transform
        self.if_GT = if_GT
        

    def __getitem__(self, index):
        blended = cv2.imread(self.blended_list[index])
        trans = blended
        if self.if_GT:
            trans= cv2.imread(self.trans_list[index])
                    
        if self.transform == True:
            
            blended = resize_and_crop(blended)
            trans = resize_and_crop(trans)
            
            # if trans.shape[0] > trans.shape[1]:
            #     neww = 224
            #     newh = round((neww / trans.shape[1]) * trans.shape[0])
            # if trans.shape[0] < trans.shape[1]:
            #     newh = 224
            #     neww = round((newh / trans.shape[0]) * trans.shape[1])
            # blended = cv2.resize(np.float32(blended), (neww, newh), cv2.INTER_CUBIC)/255.0
            # trans = cv2.resize(np.float32(trans), (neww, newh), cv2.INTER_CUBIC)/255.0
        
        blended = self.to_tensor(blended)
        trans = self.to_tensor(trans)

        return blended,trans
    
    def __len__(self):
        return len(self.blended_list)

def test_diffdataset(test_loader,save_path=None,if_GT=True):
    ssim_sum = 0
    psnr_sum = 0
    count = 0
    
    for j, (image, gt) in enumerate(test_loader):
        count += 1
        
        image = image.to(device)
        gt = gt.to(device)
        with torch.no_grad():   
            image.requires_grad_(False)

            pretrained_R, pretrained_T = gr_model(image, vgg(image))
            
            output_t, r_final= gt_model(pretrained_T, pretrained_R, image)

            output_t = index.tensor2im(output_t)
            gt = index.tensor2im(gt)
            r_final = index.tensor2im(r_final)
            
            if if_GT:
                res, psnr, ssim = index.quality_assess(output_t, gt)
                ssim_sum += ssim
                psnr_sum += psnr 
                
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)  
                # if not os.path.exists(save_path):
                #     os.mkdir(save_path) 
                image = index.tensor2im(image)
                cv2.imwrite("%s/%s_pretr.png"%(save_path,j),r_final,[int(cv2.IMWRITE_JPEG_QUALITY),100])   
                cv2.imwrite("%s/%s_b.png"%(save_path,j),image,[int(cv2.IMWRITE_JPEG_QUALITY),100])  
                cv2.imwrite("%s/%s_t.png"%(save_path,j),output_t,[int(cv2.IMWRITE_JPEG_QUALITY),100])     
                if if_GT:
                    cv2.imwrite("%s/%s_gt.png"%(save_path,j),gt,[int(cv2.IMWRITE_JPEG_QUALITY),100])     
                
    print("Length of dataset:", count)
    print("Lenght of dataset:", len(test_loader),'SSIM:',ssim_sum/len(test_loader),'PSNR:',psnr_sum/len(test_loader))
    return (len(test_loader),ssim_sum,psnr_sum)


def test_state(state_T, state_R, epoch, results_psnr, results_ssim):
    print(device)
  
    gt_model.load_state_dict(state_T)
    gr_model.load_state_dict(state_R)
    del(state_T)
    del(state_R)
    print("XX")

    datasets = odict([('sir_wild', True), ('newdata', True), ('real20', True), ('nature', True)])# ('sir_solid', True), ('sir_postcard', True)])#, ])
    
    
    psnr_all, ssim_all, num_all= 0, 0, 0
    psnr, ssim = [], []
    psnr.append(epoch)
    ssim.append(epoch)

    print("xx")
    for dataset, with_GT in datasets.items():
        
        
        
        if getattr(args, dataset):
            if args.save_result_path:
                
                save_path = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/resultF/trials/' + dataset
            else:
                save_path = None
            print('testing dataset:',dataset)
            num, ssim_sum, psnr_sum = test_diffdataset(eval('test_loader_'+dataset.replace('_','')), save_path, with_GT)
            psnr.append(psnr_sum/num) 
            ssim.append(psnr_sum/num) 
            
            
            if with_GT:
                psnr_all += psnr_sum
                ssim_all += ssim_sum
                num_all += num

    ssim_av = ssim_all/num_all
    psnr_av = psnr_all/num_all
    psnr.append(psnr_av)
    ssim.append(ssim_av)
    
    results_psnr.append(psnr)
    results_ssim.append(ssim)
    
    return psnr_av,ssim_av,results_psnr,results_ssim



image_list_sirwild, gt_list_sirwild = creat_list(path_sir_wild)
test_dataset_sirwild = TestDataset(image_list_sirwild,gt_list_sirwild)
test_loader_sirwild = torch.utils.data.DataLoader(dataset=test_dataset_sirwild,\
                                                  batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

image_list_newdata, gt_list_newdata = creat_list(path_newdata)
test_newdata = TestDataset(image_list_newdata,gt_list_newdata)
test_loader_newdata = torch.utils.data.DataLoader(dataset=test_newdata,\
                                                  batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

image_list_nature, gt_list_nature = creat_list(path_nature)
test_dataset_nature = TestDataset(image_list_nature,gt_list_nature)
test_loader_nature = torch.utils.data.DataLoader(dataset=test_dataset_nature,\
                                                  batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)


image_list_real20, gt_list_real20 = creat_list(path_real20)
test_dataset_real20 = TestDataset(image_list_real20,gt_list_real20)
test_loader_real20 = torch.utils.data.DataLoader(dataset=test_dataset_real20,\
                                                  batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)                                                

# image_list_sirpostcard, gt_list_sirpostcard = creat_list(path_sir_postcard)
# test_dataset_sirpostcard = TestDataset(image_list_sirpostcard,gt_list_sirpostcard)
# test_loader_sirpostcard = torch.utils.data.DataLoader(dataset=test_dataset_sirpostcard,\
#                                                       batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

# image_list_sirsolid, gt_list_sirsolid = creat_list(path_sir_solid)
# test_dataset_sirsolid = TestDataset(image_list_sirsolid,gt_list_sirsolid)
# test_loader_sirsolid = torch.utils.data.DataLoader(dataset=test_dataset_sirsolid,\
#                                                    batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

# image_list_real45, gt_list_real45 = creat_list(path_real45,if_gt=False)
# test_dataset_real45 = TestDataset(image_list_real45,gt_list_real45,if_GT=False)
# test_loader_real45 = torch.utils.data.DataLoader(dataset=test_dataset_real45,\
#                                                  batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

if __name__ == '__main__':
    
    ckpt_path  = './checkpoint/pretrain.pth'
    ckpt_pre = torch.load(ckpt_path)
    print("loading checkpoint'{}'".format(ckpt_path))
    
    psnr_av,ssim_av = test_state(ckpt_pre['GT_state'],ckpt_pre['GR_state'])    
    print('The average PSNR/SSIM of all chosen testsets:',psnr_av,ssim_av) 
