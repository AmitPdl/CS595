import os
import torch
import torchvision.models as models
from os.path import join
import torch.nn.functional as F
from models.GT import Discriminator,GT_Model
from models.GR import FeaturePyramidVGG
from models.vgg import Vgg19
from models.encoder_build import encoder
from models.encoder_new import encoder as inpenc
from data.dataset import MyDataset,FusionDataset
from data.datasets_pairs import my_dataset_wTxt,FusionDataset
from utils.utils import is_image_file
from test import test_state
import cv2
import data.new_dataset as datasets
import numpy as np
import csv



# Set up main device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns

EPS=1e-12


class Solver:
    def __init__(self,args):
        self.args = args
        self.start_epoch=0
        self.global_step = 0
        self.is_blur=0
        
        self.encoder_I = encoder()
        self.encoder_I.cuda()
        self.encoder_R = encoder()
        self.encoder_R.cuda()
        self.inpenc = inpenc()
        self.inpenc.cuda()
        
        self.vgg_feature = models.vgg19(pretrained=True)
        # vgg_feature_state_dict = torch.load(self.args.vgg_feature_pretrained_ckpt)
        # self.vgg_feature.load_state_dict(vgg_feature_state_dict)
        # del(vgg_feature_state_dict)
        self.vgg_feature.cuda()
        
        self.encoder = encoder()
        self.encoder.cuda()


    def prepare_data(self,train_path,is_ref_syn):
        blended=[]
        transmission=[]
        reflection=[]
        if is_ref_syn:
            print('loading synthetic data...')
        else:
            print('loading real data...')
            
        for dirname in train_path:
            train_t_gt= dirname+"/transmission_layer/"
            train_r_gt=dirname+"/reflection_layer/"
            train_b=dirname+"/blended/"
            if is_ref_syn:
                r_list=os.listdir(train_r_gt)
            for _,_,fnames in sorted(os.walk(train_t_gt)):
                for fname in fnames:
                    if is_ref_syn:
                        if not fname in r_list:
                            continue
                    if is_image_file(fname):
                        path_transmission=os.path.join(train_t_gt,fname)
                        transmission_img = cv2.imread(path_transmission)
                        path_blended=os.path.join(train_b,fname)
                        blended_img = cv2.imread(path_blended)
                        # path_reflection=os.path.join(train_r_gt,fname)
                        reflection_img = t_img.copy()

                        blended.append(blended_img)
                        transmission.append(transmission_img)
                        reflection.append(reflection_img)

        return blended,transmission,reflection
    
    def feat_loss(self,output_t,gt_t):
        feat_loss = 0
        conv_list = [2, 7, 12, 21, 30]
        sigma_list = [1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5]
        i = 0
        for index in range(31):
            torch.cuda.empty_cache()
            
            output_t = self.vgg_feature.features[index](output_t)
            gt_t = self.vgg_feature.features[index](gt_t)

            if index in conv_list:
                feat_loss+=self.l1_loss(output_t,gt_t)*sigma_list[i]
                i += 1
        return feat_loss
    
    def exclusion_loss(self,img_T,img_R,level=3): 
        grad_x_loss = []
        grad_y_loss = []
        for l in range(level):
            grad_x_T,grad_y_T = self.compute_grad(img_T)
            grad_x_R,grad_y_R = self.compute_grad(img_R)

            alphax = 2.0 * torch.mean(torch.abs(grad_x_T)) / torch.mean(torch.abs(grad_x_R))
            alphay = 2.0 * torch.mean(torch.abs(grad_y_T)) / torch.mean(torch.abs(grad_y_R))
        
            gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
            gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1
            
            grad_x_loss.append(torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2))) ** 0.25)
            grad_y_loss.append(torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2))) ** 0.25)
            
            img_T=F.interpolate(img_T,scale_factor=0.5,mode='bilinear')
            img_R=F.interpolate(img_R,scale_factor=0.5,mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss)/3) + torch.sum(sum(grad_y_loss)/3)
        
        return loss_gradxy/2
    
    def l1_loss(self,input,output):
        return torch.mean(torch.abs(input - output))
    
    def convert_L(self, img):
        img_L = (0.114*img[0,0,:,:]+0.587*img[0,1,:,:]+\
                 0.299*img[0,2,:,:]).unsqueeze(0).unsqueeze(0)
        return img_L
    
    def compute_grad(self,img):
        gradx = img[:,:,1:,:]-img[:,:,:-1,:]
        grady = img[:,:,:,1:]-img[:,:,:,:-1]
        return gradx,grady
    
    def thr_loss(self, list_map, list_R, list_mask, list_mask_encoder, list_map_encoder):
        mask_loss = 0
        mask_encoder_loss = 0
        for i in range(4):
            mask_loss = mask_loss + self.l1_loss(list_map[i]*list_mask[i], list_map[i]*1)
            mask_encoder_loss = mask_encoder_loss + \
                self.l1_loss(list_map_encoder[i]*list_mask_encoder[i], list_map_encoder[i]*0)
        return mask_loss + mask_encoder_loss 
    
    


    def train_model(self):
    
        self.gr_model = FeaturePyramidVGG(out_channels=64)
        self.gr_model.cuda()
        
        self.gt_model = GT_Model(self.encoder_I,self.encoder_R,self.inpenc)
        self.gt_model.cuda()
        
        self.discriminator = Discriminator()
        self.discriminator.cuda()

        param = list(self.gt_model.parameters()) + list(self.gr_model.parameters())
        total_params1 = sum(p.numel() for p in param)

        self.G_opt = torch.optim.Adam(param,lr=self.args.lr,)
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(),lr=self.args.lr,)

        total_params2 = sum(p.numel() for p in self.discriminator.parameters())
        print(f'Total number of parameters: {total_params1+total_params2}')


        #resume from a checkpoit
        if self.args.resume_file:
            print("X1")
            if os.path.isfile(self.args.resume_file):
                print("loading checkpoint'{}'".format(self.args.resume_file))
                checkpoint = torch.load(self.args.resume_file)
                self.start_epoch = checkpoint['epoch']
                self.global_step = checkpoint['global_step']
                self.gt_model.load_state_dict(checkpoint['GT_state'])
                self.gr_model.load_state_dict(checkpoint['GR_state'])
                del(checkpoint)
                print("'{}' loaded".format(self.args.resume_file,self.args.start_epoch))
            else:
                print("no checkpoint found at '{}'".format(self.args.resume_file))
                return 1

        torch.backends.cudnn.benchmark = True
        datadir = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD'
        
        

        train_cvpr_dataset = my_dataset_wTxt(datadir, '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/image_pairs.txt',
                                            crop_size=224,
                                            # fix_sample_A=10000,
                                            regular_aug=False) 

        
        datadir_syn = join(datadir, 'train/VOCdevkit/VOC2012/PNGImages')
        datadir_real = join(datadir, 'train/real')
        datadir_nature = join(datadir, 'train/nature')
        datadir_newdata = join(datadir, 'train/newdata')

        train_dataset_syn = datasets.DSRDataset(
            datadir_syn, read_fns('/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/data/VOC2012_224_train_png.txt'), enable_transforms=True)       #, size = 150)

        train_dataset_real = datasets.DSRTestDataset(datadir_real, enable_transforms=True, if_align=True)#, size = self.args.size)
        train_dataset_nature = datasets.DSRTestDataset(datadir_nature, enable_transforms=True, if_align=True)#, size = self.args.size)
        train_dataset_newdata = datasets.DSRTestDataset(datadir_newdata, enable_transforms=True, if_align=True)#, size = self.args.size)
        

        
        from torch.utils.data import ConcatDataset
        
        real = ConcatDataset([train_dataset_nature, train_dataset_real, train_dataset_newdata, train_cvpr_dataset])
        

        train_dataset_fusion = datasets.FusionDataset([train_dataset_syn,
                                                    real])
                                                    
                                                    
                                                   

        #train_dataloader_fusion = datasets.DataLoader(
        #    train_dataset_fusion, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        # input_real_b,output_real_t,output_real_r=self.prepare_data([self.args.ref_real_dir],is_ref_syn=False)
        # input_syn_b,output_syn_t,output_syn_r=self.prepare_data([self.args.ref_syn_dir],is_ref_syn=True)
     
        # train_refreal_dataset = MyDataset(input_real_b,output_real_t,output_real_r,is_ref_syn=False)
        # train_refsyn_dataset = MyDataset(input_syn_b,output_syn_t,output_syn_r,is_ref_syn=True)

        # train_fusion_dataset=FusionDataset([train_refsyn_dataset,train_refreal_dataset],[0.7,0.3])

        train_dataloader_fusion = torch.utils.data.DataLoader(train_dataset_fusion, 
                 batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.load_workers)

        # train the model
        best_psnr = best_ssim = 0
        
        results_psnr, results_ssim = [], []
        results_psnr.append(["Epoch","Wild","Solid","Postcard","Average"])
        results_ssim.append(["Epoch","Wild","Solid","Postcard","Average"])


        print("XX") 
        for epoch in range(self.args.num_epochs):

            # G_loss_avg,D_loss_avg = self.train_epoch(train_dataloader_fusion,epoch)

            if epoch>=0:          #% self.args.save_model_freq == 0 and epoch !=0 :
                state = {
                    'epoch': epoch + 1,
                    'global_step': self.global_step,
                    'GT_state': self.gt_model.state_dict(),
                    'GR_state': self.gr_model.state_dict(),
                }
                
                psnr,ssim,results_psnr,results_ssim = test_state(state['GT_state'],state['GR_state'],epoch, results_psnr, results_ssim)
                print("######################### psnr, ssim:", psnr, ssim)

                G_loss_avg,D_loss_avg = self.train_epoch(train_dataloader_fusion,epoch)
                
                if not os.path.exists('./checkpoint'):
                    os.makedirs('./checkpoint')
                
                if psnr > best_psnr or ssim > best_ssim:
                    print('Saving checkpoint, G_loss: {} D_loss: {}'.format(G_loss_avg, D_loss_avg))
                    if psnr > best_psnr:
                        best_psnr = psnr
                        torch.save(state, './checkpoint/epoch_{:0>3}_G_{:.3f}_P_{:.3f}.pth'.format(epoch,G_loss_avg,psnr))
                    else:
                        best_ssim = ssim
                        torch.save(state, './checkpoint/epoch_{:0>3}_G_{:.3f}_P_{:.3f}.pth'.format(epoch,G_loss_avg,psnr))
        
        csv_file_path_psnr = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/DEC4/psnr_results.csv'
        csv_file_path_ssim = '/scratch/apaudya/Summer/RAGNet/RAGNet-master_OLD/DEC4/ssim_results.csv'
        
        # Ensure the directories exist
        os.makedirs(os.path.dirname(csv_file_path_psnr), exist_ok=True)
        os.makedirs(os.path.dirname(csv_file_path_ssim), exist_ok=True)

        # Create and write the 2D list into a new CSV file
        with open(csv_file_path_psnr, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write each row from the 2D list into the CSV
            for row in results_psnr:
                writer.writerow(row)
                
        with open(csv_file_path_ssim, mode='w', newline='') as file1:
            writer1 = csv.writer(file1)
            
            # Write each row from the 2D list into the CSV
            for roww in results_ssim:
                writer1.writerow(roww)
        
            
    def train_epoch(self,train_dataloader_fusion,epoch):
        train_dataloader_fusion.dataset.reset()
        self.vgg = Vgg19(requires_grad=False).cuda()

        self.gt_model.train()
        self.discriminator.train()
        self.gr_model.train()                             
        
        G_loss_sum=0
        D_loss_sum=0

        self.steps_per_epoch = len(train_dataloader_fusion)
        
        for index, (input_b,gt_t,gt_r, is_ref_syn) in enumerate(train_dataloader_fusion):
            input_b = input_b.cuda(non_blocking=True)
            gt_t = gt_t.cuda(non_blocking=True)
            gt_r = gt_r.cuda(non_blocking=True)
            
            gray_r = self.convert_L(gt_r)
            map_thr = torch.zeros(gray_r.shape)
            map_thr[0,0,gray_r[0,0,:,:]<0.01] = 1
            map_thr = map_thr.cuda()
            
            map_encoder = torch.zeros(gray_r.shape)
            map_encoder[0,0,gray_r[0,0,:,:]>0.3] = 1
            map_encoder = map_encoder.cuda()
           
            if index%2==0:
                self.D_opt.zero_grad()
                with torch.no_grad():
                    pretrained_R, pretrained_T = self.gr_model(input_b, self.vgg(input_b))

                    output_t, r_final, rr ,list_map, list_R, list_mask, list_mask_encoder, list_map_encoder = self.gt_model(pretrained_T, pretrained_R, input_b, map_thr, map_encoder)
                    
                score_fake = self.discriminator(input_b, output_t)
                score_real = self.discriminator(input_b, gt_t)
                D_loss = (torch.mean(-(torch.log(score_real + EPS) + torch.log(1-score_fake+EPS))))*0.5
                D_loss_sum += D_loss.item()
                D_loss.cuda()
                D_loss.backward()
                self.D_opt.step()
                if index % self.args.print_freq == 0:
                    print('D_loss: {0}\tstep: {1}'.format(D_loss,index))
            
            pretrained_R, pretrained_T = self.gr_model(input_b, self.vgg(input_b))
            output_t, r_final, rr, list_map, list_R, list_mask, list_mask_encoder, list_map_encoder = self.gt_model(pretrained_T, pretrained_R, input_b, map_thr, map_encoder)
            
            score_fake = self.discriminator(input_b, output_t)
            self.G_opt.zero_grad()
            
            F_loss = self.feat_loss(output_t, gt_t)
            A_loss = torch.mean(-torch.log(score_fake + EPS))
            E_loss= self.exclusion_loss(output_t, r_final, level=3)
            LRM_loss = self.l1_loss(rr+output_t+r_final,input_b)
        
            if is_ref_syn:
                Thr_loss = self.thr_loss(list_map, list_R, list_mask, list_mask_encoder, list_map_encoder)
                F_loss = F_loss + self.feat_loss(r_final,gt_r)
                R_loss = self.l1_loss(r_final, gt_r)
                
                G_loss = 0.2 * F_loss + E_loss + R_loss + 0.01*A_loss +Thr_loss + 0.2*LRM_loss
                if index % self.args.print_freq == 0:
                    print('A_loss: {0}\tF_loss: {1}\tE_loss: {2}\tthr_loss: {3}\tepoch: {4}'.format(0.01*A_loss,
                                                                            0.2*F_loss, E_loss,Thr_loss,epoch))
                            
            else: 
                G_loss = 0.2 * F_loss + 0.01*A_loss + E_loss + 0.2*LRM_loss
                if index % self.args.print_freq == 0:
                    print('A_loss: {0}\tF_loss: {1}\tE_loss: {2}\tepoch: {3}'.format(0.01*A_loss,
                                                                                     0.2*F_loss,E_loss,epoch))  
                
            G_loss_sum += G_loss.item()
            G_loss = G_loss.cuda()
            torch.cuda.empty_cache()
            G_loss.backward()
            self.G_opt.step()
            torch.cuda.empty_cache()
            
        self.global_step+=1

        return G_loss_sum/self.steps_per_epoch, D_loss_sum/(self.steps_per_epoch/2)