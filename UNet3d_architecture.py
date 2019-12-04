import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
from time import sleep
use_gpu = torch.cuda.is_available()


class UNet_n_base(nn.Module) :

    def norm_lrelu_conv(self, feat_in, feat_out, kernel=3, stride =1, padding=1):  # 'residual block'
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=kernel, stride=stride, padding=padding, bias=False))

    def conv_norml_lrelu(self, feat_in, feat_out, kernel=3, stride =1, padding=1):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def lrelu_conv(self, feat_in, feat_out, kernel=3, stride =1, padding=1):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=kernel, stride=stride, padding=padding, bias=False))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out, kernel=3, stride =1, padding=1):
            return nn.Sequential(
                nn.Upsample(scale_factor= 2, mode='nearest'),
                nn.Conv3d(feat_in, feat_out, kernel_size=kernel, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(feat_in),
                nn.LeakyReLU())
        
    
    
    def __init__(self, in_channels, class_number, n_base_filter):   
        super(UNet_n_base, self).__init__()
        
        """
        n_base_filter = 21 in the Lancet Onc paper.
        
        
        """


        ######## level 1 context pathway : 128x128x128
        self.context1_1 = nn.Conv3d(in_channels, out_channels=1*n_base_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.context1_2_1 = self.norm_lrelu_conv(feat_in=1*n_base_filter, feat_out=1*n_base_filter, kernel=3, stride=1, padding=1)
        self.context1_dropout = nn.Dropout3d(p=0.3)
        self.context1_2_2 = self.norm_lrelu_conv(feat_in=1*n_base_filter, feat_out=1*n_base_filter, kernel=3, stride=1, padding=1)
            #Elementwise_sum  
        self.context1_2_norm = nn.InstanceNorm3d(1*n_base_filter)
        self.context1_2_lrelu = nn.LeakyReLU()
        
        ######## level 2 context pathway : 64x64x64
        self.context2_1 = nn.Conv3d(in_channels=1*n_base_filter, out_channels=2**1*n_base_filter, kernel_size=3, stride=2, padding=1, bias=False)
        self.context2_2_1 = self.norm_lrelu_conv(feat_in=2**1*n_base_filter, feat_out=2**1*n_base_filter, kernel=3, stride=1, padding=1)
        self.context2_dropout = nn.Dropout3d(p=0.3)
        self.context2_2_2 = self.norm_lrelu_conv(feat_in=2**1*n_base_filter, feat_out=2**1*n_base_filter, kernel=3, stride=1, padding=1)
            #Elementwise_sum
        self.context2_2_norm = nn.InstanceNorm3d(2**1*n_base_filter)
        self.context2_2_lrelu = nn.LeakyReLU()
        
        ######## level 3 context pathway : 32x32x32
        self.context3_1 = nn.Conv3d(in_channels=2**1*n_base_filter, out_channels=2**2*n_base_filter, kernel_size=3, stride=2, padding=1, bias=False)
        self.context3_2_1 = self.norm_lrelu_conv(feat_in=2**2*n_base_filter, feat_out=2**2*n_base_filter, kernel=3, stride=1, padding=1)
        self.context3_dropout = nn.Dropout3d(p=0.3)
        self.context3_2_2 = self.norm_lrelu_conv(feat_in=2**2*n_base_filter, feat_out=2**2*n_base_filter, kernel=3, stride=1, padding=1)
            #Elementwise_sum
        self.context3_2_norm = nn.InstanceNorm3d(2**2*n_base_filter)
        self.context3_2_lrelu = nn.LeakyReLU()
        
        ######## level 4 context pathway :16x16x16
        self.context4_1  = nn.Conv3d(in_channels=2**2*n_base_filter, out_channels=2**3*n_base_filter, kernel_size=3, stride=2, padding=1, bias=False)
        self.context4_2_1 = self.norm_lrelu_conv(feat_in=2**3*n_base_filter, feat_out=2**3*n_base_filter, kernel=3, stride=1, padding=1)
        self.context4_dropout = nn.Dropout3d(p=0.3)
        self.context4_2_2 = self.norm_lrelu_conv(feat_in=2**3*n_base_filter, feat_out=2**3*n_base_filter, kernel=3, stride=1, padding=1)
            #Elementwise_sum
        self.context4_2_norm = nn.InstanceNorm3d(2**3*n_base_filter)
        self.context4_2_lrelu = nn.LeakyReLU()
        
        ######## level 5 context pathway: 8x8x8
        self.context5_1  = nn.Conv3d(in_channels=2**3*n_base_filter, out_channels=2**4*n_base_filter, kernel_size=3, stride=2, padding=1, bias=False)
        self.context5_2_1 = self.norm_lrelu_conv(feat_in=2**4*n_base_filter, feat_out=2**4*n_base_filter, kernel=3, stride=1, padding=1)
        self.context5_dropout = nn.Dropout3d(p=0.3)
        self.context5_2_2 = self.norm_lrelu_conv(feat_in=2**4*n_base_filter, feat_out=2**4*n_base_filter, kernel=3, stride=1, padding=1)
            #Elementwise_sum
        self.context5_2_norm = nn.InstanceNorm3d(2**4*n_base_filter)
        self.context5_2_lrelu = nn.LeakyReLU()
        

        ####### level 5 upsampling
        self.upsample5 = self.upscale_conv_norm_lrelu(feat_in=2**4*n_base_filter, feat_out=2**3*n_base_filter, kernel=3, stride =1, padding=1)

        ##### level 4 concat + localization + upsampling
            ## concat
        self.local4_1 = self.conv_norml_lrelu(feat_in=2**4*n_base_filter, feat_out=2**4*n_base_filter, kernel=3, stride=1, padding=1)
        self.local4_2 = self.conv_norml_lrelu(feat_in=2**4*n_base_filter, feat_out=2**3*n_base_filter, kernel=1, stride=1, padding=0)
        self.upsample4 = self.upscale_conv_norm_lrelu(feat_in=2**3*n_base_filter, feat_out=2**2*n_base_filter, kernel=3, stride =1, padding=1)

        ##### level 3 concat + localization + upsampling
            ## concat
        self.local3_1 = self.conv_norml_lrelu(feat_in=2**3*n_base_filter, feat_out=2**3*n_base_filter, kernel=3, stride=1, padding=1)
             #segment3 pulled out
        self.local3_2 = self.conv_norml_lrelu(feat_in=2**3*n_base_filter, feat_out=2**2*n_base_filter, kernel=1, stride=1, padding=0)
        self.upsample3 = self.upscale_conv_norm_lrelu(feat_in=2**2*n_base_filter, feat_out=2**1*n_base_filter, kernel=3, stride =1, padding=1)
        
        ##### level 2 concat + localization + upsampling
            ## concat
        self.local2_1 = self.conv_norml_lrelu(feat_in=2**2*n_base_filter, feat_out=2**2*n_base_filter, kernel=3, stride=1, padding=1)
             #segment2 pulled out
        self.local2_2 = self.conv_norml_lrelu(feat_in=2**2*n_base_filter, feat_out=2**1*n_base_filter, kernel=1, stride=1, padding=0)
        self.upsample2 = self.upscale_conv_norm_lrelu(feat_in=2**1*n_base_filter, feat_out=2**0*n_base_filter, kernel=3, stride =1, padding=1)
        
        ##### level 1 concat + localization + upsampling
            ## concat
        self.local1 = self.conv_norml_lrelu(feat_in=2**1*n_base_filter, feat_out=2**1*n_base_filter, kernel=3, stride=1, padding=1)
            #segment 1 pulled out

        
        
        
        
        #### segmentation layer 
        self.seg3 = nn.Conv3d(in_channels=2**3*n_base_filter, out_channels=class_number, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg2 = nn.Conv3d(in_channels=2**2*n_base_filter, out_channels=class_number, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg1 = nn.Conv3d(in_channels=2**1*n_base_filter, out_channels=class_number, kernel_size=1, stride=1, padding=0, bias=True)
               
        
    def forward(self, x):

        ######## level 1 context pathway : 128x128x128
        
        #print("context 1")
        
        out_context1_1 = self.context1_1(x)
        residual1 = out_context1_1
        out_context1_2_1 = self.context1_2_1(out_context1_1)
        out_context1_dropout = self.context1_dropout(out_context1_2_1)
        out_context1_2_2 = self.context1_2_2(out_context1_dropout)
        #Elementwise summation
        out_context1_2_2 += residual1
        out_context1_2_norm = self.context1_2_norm(out_context1_2_2)
        out_context1_2_lrelu = self.context1_2_lrelu(out_context1_2_norm)
        context1 = out_context1_2_lrelu 
        
        
        
        ######## level 2 context pathway : 64x64x64
        #print("context 2")
        
        out_context2_1 = self.context2_1(out_context1_2_lrelu)
        #print(out_context2_1.shape)
        residual2 = out_context2_1
        out_context2_2_1 = self.context2_2_1(out_context2_1)
        out_context2_dropout = self.context2_dropout(out_context2_2_1)
        out_context2_2_2 = self.context2_2_2(out_context2_dropout)
        #Elementwise summation
        out_context2_2_2 += residual2
        out_context2_2_norm = self.context2_2_norm(out_context2_2_2)
        out_context2_2_lrelu = self.context2_2_lrelu(out_context2_2_norm)
        context2 = out_context2_2_lrelu
        
        ######## level 3 context pathway : 32x32x32
        #print("context 3")
        out_context3_1 = self.context3_1(out_context2_2_lrelu)
        residual3 = out_context3_1
        out_context3_2_1 = self.context3_2_1(out_context3_1)
        out_context3_dropout = self.context3_dropout(out_context3_2_1)
        out_context3_2_2 = self.context3_2_2(out_context3_dropout)
        #Elementwise summation
        out_context3_2_2 += residual3
        out_context3_2_norm = self.context3_2_norm(out_context3_2_2)
        out_context3_2_lrelu = self.context3_2_lrelu(out_context3_2_norm)
        context3 = out_context3_2_lrelu
        
        ######## level 4 context pathway : 16x16x16
        #print("context 4")
        out_context4_1 = self.context4_1(out_context3_2_lrelu)
        residual4 = out_context4_1
        out_context4_2_1 = self.context4_2_1(out_context4_1)
        out_context4_dropout = self.context4_dropout(out_context4_2_1)
        out_context4_2_2 = self.context4_2_2(out_context4_dropout)
        #Elementwise summation
        out_context4_2_2 += residual4
        out_context4_2_norm = self.context4_2_norm(out_context4_2_2)
        out_context4_2_lrelu = self.context4_2_lrelu(out_context4_2_norm)
        context4 = out_context4_2_lrelu
        
        
        ######## level 5 context pathway : 8x8x8
        #print("context 5")
        out_context5_1 = self.context5_1(out_context4_2_lrelu)
        residual5 = out_context5_1
        out_context5_2_1 = self.context5_2_1(out_context5_1)
        out_context5_dropout = self.context5_dropout(out_context5_2_1)
        out_context5_2_2 = self.context5_2_2(out_context5_dropout)
        #Elementwise summation
        out_context5_2_2 += residual5
        out_context5_2_norm = self.context5_2_norm(out_context5_2_2)
        out_context5_2_lrelu = self.context5_2_lrelu(out_context5_2_norm)
        
        
              
        ####### level 5 upsampling
        #print("decode 5")
        out_upsample5 = self.upsample5(out_context5_2_lrelu)
        
        ##### level 4 concat + localization + upsampling
        #print("decode 4")
            ## concat
        out_concat4 = torch.cat([out_upsample5, context4], dim=1)  
        out_local4_1 = self.local4_1(out_concat4)
        out_local4_2 = self.local4_2(out_local4_1)
        out_upsample4 = self.upsample4(out_local4_2)
        
        ##### level 3 concat + localization + upsampling
        #print("decode 3")
            ## concat
        out_concat3 = torch.cat([out_upsample4, context3], dim=1)  
        out_local3_1 = self.local3_1(out_concat3)
            ## segment3 pulled out
        segment3 = out_local3_1
        out_local3_2 = self.local3_2(out_local3_1)
        out_upsample3 = self.upsample3(out_local3_2)
        
        ##### level 2 concat + localization + upsampling
        #print("decode 2")
            ## concat
        out_concat2 = torch.cat([out_upsample3, context2], dim=1)  
        out_local2_1 = self.local2_1(out_concat2)
            ## segment3 pulled out
        segment2 = out_local2_1
        out_local2_2 = self.local2_2(out_local2_1)
        out_upsample2 = self.upsample2(out_local2_2)
        
        ##### level 1 concat + localization + upsampling
        #print("decode 1")
            ## concat
        out_concat1 = torch.cat([out_upsample2, context1], dim=1)  
        out_local1 = self.local1(out_concat1)
            ## segment3 pulled out
        segment1 = out_local1
        
        
        #### segmentation layer 
        #print("segment layer")
        segment3 = self.seg3(segment3)
        segment3 = nn.Upsample(size=(128,128,128))(segment3)
        #segment3 = nn.Softmax(dim=1)(segment3)
        
        segment2 = self.seg2(segment2)
        segment2 = nn.Upsample(size=(128,128,128))(segment2)
        #segment2 = nn.Softmax(dim=1)(segment2)
        
        segment1 = self.seg1(segment1)
        #segment1 = nn.Softmax(dim=1)(segment1)
        
        output_segment = torch.cat([segment1, segment2, segment3], dim=1)
        
        #return segment1, segment2, segment3
        return output_segment
        #return segment1