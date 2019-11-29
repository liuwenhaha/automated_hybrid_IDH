# automated_hybrid_IDH
This is public repository for "Fully Automated Hybrid Network to Predict IDH Mutation Status of Glioma via Deep Learning and Radiomics" by Choi et al.
 
  The hybrid model comprised three parts : 
  - Model 1 : CNN (U-Net) for tumor segmentation
  - Model 2 : CNN (ResNet) classifier for IDH status
  - Model 3 : Radiomics classifier for IDH status 
  
   Model 1 and 2 are written in Python (PyTorch). The Model 3 (radiomics_classifier.rds) and the logistic model (hybrid_logit.rds) to combine Model 2 and Model 3 are written in R. 
   
   One sample with 3 images (T1C, FLAIR, T2) is availalbe in this repository (example_t1c.nii.gz, example_flair.nii.gz, and example_t2.nii.gz). The overall process of applying the pretrained model to this sample is demonstrated in Model_testing.ipynb
   
   

