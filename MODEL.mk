#
# Datasets
#
PUBTABNET = dataset=pubtabnet ++dataset.root_dir=$(pubtabnet_dir)

#
# Models
#
IMGCNN = model/model/backbone=imgcnn
IMGLINEAR = model/model/backbone=imglinear
IMGCONVSTEM = model/model/backbone=imgconvstem

# encoder decoder
e2_d4 = ++model.model.encoder.nlayer=2 ++model.model.decoder.nlayer=4
e4_d4 = ++model.model.encoder.nlayer=4 ++model.model.decoder.nlayer=4

# input size
i224 = ++trainer.img_size=224
i252 = ++trainer.img_size=252
i476 = ++trainer.img_size=476
i392 = ++trainer.img_size=392
i504 = ++trainer.img_size=504

# resnets
r18 = ++model.model.backbone.backbone._target_=torchvision.models.resnet18 ++model.model.backbone.output_channels=512
r34 = ++model.model.backbone.backbone._target_=torchvision.models.resnet34 ++model.model.backbone.output_channels=512
r50 = ++model.model.backbone.backbone._target_=torchvision.models.resnet50 ++model.model.backbone.output_channels=2048

# downsample factor/patch size
p8 = ++model.backbone_downsampling_factor=8
p14 = ++model.backbone_downsampling_factor=14
p16 = ++model.backbone_downsampling_factor=16
p28 = ++model.backbone_downsampling_factor=28
p56 = ++model.backbone_downsampling_factor=56
p112 = ++model.backbone_downsampling_factor=112

# output channels of conv stem
cs_c192 = ++model.model.backbone.output_channels=192
cs_c384 = ++model.model.backbone.output_channels=384

# kernel size in conv stem
cs_k3 = ++model.model.backbone.kernel_size=3
cs_k5 = ++model.model.backbone.kernel_size=5

# transformer
nhead8 = ++model.nhead=8

# cnn backbone
MODEL_r18_e2_d4 = $(IMGCNN) $(e2_d4) $(r18)
MODEL_r34_e2_d4 = $(IMGCNN) $(e2_d4) $(r34)
MODEL_r50_e2_d4 = $(IMGCNN) $(e2_d4) $(r50)

# linear projection
MODEL_p14_e4_d4_nhead8 = $(IMGLINEAR) $(p14) $(e4_d4) $(nhead8)
MODEL_p16_e4_d4_nhead8 = $(IMGLINEAR) $(p16) $(e4_d4) $(nhead8)
MODEL_p28_e4_d4_nhead8 = $(IMGLINEAR) $(p28) $(e4_d4) $(nhead8)
MODEL_p56_e4_d4_nhead8 = $(IMGLINEAR) $(p56) $(e4_d4) $(nhead8)
MODEL_p112_e4_d4_nhead8 = $(IMGLINEAR) $(p112) $(e4_d4) $(nhead8)

# conv stem
MODEL_cs_c384_e4_d4_nhead8 = $(IMGCONVSTEM) $(cs_c384) $(cs_k3) $(e4_d4) $(nhead8)
MODEL_cs_c384_k5_e4_d4_nhead8_i476 = $(IMGCONVSTEM) $(cs_c384) $(cs_k5) $(e4_d4) $(nhead8) $(i476)
MODEL_cs_c192_p8_k5_e4_d4_nhead8_i224 = $(IMGCONVSTEM) $(cs_c192) $(p8) $(cs_k5) $(e4_d4) $(nhead8) $(i224)
MODEL_cs_c384_e4_d4_nhead8_i252 = $(IMGCONVSTEM) $(cs_c384) $(cs_k3) $(e4_d4) $(nhead8) $(i252)
MODEL_cs_c384_k5_e4_d4_nhead8_i392 = $(IMGCONVSTEM) $(cs_c384) $(cs_k5) $(e4_d4) $(nhead8) $(i392)
MODEL_cs_c384_k5_e4_d4_nhead8_i504 = $(IMGCONVSTEM) $(cs_c384) $(cs_k5) $(e4_d4) $(nhead8) $(i504)
MODEL_cs_c384_k5_e4_d4_nhead8 = $(IMGCONVSTEM) $(cs_c384) $(cs_k5) $(e4_d4) $(nhead8)

#
# Optimizer
#
OPT_adamw = trainer/train/optimizer=adamw

#
# lr + scheduler
#
OPT_cosinelr = trainer/train/lr_scheduler=cosine

#
# Regularization
#
REG_d00 = ++model.dropout=0.
REG_d02 = ++model.dropout=0.2