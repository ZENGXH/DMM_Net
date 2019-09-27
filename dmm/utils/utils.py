from torch.autograd import Variable
import torch
import os
import numpy as np
import pickle
import yaml
import logging
from collections import OrderedDict
from torchvision import transforms
from maskrcnn_benchmark.structures.bounding_box import BoxList
def load_checkpoint(model_name,use_gpu=True):
    if use_gpu:
        encoder_dict = torch.load(os.path.join(model_name,'encoder.pt'))
        decoder_dict = torch.load(os.path.join(model_name,'decoder.pt'))
        enc_opt_dict = torch.load(os.path.join(model_name,'enc_opt.pt'))
        dec_opt_dict = torch.load(os.path.join(model_name,'dec_opt.pt'))
    else:
        encoder_dict = torch.load(os.path.join(model_name,'encoder.pt'), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join(model_name,'decoder.pt'), map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join(model_name,'enc_opt.pt'), map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join(model_name,'dec_opt.pt'), map_location=lambda storage, location: storage)
    # save parameters for future use
    args = pickle.load(open(os.path.join(model_name,'args.pkl'),'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args


def get_image_transforms():
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], # for BGR
                                     std=[1,1,1])
    to_bgr_transform = transforms.Lambda(lambda x: x[[2,1,0]] * 255)
    # do normalize on rgb with [0-1]; then to bgr and scale 255 
    image_transforms = transforms.Compose([  #to_tensor, 
                                            to_bgr_transform, 
                                            normalize,
                                              ])
    return image_transforms
 
def load_trainer_iter(model_name, current_args):
    """
    return {} is empty; for resume-training 
    """
    # save parameters for future use
    if current_args.overwrite_loadargs:
        args = current_args 
        if current_args.local_rank == 0: logging.info('overwrite_loadargs by current_args')
    else:
        args = pickle.load(open(os.path.join(model_name,'../args.pkl'),'rb'))
    if current_args.local_rank == 0: logging.info('load ckp from %s'%model_name)
    trainer_dict = torch.load(os.path.join(model_name, 'trainer.pt'), map_location=torch.device("cpu"))
    enc_opt_dict = torch.load(os.path.join(model_name,'enc_opt.pt'), map_location=torch.device("cpu"))
    dec_opt_dict = torch.load(os.path.join(model_name,'dec_opt.pt'))
    return trainer_dict, enc_opt_dict, dec_opt_dict


def load_checkpoint_iter(model_name, current_args):
    """ for evaluation 
    """
    def parse_modules_dict(trainer_dict, name='encoder'):
        new_state_dict = OrderedDict()
        # for non-data parallel model 
        for k, v in trainer_dict.items():
            if name in k:
                new_k = k.split(name)[-1][1:] # get rid of decoder 
                new_state_dict[new_k] = v 

        name = 'module.' + name
        for k, v in trainer_dict.items():
            if name in k:
                new_k = k.split(name)[-1][1:] # get rid of decoder 
                new_state_dict[new_k] = v 
        return new_state_dict
    lrank0 = current_args.local_rank == 0
    # save parameters for future use
    if not os.path.exists(os.path.join(model_name,'../args.pkl')):
        files = '/'.join(os.path.dirname(model_name).split('/')[:-1])
        # files = os.listdir(model_name+'../')
        getf = []
        for f in files:
            if 'args.pkl' in f:
                getf.append(f)
        if lrank0: logging.info('found pkl {}'.format(getf))
        if len(getf) == 0:
            args = current_args 
            if lrank0: logging.info('pkl not found in %s/../, use current_args'%model_name)
        else:
            args = pickle.load(open(os.path.join(model_name,'../%s'%getf[0]),'rb'))
    else:
        args = pickle.load(open(os.path.join(model_name,'../args.pkl'),'rb'))

    for k, v in current_args.__dict__.items():
        if k not in args.__dict__:
            if lrank0: logging.info('-'*10 + ' copy attribute {}: {}; '.format(k, v)+ '-'*10)
            args.__dict__[k] = v 
    for k in ['pred_offline_path', 'pred_offline_meta']:
        if current_args.__dict__[k] != args.__dict__[k]:
            if lrank0: logging.info('update args dict {} from {} -> {}'.format(k, args.__dict__[k], current_args.__dict__[k]))
            args.__dict__[k] = current_args.__dict__[k]
    if lrank0: 
        logging.info('====='*20)
        logging.info('load ckp from %s'%model_name)
    trainer_dict = torch.load(os.path.join(model_name, 'trainer.pt'))
    enc_opt_dict = torch.load(os.path.join(model_name,'enc_opt.pt'))
    dec_opt_dict = torch.load(os.path.join(model_name,'dec_opt.pt'))
    logging.debug('{}'.format(trainer_dict.keys()))
    
    encoder_dict = parse_modules_dict(trainer_dict, 'encoder')
    decoder_dict = parse_modules_dict(trainer_dict, 'decoder')
    #print(trainer_dict.keys())
    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args


def binmask_to_bbox_xyxy_pt(mask, pad=0, zero_pad=False):
    """
    # refer: https://github.com/scaelles/DEXTR-PyTorch/blob/00a0a84002992df506c9fda6a685bd496a67f097/dataloaders/helpers.py#L153
    return the bounding box of those >0 region in the mask 
    can not be applied on those has several objects or has large noise
    """
    inds = torch.nonzero((mask > 0).float())
    assert(len(mask.shape) == 2), mask.shape
    if inds.shape[0] == 0:
        return [] 
    if zero_pad:
        x_min_bound = -torch.inf
        y_min_bound = -torch.inf
        x_max_bound = torch.inf
        y_max_bound = torch.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        # mask: H(vertical), W(hxri)
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1
    # inds shape: [N_non_zero X 2] # 1->indx, 0->indy

    x_min = max(inds[:,1].min() - pad, x_min_bound)
    y_min = max(inds[:,0].min() - pad, y_min_bound)
    x_max = min(inds[:,1].max() + pad, x_max_bound)
    y_max = min(inds[:,0].max() + pad, y_max_bound)
    assert(x_min <= x_max and y_min <= y_max), '{} {} {} {} {}'.format(inds[:,1].min(), inds[:,0].min(),
            inds[:,1].max(), inds[:,0].max(), mask.shape)
    return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]


def binmask_to_bbox_xyxy(mask, pad=0, zero_pad=False):
    """
    # refer: https://github.com/scaelles/DEXTR-PyTorch/blob/00a0a84002992df506c9fda6a685bd496a67f097/dataloaders/helpers.py#L153
    return the bounding box of those >0 region in the mask 
    can not be applied on those has several objects or has large noise
    """
    inds = np.where(mask > 0)
    assert(len(mask.shape) == 2), mask.shape
    if inds[0].shape[0] == 0:
        return [] 
    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)
    if (mask.shape[0] == 1 or mask.shape[1] == 1):
        logging.warning('get mask with shape {}; b {},{},{},{}'.format(mask.shape, x_min, x_max, y_min, y_max))

    assert(x_min <= x_max and y_min <= y_max), '{} {} {} {} {}'.format(inds[1].min(), inds[0].min(),
            inds[1].max(), inds[0].max(), mask.shape)
    return [x_min, y_min, x_max, y_max]


def ohw_mask2boxlist(ohw_mask):
    """
    binary masks of all object, 1 image, to boxlist / rois
    arguments:
        ohw_mask: tensor, OHW, Nobjs binary mask 
    return:
        list_ref_box:
        template_valid: tensor, O 
    """
    O,H,W = ohw_mask.shape
    boxes_xyxy_o = []
    # check if template_valid is valid, by check if the sum is zero  
    template_valid = (ohw_mask.sum(2).sum(1) > 0).long()
    for o in range(O):
        bin_mask = ohw_mask[o]
        boxes_xyxy = binmask_to_bbox_xyxy_pt(bin_mask)
        if len(boxes_xyxy) == 0:
            boxes_xyxy = [0,0,W-1,H-1]
        #check_boxes_xyxy = binmask_to_bbox_xyxy(bin_mask.cpu().numpy())
        #assert(boxes_xyxy == check_boxes_xyxy), '{} {}'.format(boxes_xyxy, check_boxes_xyxy)
        boxes_xyxy_o.append(boxes_xyxy)

    list_ref_box = BoxList(boxes_xyxy_o, (W,H), mode='xyxy')
    list_ref_box.add_field('mask', ohw_mask)
    list_ref_box = list_ref_box.to(ohw_mask.device)
    scores = ohw_mask.new_zeros((O)) + 1
    list_ref_box.add_field('scores', scores)
    template_valid.requires_grad = False
    #if template_valid.sum().item() > 0:
    #    template_valid[0,0] = 1 # at least one template 

    return list_ref_box, template_valid

def load_DMM_config(config_train, local_rank, default='dmm/configs/default.yaml'):
    if local_rank == 0: logging.info('[load_DMM_config] {}'.format(config_train))
    with open(default, 'r') as f:
        cfgs = yaml.safe_load(f)
    info = []
    if config_train is not None:
        with open(config_train, 'r') as f:
            new_cfgs = yaml.safe_load(f)
        for key in new_cfgs.keys():
            if type(new_cfgs[key]) == dict:
                for kkey in new_cfgs[key].keys():
                    if new_cfgs[key][kkey] == cfgs[key][kkey]: continue 
                    info.append('ud {} {} -> {}'.format(key, cfgs[key][kkey], new_cfgs[key][kkey]))
                    cfgs[key][kkey] = new_cfgs[key][kkey]
            elif new_cfgs[key] == cfgs[key]: 
                continue
            else:
                info.append('ud {} {} -> {}'.format(key, cfgs[key], new_cfgs[key]))
                cfgs[key] = new_cfgs[key]
    info = '|'.join(info)
    if local_rank == 0: logging.info(info)
    return cfgs 

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def get_optimizer(optim_name, lr, parameters, weight_decay = 0, momentum = 0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                lr=lr, weight_decay = weight_decay,
                                momentum = momentum)
    elif optim_name =='adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    elif optim_name =='rmsprop':
        opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    return opt

def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'coco':
        skip_dims_in = [2048,1024,512,256,64]
    elif model_name == 'resnet34':
        skip_dims_in = [512,256,128,64,64]
    elif model_name =='vgg16':
        skip_dims_in = [512,512,256,128,64]
    return skip_dims_in
