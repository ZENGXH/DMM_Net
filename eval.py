import faulthandler
faulthandler.enable()
from args import get_parser
from PIL import Image
import torch
import torch.utils.data as data
import sys, os
import json
import logging
import time
from collections import OrderedDict
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import dmm.utils.Colorer 
from dmm.utils.utils import load_checkpoint_iter, load_DMM_config 
from dmm.modules.model_encoder import FeatureExtractor
from dmm.modules.base import RSISMask 
from dmm.modules.dmm_model import DMM_Model
from dmm.modules.evaluator import Evaler
from dmm.dataloader.dataset_utils import get_dataset
import dmm.dataloader.collate as collate 
from dmm.utils.checker import *

def strip_prefix_if_present(state_dict, prefix, replacement):
    keys = sorted(state_dict.keys())
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, replacement)] = value
    return stripped_state_dict

class Evaluate():
    def __init__(self,args):
        self.split = args.eval_split
        self.dataset = args.dataset
        self.args = args
        encoder_dict, decoder_dict, _, _, _ = load_checkpoint_iter(args.model_name, args)
        encoder = FeatureExtractor(args)
        decoder = RSISMask(args)
        dmm_config = load_DMM_config(args.config_train, args.local_rank)
        DMM = DMM_Model(dmm_config, is_test=1) 

        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)
        if args.use_gpu:
            encoder.cuda()
            decoder.cuda()
            DMM.cuda()
        self.evaler = Evaler(DMM, encoder, decoder, args, dmm_config)
        self.evaler.eval() 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_transforms = transforms.Compose([transforms.ToTensor(), normalize])
        dataset = get_dataset(args, split=self.split, image_transforms=image_transforms,
                            target_transforms=None, augment=0, inputRes=(args.test_image_h, args.test_image_w),
                            video_mode=True, use_prev_mask=True)
        assert(args.pad_video) # pad to same length 
        if args.distributed_manully:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset, 
                           num_replicas=args.distributed_manully_Nrep, rank=args.distributed_manully_rank)
        else: # pad video, not distributed 
            eval_sampler = torch.utils.data.distributed.DistributedSampler( dataset, num_replicas=1, rank=0)
        self.loader = data.DataLoader(dataset, collate_fn=collate.eval_collate, batch_size=args.batch_size,
                                     shuffle=False, sampler=eval_sampler, num_workers=args.num_workers, drop_last=False)
        if args.ngpus > 1 and args.use_gpu:
            self.evaler = torch.nn.DataParallel(self.evaler,device_ids=range(args.ngpus))
        self.video_mode = True

    def run_eval(self):
        if isrank0:
            logging.info("Dataset is %s; len of loader %d"%(self.dataset, len(self.loader)))
            logging.info("Split is %s"%(self.split))
        meters =  MetricLogger(delimiter=" ") 

        # loop over data loader 
        start = time.time() 
        # -------------------   forward model ------------------------------
        for batch_idx, (inputs,imgs_names,targets,seq_name,starting_frame) in enumerate(self.loader):
            meters.update(dT=time.time() - start) 
            if batch_idx % 5 == 0:
                logging.info('[{}] {}/{};{} '.format(args.distributed_manully_rank, batch_idx, len(self.loader), meters))
            targets = targets.cuda() # use our collate function
            inputs = inputs.cuda() 
            cur_device = inputs.device
            CHECK4D(targets) # B, Len, O, HW
            CHECK5D(inputs) # B Len D H W
            if args.load_proposals_dataset:
                proposals_cur_batch = imgs_names 
                proposals = []
                for proposal_cur_vid in proposals_cur_batch:
                    boxlist = list(proposal_cur_vid) # BoxList of current batch 
                    boxlist = [b.to(cur_device) for b in boxlist]
                    proposals.append(boxlist) # BoxList of current batch 
                imgs_names = None 
            else:
                proposals = None
            with torch.no_grad():
                self.evaler(batch_idx,inputs,imgs_names,targets,seq_name,args, proposals)
            meters.update(bT=time.time() - start)
            start = time.time() 

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert(args.test)
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logger = logging.getLogger()
    logger.handlers = [] 
    eval_log_dir = 'experiments/log/'
    file_log = '%s/%seval.log'%(eval_log_dir, args.model_name.replace('//','').replace('/', '-'))
    if not os.path.exists(os.path.dirname(file_log)):
        os.makedirs(os.path.dirname(file_log))
    fh = logging.FileHandler(file_log)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    global isrank0
    isrank0 = args.local_rank == 0
    if isrank0: logging.info('[model_name] %s'%args.model_name)
    E = Evaluate(args)
    if isrank0: 
        logging.info('using num gpu {}'.format(args.ngpus))
        logging.info('save log at %s'%(file_log)) 
    E.run_eval()
    logging.info('~~'*40 + 'DONE rank %d'%args.local_rank)
