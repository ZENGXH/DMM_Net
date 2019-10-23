import matplotlib
matplotlib.use('Agg')
import faulthandler
faulthandler.enable()
import matplotlib.pyplot as plt
from args import get_parser
import logging
import torch
import torchvision.models as models
import yaml
from dmm.modules.dmm_model import DMM_Model
from dmm.modules.trainer import Trainer
from dmm.modules.base import RSISMask as RefineMask
from dmm.modules.model_encoder import FeatureExtractor
from dmm.utils.checker import *
from dmm.utils.utils import get_optimizer, make_dir, load_DMM_config, load_checkpoint_iter, load_trainer_iter, get_image_transforms, load_checkpoint 
from dmm.dataloader.default_collate import default_collate
from dmm.dataloader.dataset_utils import get_dataset
import dmm.utils.Colorer 

import numpy as np
from torchvision import transforms
import torch.utils.data as data
from torch import distributed, nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# others 
import subprocess  
import time
import os
import warnings
import sys
from PIL import Image
import pickle
import random
from tensorboardX import SummaryWriter
from dmm.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict 
from maskrcnn_benchmark.utils.model_serialization import load_state_dict 
seed=31
DEBUG=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=True 
warnings.filterwarnings("ignore")

def save_checkpoint_iter(model, args, it, enc_opt, dec_opt):
    if args.local_rank > 0: return 
    if not os.path.exists(os.path.join(args.models_root,args.model_name, it)):
        os.makedirs(os.path.join(args.models_root,args.model_name, it)) 
    torch.save(model.state_dict(), os.path.join(args.models_root,args.model_name, it, 'trainer.pt'))
    saved_dict = model.state_dict()
    if enc_opt is not None: 
        torch.save(enc_opt.state_dict(), os.path.join(args.models_root,args.model_name, it, 'enc_opt.pt'))
    if dec_opt is not None: 
        torch.save(dec_opt.state_dict(), os.path.join(args.models_root,args.model_name, it, 'dec_opt.pt'))
    logging.info('save checkpoints at %s'%os.path.join(args.models_root,args.model_name, it))

def average_gradients(model, rank=-1): 
    size = float(dist.get_world_size()) 
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, async_op=True) 
        param.grad.data /= size
def init_dataloaders(args):
    loaders = {}
    # init dataloaders for training and validation
    for split in [args.train_split, args.eval_split]:
        batch_size = args.batch_size  if split == args.train_split \
                else max(1, int(args.batch_size / 2))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_transforms = transforms.Compose([transforms.ToTensor(), normalize])
        # image_transforms = get_image_transforms()
        dataset = get_dataset(args,
                                split=split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and split == args.train_split,
                                inputRes = (args.train_h, args.train_w), 
                                video_mode = True,
                                use_prev_mask = False)

        if args.ngpus > 1 and args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train_sampler is None,
                                         collate_fn=default_collate,
                                         sampler=train_sampler,
                                         num_workers=args.num_workers, 
                                         drop_last=True)
        else:
            train_sampler = None
            loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=default_collate,
                                         num_workers=args.num_workers, 
                                         drop_last=True)
        if args.local_rank == 0: 
            logging.info('INPUT shape: {} {}'.format(args.train_h, args.train_w))
    return loaders

def build_model(args):
    # init model:
    start = time.time()
    if args.resume:
        # will resume training the model with name args.model_name
        trainer_dict, enc_opt_dict, dec_opt_dict = load_trainer_iter(args.resume_path, args) 
        epoch_resume = args.epoch_resume
        if args.local_rank == 0: logging.info('resume from model {}; overwrite_loadargs by args {}'.format(args.model_name, args.overwrite_loadargs))
        load_args = args
        # build encoder:
        encoder = FeatureExtractor(load_args)
        # benchmark_cfg = encoder.benchmark_cfg  
        # build DMM:
        cfgs = load_DMM_config(args.config_train, args.local_rank)
        if args.local_rank == 0:
           yaml.dump(cfgs, open(args.model_dir+'/'+timestr+'_dmmcfg.yaml', 'w'))
           # yaml.dump(benchmark_cfg, open(args.model_dir+'/'+timestr+'_benchmarkcfg.yaml', 'w'))
           logging.info('{}'.format(cfgs))
        # build decoder  :
        DMM = DMM_Model(cfgs) 
        decoder = RefineMask(load_args)
        trainer = Trainer(DMM, encoder, decoder, args)
    else:
        encoder = FeatureExtractor(args)
        decoder = RefineMask(args)
        if args.base_model == 'resnet101':
            pretrained_path = 'experiments/dmmnet/pretrained_rvos/one-shot-model-youtubevos/'
            if not os.path.isdir(pretrained_path):
                msg = 'pretrained model from rvos not found in %s; please run '%pretrained_path
                msg += '\n wget https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/one-shot-model-youtubevos.zip'
                msg += '\n zip -rq one-shot-model-youtubevos.zip '
                ppath = 'experiments/dmmnet/pretrained_rvos/'
                msg += '\n mkdir -p %s && mv one-shot-model-youtubevos %s/'%(ppath, ppath)
                print(msg)
                raise FileNotFoundError(pretrained_path)

            encoder_dict, decoder_dict,_,_,_ = load_checkpoint(pretrained_path)
            enc_dict_new = encoder.state_dict()
            decoder_dict_new = decoder.state_dict()
            for name, param in enc_dict_new.items(): # named_parameters():
                if name in encoder_dict:
                    CHECKEQ(param.shape, encoder_dict[name].shape)
                    enc_dict_new[name] = encoder_dict[name]
            for name, param in decoder_dict_new.items():
                if name in decoder_dict:
                    CHECKEQ(param.shape, decoder_dict[name].shape)
                    decoder_dict_new[name] = decoder_dict[name]
            decoder.load_state_dict(decoder_dict_new)
            encoder.load_state_dict(enc_dict_new)
        cfgs = load_DMM_config(args.config_train, args.local_rank)
        # benchmark_cfg = encoder.benchmark_cfg  
        DMM = DMM_Model(cfgs)
        if args.local_rank == 0:
           yaml.dump(cfgs, open(args.model_dir+'/'+timestr+'_dmmcfg.yaml', 'w'))
           # yaml.dump(benchmark_cfg, open(args.model_dir+'/'+timestr+'_benchmarkcfg.yaml', 'w'))
           logging.info('{}'.format(cfgs))
        trainer = Trainer(DMM, encoder, decoder, args)
    if args.local_rank == 0: logging.info('init model %.3f'%(time.time() - start))
    start = time.time()
    
    enc_opt = trainer.encoder.get_optimizer(args.optim_cnn, args.lr_cnn, args, args.weight_decay_cnn)
    decoder_params = list(trainer.decoder.parameters()) 
    dec_opt = get_optimizer(args.optim, args.lr_decoder, decoder_params, args.weight_decay)
    if args.local_rank == 0: 
        logging.info('optimizer %.3f'%(time.time() - start))
        logging.info('[enc_opt] len: {}; len for each param group: {}'.format(len(enc_opt.state_dict()['param_groups']),
            [len(k['params']) for k in enc_opt.state_dict()['param_groups']]))
        logging.info('[dec_opt] len: {}; len for each param group: {}'.format(len(dec_opt.state_dict()['param_groups']),
            [len(k['params']) for k in dec_opt.state_dict()['param_groups']]))

    # ---------- init parallel --------------
    if args.ngpus > 1 and args.distributed:
        logging.info('init DistributedDataParallel rank %d'%local_rank)
        trainer = torch.nn.parallel.DistributedDataParallel(
            trainer, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        ).to(args.device)
    elif args.ngpus > 1:
        trainer = torch.nn.DataParallel(trainer, list(range(torch.cuda.device_count()))).cuda()

    # ----------- load model weight ---------
    if args.resume:
        logging.debug('[LOADED enc_opt] len: {}; len for each param group: {}'.format(len(enc_opt_dict['param_groups']),
            [len(k['params']) for k in enc_opt_dict['param_groups']]))
        logging.debug('[LOADED dec_opt] len: {}; len for each param group: {}'.format(len(dec_opt_dict['param_groups']),
            [len(k['params']) for k in dec_opt_dict['param_groups']]))
        if enc_opt is not None: 
            enc_opt.load_state_dict(enc_opt_dict)
        if dec_opt is not None:
            dec_opt.load_state_dict(dec_opt_dict)
        load_state_dict(trainer, trainer_dict)
        logging.info('loaded state_dict from resume model: %s'%args.resume_path)
    return enc_opt, dec_opt, trainer

def trainIters(args):
    """ main: loop over different epoch. and datasplit """
    epoch_resume = args.epoch_resume 
    model_dir = os.path.join(args.models_root, args.model_name)
    board_dir = os.path.join(args.models_root, 'boards', args.model_name)
    if args.local_rank == 0:
        make_dir(board_dir)
        make_dir(model_dir)
    start = time.time()
    meters = {args.train_split: MetricLogger(delimiter=" "), 
                args.eval_split: MetricLogger(delimiter=" ") 
             }
    args.model_dir = model_dir 
    enc_opt, dec_opt, trainer = build_model(args)
    max_eval_iter = args.max_eval_iter 
    # save parameters for future use
    if args.local_rank == 0:
       tb_writer = SummaryWriter(board_dir)
       pickle.dump(args, open(os.path.join(model_dir, timestr + '_args.pkl'),'wb'))
       pickle.dump(args, open(os.path.join(model_dir, 'args.pkl'),'wb')) # overwrite the latest args
       logging.info('save args in %s'%os.path.join(model_dir, timestr + 'args.pkl'))
       logging.info('{}'.format(args))
    start = time.time()
    # vars for early stopping
    best_val_loss = args.best_val_loss
    best_val_epo = 0
    acc_patience = 0
    mt_val = -1

    # keep track of the number of batches in each epoch for continuity when plotting curves
    if args.local_rank == 0: logging.info('init_dataloaders')
    start = time.time()
    loaders = init_dataloaders(args)
    num_batches = {args.train_split: 0, args.eval_split: 0}

    if args.local_rank == 0: logging.info('dataloader %.3f'%(time.time() - start))

    for e in range(args.max_epoch):
        # check if it's time to do some changes here
        if e + epoch_resume >= args.finetune_after and not args.sample_inference_mask and not args.finetune_after == -1:
            args.sample_inference_mask = 1 
            logging.info('='*10 + '> start sample_inference_mask')
            acc_patience, best_val_loss = 0,0
        # in current epoch, loop over split 
        # we validate after each epoch
        if max_eval_iter >0  and e == 0:
            splits = [args.eval_split, args.train_split, args.eval_split]
        elif max_eval_iter == 0:
            splits = [args.train_split]
        else:
            splits = [args.train_split, args.eval_split] 
        for split in splits: 
            if split == args.eval_split:
                trainer.eval() 
            # loop over batches in current epoch
            if args.local_rank == 0:
                logging.info('epoch %d - %s; '%(e+epoch_resume, split))
                logging.info('-- loss weight loss_weight_match: {} loss_weight_iouraw {}; '.format(
                            args.loss_weight_match, args.loss_weight_iouraw ))
                sd = time.time()
                start = time.time()
            iter_time = []
            for batch_idx, (inputs,imgs_names,targets,seq_name,starting_frame) in enumerate(loaders[split]):
                # imgs_names: can be proposals: List[tuple(BoxList)], len of list=Nframe, len-of-tuple=BatchSize 
                if args.local_rank == 0: 
                    start_iter = time.time()
                    dataT = time.time() - sd
                assert(type(targets) == list)
                inputs = [sub.to(args.device) for sub in inputs]
                targets = [sub.to(args.device) for sub in targets]
                if args.load_proposals_dataset:
                    proposals_cur_batch = imgs_names # len=framelen 
                    proposals = [] # BoxList of current batch 
                    for p in proposals_cur_batch:
                        boxlist = list(p) # BoxList of current batch 
                        proposals.append([b.to(args.device) for b in boxlist]) # len=BatchSize 
                    imgs_names = None 
                else:
                    proposals = None

                # forward
                if split == args.eval_split:
                    with torch.no_grad():
                        loss, losses = trainer(batch_idx, inputs, imgs_names, targets, seq_name, starting_frame, split, args, proposals)
                else:
                    loss, losses = trainer(batch_idx, inputs, imgs_names, targets, seq_name, starting_frame, split, args, proposals)
                ## import pdb; pdb.set_trace() 
                #if DEBUG: 
                #    logging.info('>> profile ')
                #    logging.info('seq_name {}, inputs sum {}; proposals: {} imgs_names {}'.format(seq_name, inputs[0].sum(), proposals[0][0].bbox.sum(), imgs_names))
                #    info = {'batch_idx': batch_idx, 'info':[seq_name, inputs[0].shape, inputs[0].sum(), proposals[0][0].bbox.sum(), imgs_names, losses, loss]}
                #    check_info = torch.load('../../drvos/src/debug/%d.pth'%batch_idx)
                #    CHECKDEBUG(info, check_info)

                loss = loss.mean() #reduce_loss_dict({'loss':loss}) 

                if split == args.train_split: # and args.local_rank == 0:
                    dec_opt.zero_grad()
                    enc_opt.zero_grad() 
                    if loss.requires_grad:
                        loss.backward()
                        if args.distributed:
                            average_gradients(trainer, args.local_rank)
                            torch.cuda.synchronize()
                        dec_opt.step()
                        enc_opt.step()
                # record the losses
                # store loss values in dictionary separately
                if args.distributed:
                    losses = reduce_loss_dict(losses)

                if args.ngpus > 1 and args.local_rank == 0:
                    for k, v in losses.items():
                        if not args.distributed:
                            losses[k] = v.mean()
                        tb_writer.add_scalar('%s/%s'%(k, split), losses[k], batch_idx+(e+epoch_resume)*len(loaders[split]))
                elif args.local_rank == 0:
                    for k, v in losses.items(): 
                        tb_writer.add_scalar('%s/%s'%(k, split), v, batch_idx+(e+epoch_resume)*len(loaders[split]))
                if args.local_rank == 0: meters[split].update(**losses)
                # print after some iterations
                if (batch_idx + 1)% args.print_every == 0 and args.local_rank == 0: # iteration 
                    te = time.time() - start_iter
                    iter_time.append(te)
                    remain_t = (sum(iter_time)/len(iter_time) * (len(loaders[split]) - batch_idx))/60.0/60.0 
                    max_mem = "mem: {memory:.0f}".format( memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                    meters[split].update(time=te, dt=dataT)
                    logging.info("%s:%s:p%d(%d-%.2f):E%d it%d/%d: rt(%.2fh) %s|%s"%(args.model_name, split, acc_patience, \
                            best_val_epo, best_val_loss, (e+epoch_resume), batch_idx, len(loaders[split]), remain_t, \
                            str(meters[split]), max_mem))
                    start = time.time()
                if args.local_rank == 0 and split == args.train_split  and (((batch_idx + 1) % args.save_every == 0) \
                        or batch_idx + 1 == len(loaders[split])):
                    logging.info('save model at {} {}'.format(batch_idx, e+epoch_resume))
                    save_checkpoint_iter(trainer, args, 'epo%02d_iter%05d'%(e+epoch_resume, batch_idx), enc_opt, dec_opt)
                sd = time.time()
            # out of for-all-batches in current split 
            num_batches[split] = batch_idx + 1
            # for loss_name in ['loss', 'match_loss', 'iou', 'iouraw', 'hard_iou_raw']:
            if split == args.eval_split:
                for loss_name in ['hard_iou_raw', 'hard_iou']: # prefer hard_iou than hard_iou_raw
                    if loss_name in meters[args.eval_split].fields() and max_eval_iter != 0:
                        mt_val = meters[args.eval_split].load_field(loss_name).global_avg
                meters[args.eval_split] = MetricLogger(delimiter=" ") 
                if mt_val > (best_val_loss + args.min_delta):
                    logging.info("Saving checkpoint.")
                    best_val_loss = mt_val
                    best_val_epo  = e+epoch_resume
                    # saves model, params, and optimizers
                    save_checkpoint_iter(trainer, args, 'best_%.3f_epo%02d'%(best_val_loss, e+epoch_resume),  enc_opt, dec_opt)
                    acc_patience = 0
                else:
                    acc_patience += 1
        if acc_patience > args.patience_stop:
            logging.info('acc_patience reach maximum, I killed my self: Bye ')
            break

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    assert(args.eval_split in ['trainval', 'davis_val']), 'can not used split: {} as eval'.format(args.eval_split)
    assert(args.length_clip > 1), 'not support len {}: no loss will be given '.format(args.length_clip)
    assert(args.eval_split != args.train_split), 'can not use the same name for both train and eval; will have bug in training '

    gpu_id = args.gpu_id
    # set up logger 
    logger = logging.getLogger()
    logger.handlers = [] 
    global timestr
    timestr = time.strftime('%m-%d-%H-%M')
    githash = subprocess.check_output(["git", "describe", "--always"]).strip()
    githash = githash.decode("utf-8")

    log_file = args.models_root+'log/' + timestr + '-GIT%s-'%githash + args.model_name + '-train.log'
    logformat = '%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s'
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    # log file 
    logging.basicConfig(level=logging.INFO, format=logformat, filename=log_file)
    # log terminal 
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(logformat) 
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


    num_gpus = args.ngpus 
    if args.local_rank == 0: 
        logging.info('[model_name] {}'.format(args.model_name))
        logging.info('get number of gpu: %d'%num_gpus)
    if args.use_gpu:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        args.device = torch.device("cuda:%d"%args.local_rank)
    else:
        args.device = torch.device("cpu")
    if args.ngpus > 1 and args.distributed:
        local_rank = args.local_rank 
        logging.info('to distributed; local_rank %d'%local_rank)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        torch.cuda.synchronize()
    elif args.ngpus == 1 and args.distributed:
        args.distributed = 0 # disable it
    trainIters(args)

