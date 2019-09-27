import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='RIASS')
    # DMM related: 
    # parser.add_argument('-benchmark_cfg', dest='benchmark_cfg', default = 'configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml')
    parser.add_argument('-config_train',  dest='config_train', default = 'configs/default.yaml')
    parser.add_argument('-prev_mask_d',   dest='prev_mask_d', default=1, \
            type=int, help='dimension of prev_mask, original is 1, but if concat with DMM prediction, become 2')
    parser.add_argument('-year', dest='year', default = '2017')
    parser.add_argument('-imsize',dest='imsize', default=480, type=int)
    parser.add_argument('-test_image_w',dest='test_image_w', default=448, type=int)
    parser.add_argument('-test_image_h',dest='test_image_h', default=256, type=int, help='shape of image during evaluation ')
    parser.add_argument('-batch_size', dest='batch_size', default = 10, type=int)
    parser.add_argument('-num_workers', dest='num_workers', default = 0, type=int)
    parser.add_argument('-length_clip', dest='length_clip', default = 3, type=int)
    parser.add_argument('--single_object',dest='single_object', action='store_true')
    parser.set_defaults(single_object=False)
    parser.add_argument('--only_temporal',dest='only_temporal', action='store_true')
    parser.set_defaults(only_temporal=False)
    parser.add_argument('--only_spatial',dest='only_spatial', action='store_true')
    parser.set_defaults(only_spatial=False)
    parser.add_argument('-load_proposals_dataset',default=0,  type=int,help='use offine prop in dataloader') 
    parser.add_argument('-load_proposals',    dest='load_proposals',default=0, type=int,help='use offine prop') 
    parser.add_argument('-use_refmask',       dest='use_refmask',default=0,type=int,help='use reference mask as input') 
    parser.add_argument('-pred_offline_path_eval',dest='pred_offline_path_eval', nargs='+',  type=str, help='offline proposal') 
    parser.add_argument('-pred_offline_path',dest='pred_offline_path', nargs='+',  type=str, help='offline proposal') 
    parser.add_argument('-pred_offline_meta',dest='pred_offline_meta', default='../data/ytb_vos/splits_813_3k_trainvaltest/meta_vid_frame_2_predid.json', type=str,
                        help='offline proposal vid+frame to pred id ') 
    
    ## TRAINING parameters ##
    parser.add_argument('--resume', dest='resume',action='store_true', help=('whether to resume training an existing model ' '(the one with name model_name will be used)'))
    parser.set_defaults(resume=False)
    parser.add_argument('-resume_path', dest='resume_path', default='epoxx_iterxxxx', type=str, help='the epoch name of the resume model; DMM model')
    parser.add_argument('-models_root', dest='models_root', default='experiments/', type=str, help='root to save model') #the epoch name of the resume model; DMM model')
    parser.add_argument('-epoch_resume', dest='epoch_resume',default= 0,type=int, help=('set epoch_resume if you want flags '
                        '--finetune_after and --update_encoder to be properly '
                        'activated (eg if you stop training for whatever reason ' 'at epoch 15, set epoch_resume to 15)'))
    parser.add_argument('-overwrite_loadargs', dest='overwrite_loadargs', default=1, type=int, help='use when resume training, use the loaded args or overwrite it')
    parser.add_argument('-seed', dest='seed',default = 123, type=int)
    parser.add_argument('-gpu_id', dest='gpu_id',default = 0, type=int)
    parser.add_argument('--local_rank', dest='local_rank',default = 0, type=int)
    parser.add_argument('-lr_decoder', dest='lr_decoder', default = 1e-3,type=float)
    parser.add_argument('-lr', dest='lr', default = 1e-3, type=float)
    parser.add_argument('-lr_cnn', dest='lr_cnn', default = 1e-4, type=float)
    parser.add_argument('-optim_cnn', dest='optim_cnn', default = 'adam', choices=['adam','sgd','rmsprop'])
    parser.add_argument('-momentum', dest='momentum', default =0.9,type=float)
    parser.add_argument('-weight_decay', dest='weight_decay', default = 1e-6, type=float)
    parser.add_argument('-weight_decay_cnn', dest='weight_decay_cnn', default = 1e-6, type=float)
    parser.add_argument('-optim', dest='optim', default = 'adam',
                        choices=['adam','sgd','rmsprop'])
    parser.add_argument('-maxseqlen', dest='maxseqlen', default = 5, type=int)
    parser.add_argument('-gt_maxseqlen', dest='gt_maxseqlen', default = 5, type=int)
    parser.add_argument('-best_val_loss', dest='best_val_loss', default = 0, type=float)
    parser.add_argument('-train_h', dest='train_h', type=int, default=255, help='shape of input image, need to be divisible by 32 ')
    parser.add_argument('-train_w', dest='train_w', type=int, default=448, help='shape of input image ')
    parser.add_argument('-max_eval_iter', dest='max_eval_iter', type=int, default=800, help='early stop for evaluation') # shape of input image ')
    parser.add_argument('-sample_inference_mask', dest='sample_inference_mask', type=int, default=0, help='use inference mask as prev mask during training') 
    parser.add_argument('-cache_data', dest='cache_data', type=int, default=1, help='cache the data in dataloader') 
    parser.add_argument('-skip_empty_starting_frame', dest='skip_empty_starting_frame', type=int, default=0, help='skip the template which is empty') 
    parser.add_argument('-random_select_frames', dest='random_select_frames', type=int, default=0, help='step size vary') 

    # base model fine tuning
    parser.add_argument('-finetune_after', dest='finetune_after', default = 0, type=int,
                        help=('epoch number to start finetuning. set -1 to not finetune.'
                        'there is a patience term that can allow starting to fine tune '
                        'earlier (does not apply if value is -1)'))
    parser.add_argument('--update_encoder', dest='update_encoder', default=1, help='update weights of encoder or not.')

    parser.add_argument('-min_delta', dest='min_delta', default=0.0, type=float)

    # stopping criterion
    parser.add_argument('-patience', dest='patience', default = 15, type=int,
                        help=('patience term to activate flags such as '
                        'use_class_loss, feed_prediction and update_encoder if '
                        'their matching vars are not -1'))
    parser.add_argument('-patience_stop', dest='patience_stop', default = 60, type=int,
                        help='patience to stop training.')
    parser.add_argument('-max_epoch', dest='max_epoch', default = 100, type=int)

    # visualization and logging
    parser.add_argument('-print_every', dest='print_every', default = 2, type=int)
    parser.add_argument('-save_every', dest='save_every', default = 3000, type=int)
    parser.add_argument('--log_term', dest='log_term', action='store_true', help='if activated, will show logs in stdout instead of log file.')
    parser.set_defaults(log_term=False)

    # loss weights
    parser.add_argument('-iou_weight',dest='iou_weight',default=1.0, type=float)
    parser.add_argument('-loss_weight_iouraw',dest='loss_weight_iouraw',default=18.0, type=float)
    parser.add_argument('-loss_weight_match',dest='loss_weight_match',default=1.0, type=float)

    # augmentation
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.set_defaults(augment=False)
    parser.add_argument('--my_augment', dest='my_augment', action='store_true')
    parser.set_defaults(my_augment=False)
    parser.add_argument('-rotation', dest='rotation', default = 10, type=int)
    parser.add_argument('-translation', dest='translation', default = 0.1, type=float)
    parser.add_argument('-shear', dest='shear', default = 0.1, type=float)
    parser.add_argument('-zoom', dest='zoom', default = 0.7, type=float)

    # GPU
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('-ngpus', dest='ngpus', default=1,type=int)
    parser.add_argument('-distributed', dest='distributed', default=0,type=int)

    parser.add_argument('-base_model', dest='base_model', default = 'coco', choices=['resnet101','resnet50','resnet34','vgg16', 'coco'])
    parser.add_argument('-skip_mode', dest='skip_mode', default = 'concat',
                        choices=['sum','concat','mul','none'])
    parser.add_argument('-model_name', dest='model_name', default='model')
    parser.add_argument('-log_file', dest='log_file', default='train.log')
    parser.add_argument('-hidden_size', dest='hidden_size', default = 128, type=int)
    parser.add_argument('-kernel_size', dest='kernel_size', default = 3, type=int)
    parser.add_argument('-dropout', dest='dropout', default = 0.0, type=float)

    # dataset parameters
    parser.add_argument('--resize',dest='resize', action='store_true')
    parser.set_defaults(resize=False)
    parser.add_argument('-num_classes', dest='num_classes', default = 21, type=int)
    parser.add_argument('-dataset', dest='dataset', default = 'youtube',choices=['youtube'])
    parser.add_argument('-youtube_dir', dest='youtube_dir', default='../../databases/YouTubeVOS/')

    # testing
    parser.add_argument('-test_model_path', dest='test_model_path', default='')
    parser.add_argument('-davis_eval_folder', dest='davis_eval_folder', default='')
    parser.add_argument('-eval_split',dest='eval_split', default='trainval')
    parser.add_argument('-train_split',dest='train_split', default='train')
    parser.add_argument('-mask_th',dest='mask_th', default=0.5, type=float)
    parser.add_argument('-test', dest='test', default=0, type=int, help='specify to enter test mode, used in load coco model when num_classes = 2')
    parser.add_argument('-eval_flag', default='pred', type=str, help='used to distingush different test setting, mannully')
    parser.add_argument('-threshold_mask',dest='threshold_mask', default=0.4, type=float, help='used in plot DMM merged')
    parser.add_argument('-max_dets',dest='max_dets', default=100, type=int)
    parser.add_argument('-min_size',dest='min_size', default=0.001, type=float)
    parser.add_argument('-pad_video', dest='pad_video', type=int, help='use when multi-gpu', default=0)
    parser.add_argument('-distributed_manully', dest='distributed_manully', type=int, help='distributed_manully', default=0) #use when multi-gpu', default=0)
    parser.add_argument('-distributed_manully_Nrep', dest='distributed_manully_Nrep', type=int, help='N part/device', default=0) 
    parser.add_argument('-distributed_manully_rank', dest='distributed_manully_rank', type=int, help='manually set local rank', default=0)
    return parser

if __name__ =="__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
