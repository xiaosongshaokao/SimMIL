from utils.supContrastLoss import SupConLoss
from .base_runner import BaseRunner
from datasets.my_imagefolder import MyImageFolder
from datasets.camelyon17 import CameLyon17
from datasets.selected_camelyon16 import SelectedCamelyon16
from datasets.debug_dataset import DebugDataset
import os
import time
import itertools
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from utils.core import AverageMeter, ProgressMeter, accuracy, binary_accuracy
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import shutil
import multiprocessing as mp
from tqdm import tqdm
import random
from PIL import ImageFilter
# import multiprocess

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class LinClsRunner(BaseRunner):
    def __init__(self, gpu, ngpus_per_node, args):
        super().__init__(gpu, ngpus_per_node, args)
        self.best_acc1 = 0

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.args.lr
        for milestone in self.args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def build_model(self):
        # create model
        self.logger.info("=> creating model '{}'".format(self.args.arch))
        if self.args.imgnet_pretrained:
            self.logger.info("=> using imagenet pretrained model")
        model = models.__dict__[self.args.arch](pretrained=self.args.imgnet_pretrained)
        in_features = model.fc.in_features
        new_model = nn.Sequential(*list(model.children())[:-1])
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            param.requires_grad = False
        # init the fc layer
        # in_features = model.fc.in_features
        # model.fc = torch.nn.Linear(in_features, self.args.class_num)
        # model.fc.weight.data.normal_(mean=0.0, std=0.01)
        # model.fc.bias.data.zero_()
        return new_model, in_features

    def build_losses(self):
        return nn.CrossEntropyLoss().cuda(self.args.gpu)
    def build_supCon_losses(self):
        return SupConLoss(temperature=self.args.temp).cuda(self.args.gpu)

    def build_optimizer(self, model):
        # optimize only params that require grads
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert len(parameters) == 2  # fc.weight, fc.bias
        optimizer = torch.optim.SGD(parameters, self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        return optimizer

    def build_dataset(self, augmentation, mode='train'):
        """
        Build dataset object based on self.args.
        """
        self.logger.info("Building dataset")
        if self.args.debug:
            self.logger.info("Debuging, using toy datasets")
            dataset = DebugDataset(augmentation)
        else:
            if self.args.dataset=="CAMELYON16":
                sub_dir = 'train' if mode =='train' else 'validation'
                traindir = os.path.join(self.args.data, sub_dir)

                ## TODO: configuration
                # class_label_dict = {"nontumor_256_20X_train": 0,
                #                     "tumor_256_20X_train": 1}
                if self.args.class_num == 3:
                    class_label_dict = {"nontumor_256_20X": 1,
                                        "tumor_256_20X": 2,
                                        "neg_256_20X": 0,}
                elif self.args.class_num == 2:
                    class_label_dict = {"nontumor_256_20X": 0,
                                        "tumor_256_20X": 1,
                                        "neg_256_20X": 0,}
                else:
                    raise Exception("Class num must be 2 or 3 using camelyon16!!")
                
                ## Added using npy
                if sub_dir=="train":
                    select_files = ["select_bag.npy", "select_neg.npy"]
                    bag_name_list =[]
                    for select_file in select_files:
                        bag_name = np.load(os.path.join(traindir, select_file))
                        bag_name_list.extend(list(bag_name))
                    
                    self.logger.info("Bags selected: {}".format(bag_name_list))
                    # label_getter = lambda x: (x['data'], x['label'])
                    dataset = CameLyon17(traindir, augmentation, None, class_label_dict, use_indexs=False, bag_name_list=bag_name_list)
                else:
                    if self.args.selected_test:
                        selected_images = np.load(self.args.test_selected_images)
                        selected_labels = np.load(self.args.test_selected_labels)
                        dataset = SelectedCamelyon16(selected_images, selected_labels, augmentation)
                    else:
                        dataset = CameLyon17(traindir, augmentation, None, class_label_dict, use_indexs=False)
                
                self.logger.info("Length of dataset: {}".format(len(dataset)))

            elif self.args.dataset=='NCTCRC':
                if mode=='train':
                    if self.args.no_norm:
                        train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K-NONORM')
                    else:
                        train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K')
                else:
                    train_dir = os.path.join(self.args.data, self.args.eval_subdir)

                dataset = MyImageFolder(train_dir, augmentation)            
		## for binary classification
                if self.args.class_num == 2 :
                    self.logger.info("using binary classification for NCTCRC")
                    samples = list(dataset.samples)
                    self.logger.info("first 10 samples before transform: {}".format(samples[:10]))
                    samples = [(x[0], int(x[1]==8)) for x in samples]
                    dataset.samples = samples
                    self.logger.info("first 10 samples after transform {}".format(dataset.samples[:10]))
                if os.path.exists(self.args.sample_file) and mode=='train':
                    ## semisupervised or weaklt supervised
                    self.logger.info("=>using sample file {}".format(self.args.sample_file))
                    with open(self.args.sample_file, "r") as f:
                        samples = [x.strip().split(" ") for x in f.readlines()]
                        self.logger.info("=> Using Partial Data")
                        samples = list(map(lambda x:x[0], samples))
                        path_list = list(filter(lambda x: x[0].rsplit("/")[-1] in samples, dataset.samples))
                        dataset.samples = path_list
                        self.logger.info("size of dataset after sampling: {}".format(len(dataset)))
            elif self.args.dataset=='NCTCRC-BAGS':
                if mode=='train':
                    if self.args.no_norm:
                        train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K-NONORM')
                    else:
                        train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K')
                else:
                    train_dir = os.path.join(self.args.data, self.args.eval_subdir)

                dataset = MyImageFolder(train_dir, augmentation)

                postfix = 'train.txt' if mode=='train' else 'val.txt'
                bag_label_file = os.path.join(self.args.bag_label_dir, postfix)
                with open(bag_label_file, 'r') as f:
                        samples = [x.strip().split(" ") for x in f.readlines()]
                        self.logger.info("=> Using MIL Setting")
                        samples = list(map(lambda x:(x[0], int(x[1])), samples))
                        dataset.samples = samples
                        self.logger.info("Examples after MIL: {}".format(dataset.samples[:5]))
            elif self.args.dataset == 'TCGA':
                # if mode=='train':
                #     if self.args.no_norm:
                #         train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K-NONORM')
                #     else:
                #         train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K')
                # else:
                #     train_dir = os.path.join(self.args.data, self.args.eval_subdir)
                train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K')

                dataset = MyImageFolder(train_dir, augmentation)

                postfix = 'train.txt' if mode=='train' else 'val.txt'
                bag_label_file = os.path.join(self.args.bag_label_dir, postfix)
                with open(bag_label_file, 'r') as f:
                        samples = [x.strip().split(" ") for x in f.readlines()]
                        self.logger.info("=> Using MIL Setting")
                        samples = list(map(lambda x:(x[0], int(x[1])), samples))
                        dataset.samples = samples
                        self.logger.info("Examples after MIL: {}".format(dataset.samples[:5]))
                print(len(dataset.samples))
            elif self.args.dataset == 'C16':
                train_dir = os.path.join(self.args.data, 'NCT-CRC-HE-100K')

                dataset = MyImageFolder(train_dir, augmentation)

                postfix = 'train.txt' if mode=='train' else 'val.txt'
                bag_label_file = os.path.join(self.args.bag_label_dir, postfix)
                with open(bag_label_file, 'r') as f:
                        samples = [x.strip().split(" ") for x in f.readlines()]
                        self.logger.info("=> Using MIL Setting")
                        samples = list(map(lambda x:(x[0], int(x[1])), samples))
                        dataset.samples = samples
                        self.logger.info("Examples after MIL: {}".format(dataset.samples[:5]))
                print(len(dataset.samples))
            else:
                ##TODO
                raise NotImplementedError
        return dataset
    
    def load_pretrained(self, model):
        try:
            self.logger.info("=> loading checkpoint '{}'".format(self.args.pretrained))
            checkpoint = torch.load(self.args.pretrained, map_location="cpu")
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            # for k in list(state_dict.keys()):
            #     # retain only encoder_q up to before the embedding layer
            #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            #         # remove prefix
            #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            #     # delete renamed or unused k
            #     del state_dict[k]
            for k in list(state_dict.keys()):
                if k.startwith('module'):
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
            self.args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            self.logger.info(msg)
            # assert set(msg.missing_keys) == {"bsz, n_views, ...fc.weight", "fc.bias"}
            self.logger.info("=> loaded pre-trained model '{}'".format(self.args.pretrained))
        except:
            self.logger.info("=> no checkpoint found at '{}'".format(self.args.pretrained))

        return model

    def resume(self, model, optimizer):
        assert not (self.args.resume and self.args.load)
        if self.args.resume or self.args.load:
            checkpoint_filepath = self.args.resume if self.args.resume else self.args.load
            if os.path.isfile(checkpoint_filepath):
                self.logger.info("=> loading checkpoint '{}'".format(checkpoint_filepath))
                if self.args.gpu is None:
                    checkpoint = torch.load(checkpoint_filepath)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.args.gpu)
                    checkpoint = torch.load(checkpoint_filepath, map_location=loc)
                if self.args.resume:
                    model.load_state_dict(checkpoint['state_dict'])
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    except:
                        self.logger.info("=> (resume) optimizer failed to load!")
                    self.args.start_epoch = checkpoint['epoch']
                    self.best_acc1 = checkpoint['best_acc1']
                    if self.args.gpu is not None:
                        # best_acc1 may be from a checkpoint from a different GPU
                        self.best_acc1 = self.best_acc1.to(self.args.gpu)
                    self.logger.info("=> (resume) loaded checkpoint '{}' (epoch {})"
                        .format(self.args.resume, checkpoint['epoch']))
                elif self.args.load:
                    # drop last layer (default)
                    def load_model(model, state_dict, strict=False):
                        key_list = list(filter(lambda x: x.split(".")[-2]=="fc", state_dict.keys()))

                        for k in key_list:
                            state_dict.pop(k)
                            self.logger.info("Not loading {}".format(k))
                        
                        # Check module as prefix (temp)
                        key_list = list(filter(lambda x: not x.startswith('module.'), state_dict.keys()))
                        for k in key_list:
                            self.logger.info("Transforming {} to module.{}".format(k, k))
                            state_dict["module.{}".format(k)] = state_dict.pop(k)
                        msg = model.load_state_dict(state_dict, strict=strict)

                        return model, msg

                    model, msg = load_model(model, checkpoint['state_dict'], strict=False)

                    self.logger.info("=> loaded checkpoint {}".format(self.args.load))
                    self.logger.info("=> msg: {}".format(msg))
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.args.resume))

        return model

    def run(self):
        model, in_features = self.build_model()
        # load from pre-trained, before DistributedDataParallel constructor
        if self.args.pretrained:
            model = self.load_pretrained(model)

        # lrs = [base * n for base in [100 ** k for k in range(-4, 1)] for n in range(1, 10)]
        lrs = [0.001]
        wds = [0]
        optims = ['sgd']
        permutes = list(itertools.product(lrs, wds, optims))
        linear_classifiers = nn.ModuleList()
        optimizers = []
        schedulers = []
        for pm in permutes:
            lr, wd, optim = pm
            linear_classifier = LinearClassifier(in_features, self.args.class_num)
            model = self.set_device(model)
            device = torch.device("cuda:0")
            model.to(device)
            linear_classifiers.append(linear_classifier)

            # set optimizer
            parameters = linear_classifier.parameters()
            optimizer = torch.optim.SGD
            optimizer = optimizer(parameters, lr, momentum=0.9, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs, eta_min=0)

            optimizers.append(optimizer)
            schedulers.append(scheduler)


        model = self.set_device(model)
        device = torch.device("cuda:0")
        model.to(device)
        criterion = self.build_losses()
        # optimizer = self.build_optimizer(model)      
        # optionally resume from a checkpoint
        #model = self.resume(model, optimizer)
        
        # Check for grad requirement
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         self.logger.info("{} requires grad".format(name)) 
        cudnn.benchmark = True

        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        # imagenet 
        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]
        # C16
        # mean: [0.7515661120414734, 0.6858345866203308, 0.7641154527664185]
        # std: [0.1495480090379715, 0.1610676795244217, 0.13506610691547394]
        # train_augmentation = transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ])
        train_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_augmentation = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        val_dataset = self.build_dataset(val_augmentation, "validation")
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)
        # if self.args.evaluate:
        #     self.validate(val_loader, model, criterion)
        #     return

        train_dataset = self.build_dataset(train_augmentation, "train")
        # if self.args.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        # else:
        #     train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True, sampler=None)

        print(len(train_loader))
        acc1 = 0.0

        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # if self.args.distributed:
            #     train_sampler.set_epoch(epoch)
            # self.adjust_learning_rate(optimizer, epoch)
            model.eval()
            linear_classifiers.eval()
            

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            if (epoch % self.args.validate_interval == 0) and epoch>1:
                self.logger.info("Evaluating...")
                acc1 = self.validate(val_loader, model, criterion)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                    and self.args.g % self.ngpus_per_node == 0):
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.args.log_dir, 'checkpoint_{:04d}.pth'.format(epoch)))
        # pool.close()
    
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filename.rsplit('/',1)[0],'model_best.pth'))

    def sanity_check(self, state_dict, pretrained_weights):
        """
        Linear classifier should not change any weights other than the linear layer.
        This sanity check asserts nothing wrong happens (e.g., BN stats updated).
        """
        self.logger.info("=> loading '{}' for sanity check".format(pretrained_weights))
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        state_dict_pre = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.weight' in k or 'fc.bias' in k:
                continue

            # name in pretrained model
            k_pre = 'module.encoder_q.' + k[len('module.'):] \
                if k.startswith('module.') else 'module.encoder_q.' + k

            assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
                '{} is changed in linear classifier training.'.format(k)

        self.logger.info("=> sanity check passed.")


    def get_meters(self):
        return dict(batch_time = AverageMeter('Time', ':6.3f'),
        data_time = AverageMeter('Data', ':6.3f'),
        losses = AverageMeter('Loss', ':.4e'),
        top1 = AverageMeter('Acc@1', ':6.2f'),
        top5 = AverageMeter('Acc@5', ':6.2f'),
        )

    def train(self, train_loader, model, criterion, optimizer, epoch):
        """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
        """
        meter_set = self.get_meters()
        progress = ProgressMeter(
            len(train_loader),
            meter_set,
            prefix="Epoch: [{}]".format(epoch),
            logger=self.logger)
        
        if not self.args.no_fixed_trunk:
            model.eval()
        else:
            model.train()

        end = time.time()
        # with model.train():
        device = torch.device("cuda:0")
        for i,  data  in enumerate(tqdm(train_loader)):
            images = data['data'].to(device)
            target = data['label'].to(device)
            # measure data loading time
            progress.update('data_time', time.time() - end)

            if self.args.gpu is not None:
                images = images.cuda(self.args.gpu, non_blocking=True)
                target = target.cuda(self.args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, self.args.class_num)))
            progress.update('losses', loss.item(), images.size(0))
            progress.update('top1', acc1[0], images.size(0))
            progress.update('top5', acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            progress.update('batch_time', time.time() - end)
            end = time.time()

            # if i % self.args.print_freq == 0:
        progress.display(i)
        # except OSError:
        #     print(OSError)
        # 每轮训练结束后手动关闭 workers
        # trainloader_iter = iter(train_loader)
        # trainloader_iter._shutdown_workers()

    def validate(self, val_loader, model, criterion):
        meter_set = self.get_meters()
        progress = ProgressMeter(
            len(val_loader),
            meter_set,
            prefix='Test: ',
            logger=self.logger)
        # switch to evaluate mode
        # if self.args.distributed:
        #     model.module.eval()
        model.eval()
        outputs = []
        targets = []
        device = torch.device("cuda:0")
        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(tqdm(val_loader)):
                images = data['data'].to(device)
                target = data['label'].to(device)
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                    target = target.cuda(self.args.gpu, non_blocking=True)

                # compute output
                output = model(images) # [N, C]
                outputs.append(output.cpu())
                targets.append(target.cpu())

                loss = criterion(output, target)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, min(self.args.class_num, 5)))
                progress.update('losses', loss.item(), images.size(0))
                progress.update('top1', acc1[0], images.size(0))
                progress.update('top5', acc5[0], images.size(0))

                # measure elapsed time
                progress.update('batch_time', time.time() - end)
                end = time.time()

                # if i % self.args.print_freq == 0:
            progress.display(i)

            ## logging: multi-class classification with detail
            ## hard prediction
            outputs = torch.cat(outputs, 0).softmax(-1).numpy()
            targets = torch.cat(targets, 0).numpy()
            if self.args.eval_class_num < outputs.shape[-1]:
                self.logger.info(" transfering a classifier head with class num {} to {}".format(outputs.shape[-1], self.args.class_num))
                outputs = outputs[:, :self.args.eval_class_num]
            # measure a binary classification performance for each class
            ## (a) turn each into a binary classification problem
            for pos_class in range(0, self.args.eval_class_num):
                binary_target = (targets==pos_class)
                pos_mask = pos_class
                neg_mask = list(range(0, self.args.eval_class_num))
                neg_mask.remove(pos_mask)
                binary_pred = torch.from_numpy(np.stack([outputs[:, neg_mask].max(1), outputs[:, pos_mask]], axis=1)).softmax(-1).numpy()
                binary_cls_report = classification_report(binary_target, binary_pred.argmax(-1), output_dict=True)
                self.logger.to_csv("binary_report_{}".format(pos_class), binary_cls_report)
                binary_auc = roc_auc_score(binary_target, binary_pred[:, 1])
                self.logger.info("* AUC for class {}: {}".format(pos_class, binary_auc))

            ## (b) directly apply multi-label roc_auc-score
            one_hot_targets = torch.nn.functional.one_hot(torch.from_numpy(targets), self.args.class_num).numpy()
            multi_label_auc = roc_auc_score(one_hot_targets, outputs)
            self.logger.info("* multi-label AUC:{}".format(multi_label_auc))

            cls_report = classification_report(targets, outputs.argmax(-1))
            confusion_mat = confusion_matrix(targets, outputs.argmax(-1))
            self.logger.to_csv("confusion_matrix", confusion_mat)
            self.logger.to_csv("classification_report", classification_report(targets, outputs.argmax(-1), output_dict=True))

            self.logger.info("\n confusion matrix: \n {}".format(confusion_mat))
            self.logger.info("\n classification report: \n {}".format(cls_report))
            self.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=meter_set['top1'], top5=meter_set['top5']))

        # saving 
        if self.args.save_result:
            file_name = os.path.join(self.args.log_dir, self.args.save_result_name)
            self.save_checkpoint({"losses": meter_set['losses'].avg, "top1": meter_set['top1'].avg, "top5": meter_set['top5'].avg}, False, file_name)

        return meter_set['top1'].avg
