import time
import os

import torch
from tqdm import tqdm

import _init_paths

from datasets.dataset_factory import get_dataset

from graspnet_6dof.options.train_options import TrainOptions
from graspnet_6dof.data.base_dataset import collate_fn
from graspnet_6dof.models import create_model
from graspnet_6dof.utils.writer import Writer
from graspnet_6dof.options.test_options import TestOptions


class DataLoader:
    """multi-threaded data loading"""
    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.num_objects_per_batch,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


class TrainOptionsMerge(TrainOptions):
    def __init__(self):
        super().__init__()
        # add the dataset arguments
        self.parser.add_argument('--dataset', default='ps_grasp')
        self.parser.add_argument('--unitTest', action="store_true")
        self.parser.add_argument('--no_collide_filter', action="store_true",
                            help="Train also on the grasps that cause collision")
    
    @staticmethod
    def add_opts(opt):
        opt.task = "vae_graspnet"
        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')

        #### Below is useless, but need to get one
        opt.kpt_type = "box"
        opt.min_open_width = None
        opt.open_width_canonical = None

        return opt


class TestOptionsMerge(TestOptions):
    def __init__(self):
        super().__init__()
        # add the dataset arguments
        self.parser.add_argument('--dataset', default='ps_grasp')
        self.parser.add_argument('--unitTest', action="store_true")
        self.parser.add_argument('--no_collide_filter', action="store_true",
                            help="Train also on the grasps that cause collision")
    
    @staticmethod
    def add_opts(opt):
        opt.task = "vae_graspnet"
        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')

        #### Below is useless, but need to get one
        opt.kpt_type = "box"
        opt.min_open_width = None
        opt.open_width_canonical = None

        return opt


def run_test(epoch=-1, name=""):
    """Test the 6DOF GraspNet on the PS_grasp dataset.
    It is modified from: https://github.com/jsll/pytorch_6dof-graspnet/blob/master/test.py
    """
    print('Running Test')
    opt = TestOptionsMerge().parse()
    opt = TestOptionsMerge.add_opts(opt)
    opt.serial_batches = True  # no shuffle
    opt.name = name
    #### NOTE: This part is changed to the ps_grasp
    #dataset = DataLoader(opt)
    Dataset = get_dataset(opt.dataset, opt.task)
    dataset = Dataset(opt, 'test')
    dataset = DataLoader(opt, dataset)
    ####
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


def main():
    opt = TrainOptionsMerge().parse()
    opt = TrainOptionsMerge.add_opts(opt)
    if opt == None:
        return

    #### NOTE: This part is changed to the ps_grasp
    #dataset = DataLoader(opt)
    Dataset = get_dataset(opt.dataset, opt.task)
    dataset = Dataset(opt, 'train')
    dataset = DataLoader(opt, dataset)
    ####

    dataset_size = len(dataset) * opt.num_grasps_per_object
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                loss_types = []
                if opt.arch == "vae":
                    loss = [
                        model.loss, model.kl_loss, model.reconstruction_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "kl_loss", "reconstruction_loss",
                        "confidence loss"
                    ]
                elif opt.arch == "gan":
                    loss = [
                        model.loss, model.reconstruction_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "reconstruction_loss", "confidence_loss"
                    ]
                else:
                    loss = [
                        model.loss, model.classification_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "classification_loss", "confidence_loss"
                    ]
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data,
                                            loss_types)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size,
                                 loss_types)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest', epoch)
            model.save_network(str(epoch), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            print("testing on the epoch: {}...".format(epoch))
            acc = run_test(epoch, name=opt.name)
            writer.plot_acc(acc, epoch)

    writer.close()


if __name__ == '__main__':
    main()
