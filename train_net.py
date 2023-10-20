# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jialian Wu from https://github.com/facebookresearch/Detic/blob/main/train_net.py
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer

from third_party.CenterNet2.detectron2.checkpoint import DetectionCheckpointer
from third_party.CenterNet2.detectron2.config import get_cfg


from cdkm.config import add_centernet_config
from cdkm.custom_solver import build_custom_optimizer
from cdkm.data.custom_build_augmentation import build_custom_augmentation
from cdkm.data.custom_dataset_dataloader import build_custom_train_loader
from cdkm.data.custom_dataset_mapper import CustomDatasetMapper
from cdkm.evaluation.eval import CDKMVGEvaluator
from third_party.CenterNet2.detectron2.data import DatasetMapper, build_detection_test_loader, MetadataCatalog
from third_party.CenterNet2.detectron2.engine import default_setup, default_argument_parser, launch, \
    PeriodicCheckpointer
from third_party.CenterNet2.detectron2.evaluation import inference_on_dataset, print_csv_format
from third_party.CenterNet2.detectron2.modeling import build_model
from third_party.CenterNet2.detectron2.solver import build_optimizer, build_lr_scheduler
from third_party.CenterNet2.detectron2.utils import comm
from third_party.CenterNet2.detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, \
    EventStorage
from third_party.CenterNet2.detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')



logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == 'vg':
            evaluator = CDKMVGEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise NotImplementedError('We have not implemented the evaluator for {}'.format(evaluator_type))
            
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = CustomDatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    data_loader = build_custom_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.output_dir_name:
        cfg.OUTPUT_DIR = args.output_dir_name
    logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), color=False, name="cdkm")
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    do_train(cfg, model, resume=args.resume)
    return


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--output-dir-name", type=str, default='./output/CDKM')
    args.add_argument("--num-gpus-per-machine", type=int, default=8)
    args.add_argument("--test-task", type=str, default='', help="Choose a task to have CDKM perform")
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        raise NotImplementedError('Use train_deepspeed.py for multi-node training')
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus_per_machine,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
