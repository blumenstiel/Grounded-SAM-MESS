# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Code based on the MaskFormer Training Script.
"""
import os

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.utils.logger import setup_logger

import mess.datasets
from mess.evaluation.sem_seg_evaluation import MESSSemSegEvaluator
from model import grounded_sam_model


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        assert evaluator_type == "sem_seg", f"Got {evaluator_type} for {dataset_name}. Only sem_seg is supported."
        evaluator = MESSSemSegEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                    )
        return evaluator


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

        return res
    else:
        print("Only evaluation is supported")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
