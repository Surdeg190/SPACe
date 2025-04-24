import os
from tqdm import tqdm
from pathlib import WindowsPath
from SPACe.SPACe.steps_single_plate.step0_args import Args
from SPACe.SPACe.steps_single_plate._segmentation import SegmentationPartI
import dask.bag as db

from cellpose import models
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def create_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return models.Cellpose(gpu=True, model_type=args.cellpose_model_type, device=device)

def step2_main_run_loop(args):
    """
    Main function for cellpaint step 2:
        It performs segmentation of nucleus and cytoplasm channels,
        (99% of the time,they are the first and the second channel of each image)
        using the cellpose python package.

        It saves the two masks as separate png files into:
        self.args.step1_save_path = args.main_path / args.experiment / "Step1_MasksP1"
    """
    args.logger.info("Cellpaint Step 2: Cellpose segmentation of Nucleus and Cytoplasm ...")

    cellpose_model = create_model(args)
    seg_class = SegmentationPartI(args, cellpose_model=cellpose_model)
    N = seg_class.args.N
    
    args.logger.info(f"Creating {N} tasks for Cellpaint Step 2 ...")
    data = list(zip(seg_class.args.img_channels_filepaths, seg_class.args.img_filename_keys))

    bag = db.from_sequence(data, partition_size=50)
    tasks = bag.map(lambda x: seg_class.run_single(x[0], x[1]))
    return tasks    
    


if __name__ == "__main__":
    camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
    camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
    camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
    camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

    main_path = WindowsPath(camii_server_flav)
    exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

    args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
    step2_main_run_loop(args)
