import os
import time
from tqdm import tqdm
from pathlib import WindowsPath

import numpy as np
import multiprocessing as mp

from SPACe.SPACe.steps_single_plate.step0_args import Args
from SPACe.SPACe.utils.shared_memory import MyBaseManager, TestProxy
from SPACe.SPACe.steps_single_plate._segmentation import SegmentationPartII

from dask import delayed, compute
import dask.bag as db

def chunkify(lst, n):
    return list([lst[i::n] for i in range(n)])

def step3_main_run_loop(args, myclass=SegmentationPartII):
    """
    Main function for cellpaint step III which:
        1) Corrects and syncs the Nucleus and Cytoplasm masks from Cellpaint stepII.
        2) Generates Nucleoli and Mitocondria masks using Nucleus and Cytoplasm masks, respectively.

        In what follows each mask is referred to as:
        Nucleus mask:      w1_mask
        Cyto mask:         w2_mask
        Nucleoli mask:     w3_mask
        Mito mask:         w5_mask

        It saves all those masks as separate png files into:
        if args.mode.lower() == "full":
            self.args.masks_path_p3 = args.main_path / args.experiment / "Step2_MasksP2"
    """
    print(args.w3_intensity_bounds)
    print("Cellpaint Step 3: \n"
          "3-1) Matching segmentation of Nucleus and Cytoplasm \n"
          "3-2) Thresholding segmentation of Nucleoli and Mitocondria ...")
    s_time = time.time()
    if args.mode == "test":
        inst = myclass(args)
        N = inst.args.N
        for ii in tqdm(range(N)):
            inst.run_single(ii)
    else:
        """
        We have to Register the ThresholdingSegmentation class object as well as its attributes as shared using:
        https://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class

        Also do not use numpy arrays in __init__ method or in the header/public section of the class,
        because python's multiprocessing module can't pickle them!!!!

        Try to use lists, dictionaries, and tuples instead.
        """
        # MyManager = MyBaseManager()
        # MyManager.register(myclass.__name__, myclass, TestProxy)
        # with MyManager as manager:
        #     inst = getattr(manager, myclass.__name__)(args)
        #     N = inst.args.N
        #     with mp.Pool(processes=inst.num_workers) as pool:
        #         for _ in pool.imap_unordered(inst.run_multi, np.arange(N)):
        #             pass
        #     # print(N)

        seg_class = myclass(args)
        N = seg_class.args.N
        args.logger.info(f"Creating {N} tasks for Cellpaint Step 3 ...")
        
        bag = db.from_sequence(range(N), partition_size=50)
        tasks_bag = bag.map(lambda ii: seg_class.run_single(ii))

        args.logger.info(f"Finished creating {N} tasks for Cellpaint Step 3 ...")


        # ranger = np.arange(N)
        # chunked_ranger = chunkify(ranger, 40)
        # i = 0
        # for chunk in chunked_ranger:
        #     tasks = []
        #     print(f"Running Cellpaint Step 3 for {i} chunk ...")
        #     for ii in chunk:
        #         tasks.append(delayed(seg_class.run_single)(ii))
        #     i += 1
        #     compute(*tasks)
        #     print(f"Finished Cellpaint Step 3 for {i} chunk ...")

        return tasks_bag
        print(f"Finished Cellpaint step 3 in: {(time.time()-s_time)/3600} hours\n")

