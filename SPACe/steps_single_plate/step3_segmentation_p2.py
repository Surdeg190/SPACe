import time
from tqdm import tqdm
from SPACe.SPACe.steps_single_plate._segmentation import SegmentationPartII
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

        seg_class = myclass(args)
        N = seg_class.args.N
        args.logger.info(f"Creating {N} tasks for Cellpaint Step 3 ...")
        
        bag = db.from_sequence(range(N), partition_size=args.partition_size)
        tasks_bag = bag.map(lambda ii: seg_class.run_single(ii))

        args.logger.info(f"Finished creating {N} tasks for Cellpaint Step 3 ...")


        return tasks_bag

