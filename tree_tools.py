from scipy import spatial
import random
import numpy as np

def set_tree(values):
    """Set KDTree for given values

    Args:
        values: Values to set on tree
    
    Returns:
        KDTree
    
    Raises:
        None
    """
    return spatial.KDTree(values)


def find_best_match(target_res : tuple,
                    mos_template : np.ndarray,
                    tree,
                    rand_choice : int=1) -> np.ndarray:
    """Find the tree index for the best image match
    
    Loop through each pixel in the low-res reference image and find the index
    of the image that has the closest colour match to the mosaic template.
    The 'match' variable returns a list of the closest rand_choice matches
    and one is randomly chosen - this is to reduce repeat images for similar
    colours that are close to each other. This number would need to be smaller
    than the number of images available.

    Args:
        target_res: Target resolution as integer tuple of number of source
            images in format (height, width)        
        mos_template: The mosaic template image as a array
        tree: Spatial KDTree that contains the source image rgb means
        rand_choice: How many of the closest match images to randomly choose
            from
    
    Returns:
        Numpy two-dimensional array, where each entry corresponds to a tree
        index for each source image for the mosaic

    Raises:
        None 
    """
    image_idx = np.zeros(target_res, dtype=np.uint32)
    for i in range(target_res[0]):
        for j in range(target_res[1]):
            template = mos_template[i, j]
            match = tree.query(template, k=rand_choice)
            pick = random.randint(0, rand_choice-1)
            image_idx[i, j] = match[1][pick]
    return image_idx