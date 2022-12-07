from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from pillow_heif import register_heif_opener

register_heif_opener()


def load_img_as_arr(source : str) -> np.ndarray:
    """Gets rgb data for image.
    
    Opens an image from specified source and returns a numpy array with
    image rgb data.

    Args:
        source: File path and name to image

    Returns:
        A numpy array with image rgb data. Array is three dimensional by
        height, width, and number of channels (i.e. three)
    
    Raises:
        None
    """
    with Image.open(source) as im:
        im_arr = np.asarray(im)
    return im_arr


def resize_img(img : Image, size : tuple) -> np.ndarray:
    """Resize image
    
    Takes an image and resizes to a given size (width, height) as passed to the
    size parameter. Uses Lanczos filter with (0.5, 0.5) centering method.
    
    Args:
        img: Image to resize
        size: Integer tuple for new image size

    Returns:
        Numpy three-dimensional array of resized image
    
    Raises:
        None
    """
    resz_img = ImageOps.fit(img, size, Image.Resampling.LANCZOS,
                            centering=(0.5, 0.5))
    return np.array(resz_img)


def img_from_arr(image : Image) -> np.ndarray:
    """Returns array from image
    
    Args:
        image: Image to get array of

    Returns:
        Numpy array of provided image
    
    Raises:
        None
    """
    return Image.fromarray(image)


def img_list_to_arr(images : list) -> np.ndarray:
    """Returns list of images as array
    
    Args:
        images: List of images

    Returns:
        Numpy array of provided image list
    
    Raises:
        None
    """
    return np.asarray(images)


def get_img_arr_hw(image : np.ndarray) -> tuple:
    """Get image array heigh and width

    Args:
        image: Numpy array of image to get heigh and width of

    Returns:
        Integer tuple of (heigh, width)
    
    Raises:
        None
    """
    return (image.shape[0], image.shape[1])


def get_img_arr_ar(image : np.ndarray) -> float:
    """Get image array aspect ratio (height to width)

    Args:
        image: Numpy array of image to get aspect ratio of

    Returns:
        Float of aspect ratio
    
    Raises:
        None
    """
    (h, w) = get_img_arr_hw(image)
    return h/w


def plt_img(image : Image):
    """Plot image
    
    Plots image using matplotlib library

    Args:
        image: PIL Image class

    Returns:
        None
    
    Raises:
        None
    """
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.show()


def plt_img_from_arr(image_arr : np.ndarray):
    """Plot image from numpy array
    
    Plots image using matplotlib library

    Args:
        image: PIL Image class

    Returns:
        None
    
    Raises:
        None
    """
    plt_img(img_from_arr(image_arr))


def get_rgb_mean(imgs_array) -> np.ndarray:
    """Get mean of RGB values for each image in list
    
    Args:
        imgs_array: Input array of images to get rgb mean of

    Returns:
        numpy array n x 3 dimensions, where n is the number of images, and each
        dimension is the mean for each rgb channel

    Raises:
        None
    """
    # Find means
    mns = np.apply_over_axes(np.mean, imgs_array, [1,2])

    # Result array is n x 1 x 1 x 3 dimensions, so reshape to n x 3
    return mns.reshape(len(imgs_array),3)


def get__resize_source_imgs(src_json : dict,
                            size : tuple,
                            max_imgs : int = None) -> list:
    """Gets list of images from json list
    
    Iterates through dictionary to get file path, checks if path is valid,
    checks if image is valid (a three-dimensional array), and returns list
    of valid images

    Args:
        src_json: JSON dictionary, where each entry has the key 'FullPath'
        that provides a file path and name for the image of interest

        max_imgs: optional, if given, then only gets this number of images

    Returns:
        A list of the valid images as three-dimensional numpy arrays
    
    Raises:
        None
    """
    images = []
    skipped = 0
    skip_print_limit = 0
    num_src_imgs = len(src_json)
    print_int = 25
    for i, src in enumerate(src_json):
        # Check if path points to a valid file; if not skip
        try:
            im = load_img_as_arr(src['FullPath'])
            if im.ndim != 3:
                skipped += 1
                if skipped < skip_print_limit:
                    print(f'\n{(i+1):,}: im.ndim != 3\n{src}\n')
                continue

            im = resize_img(img_from_arr(im), size)
            # Some images have a fourth channel? Remove them
            if im.shape[2] != 3:
                skipped += 1
                if skipped < skip_print_limit:
                    print(f'\n{(i+1):,}: im.shape[2] != 3\n{src}\n')
                continue

            images.append(im)

        except FileNotFoundError as e:
            skipped += 1
            if skipped < skip_print_limit:
                print(f'\n{(i+1):,}: {src}\n{e}\n')

        except UnidentifiedImageError as e:
            skipped += 1
            if skipped < skip_print_limit:
                print(f'\n{(i+1):,}: {src}\n{e}\n')

        except OSError as e:
            skipped += 1
            if skipped < skip_print_limit:
                print(f'\n{(i+1):,}: {src}\n{e}\n')

        if (i+1) % print_int == 0:
            print(f'Processed {(i+1):,} ' +
                f'found {len(images):,} ' +
                f'skipped {skipped:,} ' +
                f'{(i+1)/num_src_imgs:.2%}  complete',
                end='\r')
        
        if max_imgs is not None and len(images) >= max_imgs:
            print(f'\nFound max images; stopping early at {len(images):,}')
            break

    print(f'\nProcessed {(i+1):,} ' +
        f'found {len(images):,} ' +
        f'skipped {skipped:,} ' +
        f'{(i+1)/(num_src_imgs):.2%} complete')

    return images

def generate_mosaic(mo_res : tuple,
                    src_res : tuple,
                    images : list, 
                    image_idx : np.ndarray,
                    ) -> Image:
    """Generate mosaic

    Loop through the best match indices, retrieve the matching image and add it
    to the mosaic. The images are offset by the mosaic tile size so that they
    don't overlap.

    Args:
        mo_res: Target resolution for the mosaic

        src_res: Target resolution of the source images

        images: List of source images

        image_idx: Array with best source image match indicies
    
    Return:
        Mosaic
    
    Raises:
        None
    """
    
    canvas = Image.new('RGB',
                        (src_res[1]*mo_res[1],
                         src_res[0]*mo_res[0])
                    )

    for i in range(mo_res[0]):
        for j in range(mo_res[1]):
            arr = images[image_idx[i, j]]
            x, y = j*src_res[1], i*src_res[0]
            im = img_from_arr(arr)
            canvas.paste(im, (x,y))

    return canvas