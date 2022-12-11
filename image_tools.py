from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from pillow_heif import register_heif_opener
import json
import sys

register_heif_opener()


def load_img_as_arr(source : str, colour_space : str = 'RGB') -> np.ndarray:
    """Gets rgb data for image.
    
    Opens an image from specified source and returns a numpy array with
    image rgb data.

    Args:
        source: File path and name to image
        colour_space: Colour space mode to load image array as. See Pillow
        'Modes' for further info. Default used here is 'RGB'

    Returns:
        A numpy array with image colour channel data. Array is three dimensional
        by height, width, and number of channels (i.e. three)
    
    Raises:
        None
    """
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != 'RGB':
            im = im.convert('RGB')
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


def get_colour_mean(imgs_array : np.ndarray) -> np.ndarray:
    """Get mean of color channel values for each image in list
    
    Args:
        imgs_array: Input array of images to get color channel mean of

    Returns:
        numpy array n x m dimensions, where n is the number of images, and each
        dimension is the mean for each m colour channel

    Raises:
        None
    """
    num_channels = imgs_array.shape[imgs_array.ndim - 1]
    mns = np.apply_over_axes(np.mean, imgs_array, [1,2])

    # Result array is n x 1 x 1 x m dimensions, so reshape to n x m
    return mns.reshape(len(imgs_array), num_channels)


def get_resize_source_imgs(src_json : dict,
                           size : tuple,
                           max_source_imgs : int = None,
                           colour_space : str = 'RGB',
                           save_tesserae : str = None) -> list:
    """Gets list of images from json list
    
    Iterates through dictionary to get file path, checks if path is valid,
    checks if image is valid (a three-dimensional array), and returns list
    of valid images

    Args:
        src_json: JSON dictionary, where each entry has the key 'FullPath'
            that provides a file path and name for the image of interest
        size: Resolution for source images in output mosiac; format
            `height width` e.g. `40 40`"
        max_source_imgs: optional, if given, then only gets this number of images
            colour_space: Colour space mode to load image array as. See Pillow
            'Modes' for further info. Default used here is 'RGB'
        save_tesserae: Directory to save resized source images to. Uses index
            number for file name e.g. 123.jpg

    Returns:
        A list of the valid images as three-dimensional numpy arrays
    
    Raises:
        None
    """
    src_ims = []
    src_ims_cc_means = []
    src_ims_meta = []

    skipped = 0
    skip_print_limit = 10
    print_int = 10

    num_src_ims = len(src_json)

    for i, src in enumerate(src_json):
        idx = i - skipped
        # Check if path points to a valid file; if not skip
        try:
            im = load_img_as_arr(src['FullPath'], colour_space)

        except FileNotFoundError as e:
            skipped += 1
            if skipped < skip_print_limit:
                print(f'\n{(i+1):,}: {src}\n{e}\n')
            continue

        except UnidentifiedImageError as e:
            skipped += 1
            if skipped < skip_print_limit:
                print(f'\n{(i+1):,}: {src}\n{e}\n')
            continue            
        
        # Get resized image array and append to our list
        im = resize_img(img_from_arr(im), size)
        src_ims.append(im)
        
        num_channels = im.shape[im.ndim - 1]
        im_mn = np.apply_over_axes(np.mean, im, [0,1]).reshape(-1)
        src_ims_cc_means.append(im_mn)
        
        # Build resized source image file path and name
        sfp = save_tesserae + f'{idx:06d}' + '.jpg'

        # Construct dictionary of image details and add to list
        im_meta = {
            'idx': idx,
            'cc_means': im_mn.tolist(),
            'source_file_path': sfp,
            'original_file_path': src['FullPath'],
        }
        src_ims_meta.append(im_meta)

        if save_tesserae is not None:
            canvas = Image.new(colour_space, size)
            canvas.paste(img_from_arr(im))
            if canvas.mode != 'RGB':
                canvas = canvas.convert('RGB')
            canvas.save(sfp)

        # Progress printing
        if (i+1) % print_int == 0:
            print(f'Processed {(i+1):,} ' +
                f'found {len(src_ims):,} ' +
                f'skipped {skipped:,} ' +
                f'{(i+1)/num_src_ims:.2%}  complete',
                end='\r')
        
        # Stop if hit max image count
        if max_source_imgs is not None and len(src_ims) >= max_source_imgs:
            print(f'\nFound max images; stopping early at {len(src_ims):,}')
            break

    # End of loop status
    print(f'\nProcessed {(i+1):,} ' +
        f'found {len(src_ims):,} ' +
        f'skipped {skipped:,} ' +
        f'{(i+1)/(num_src_ims):.2%} complete')
    
    if save_tesserae is not None:
        with open(save_tesserae + 'tesserae.json', 'w') as outfile:
            json.dump(src_ims_meta, outfile)

    src_ims = img_list_to_arr(src_ims)
    return src_ims, np.asarray(src_ims_cc_means)


def generate_mosaic(mosaic_res : tuple,
                    tessera_res : tuple,
                    images : list, 
                    image_idx : np.ndarray,
                    colour_space : str = 'RGB',
                    ) -> Image:
    """Generate mosaic

    Loop through the best match indices, retrieve the matching image and add it
    to the mosaic. The images are offset by the mosaic tile size so that they
    don't overlap.

    Args:
        mosaic_res: Target resolution for the mosaic
        tessera_res: Target resolution of the source images
        images: List of source images
        image_idx: Array with best source image match indicies
        colour_space: Colour space mode to load image array as. See Pillow
            'Modes' for further info. Default used here is 'RGB'
    
    Return:
        Mosaic
    
    Raises:
        None
    """
    canvas = Image.new(colour_space,
                       (tessera_res[1]*mosaic_res[1],
                        tessera_res[0]*mosaic_res[0])
                       )

    for i in range(mosaic_res[0]):
        for j in range(mosaic_res[1]):
            arr = images[image_idx[i, j]]
            x, y = j*tessera_res[1], i*tessera_res[0]
            im = img_from_arr(arr)
            canvas.paste(im, (x,y))

    return canvas

def get_means(max_source_imgs, saved_tesserae):
    with open(saved_tesserae, 'r') as f:
        ft = f.read()
        src_imgs_json = json.loads(ft)

    img_means = []
    for i, src in enumerate(src_imgs_json):
        if 'channel_means' in src:
            img_means.append(src['channel_means'])
        # else:
            # print(f'uh oh for {src}')
    
    return img_means