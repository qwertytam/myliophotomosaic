import image_tools as itools
import mylio
import tree_tools as ttools
import random
import argparse
import sys
import numpy as np

def main(mosaic_fp_in : str,
        source_mylio_json : str,
        source_tesserae_json : str = None,
        mosaic_fp_out : str = None,
        mosaic_tesserae_width : int = 100,
        tessera_res : tuple = (40, 40),
        disp_img_progress : bool = False,
        max_source_imgs : int = None,
        colour_space : str = 'RGB',
        save_tesserae : str = None):
    """Main script.
    
    Main script to create mosaic from set of tesserae (plural of tessera).

    Args:
        mosaic_fp_in: File path and name string pointing towards image
            template for final mosaic
        source_mylio_json: File path and name string pointing towards output
            json from Mylio image library, where the json contains info
            for tesserae file paths.
        source_tesserae_json: File path string pointing towards previously saved
            tesserae output. Optional, default is `None`.
        mosaic_fp_out: File path and name string for where to save final mosaic.
            Optional, default is `None` with the output not saved.
        mosaic_tesserae_width: Integer number for how many tessera wide the
            final mosaic should be. Height determined by aspect ratio of mosaic
            template. Optional, default is 100.
        tessera_res: Integer tuple `(height, width)` for tessera resolution in
            pixles. Optional, default is (40, 40)
        disp_img_progress: Boolean True/False to display images during script
            progress. Optional, default is False.
        max_source_imgs: Integer specifying maximum number of source images to
            read in. Optional, default is `None` so that all available are used.
        colour_space: String specifying colour mode. Passed to Pillow library
            functions. Default is `RGB`, with most likely other option being
            `HSV`.
        save_tesserae: File path to directory to save tesserae. Option, default
            is `None` so that they are not saved.

    Returns:
        None
    
    Raises:
        None
    """

    # Load our main image
    face_im_arr = itools.load_img_as_arr(mosaic_fp_in, colour_space)
    if disp_img_progress:
        itools.plt_img_from_arr(face_im_arr)

    # Store width and height of main image
    (height, width) = itools.get_img_arr_hw(face_im_arr)

    # Calculate target mosiac resolution
    ar = itools.get_img_arr_ar(face_im_arr)
    mosaic_res = (int(mosaic_tesserae_width*ar) ,mosaic_tesserae_width)

    # Create a template for the mosaic by index slicing the image, using the
    # step for rows and columns to divide the resolution
    mos_template = face_im_arr[::(height//mosaic_res[0]),
                               ::(width//mosaic_res[1])]
    if disp_img_progress:
        itools.plt_img_from_arr(mos_template)

    # Number of images required to fill the mosaic
    num_imgs = mos_template[:,:, -1].size

    # Load in json for source image info from Mylio
    src_imgs_json = mylio.load_mylio_json(source_mylio_json)
    if max_source_imgs is None:
        max_source_imgs = len(src_imgs_json)

    # Create a list of all images as np arrays
    # Set size for mosaic images, loop through images and resize using
    # resize_image() function
    # if saved_tesserae is not None:
    #     images = 1
    #     image_means = itools.get_means(max_source_imgs, saved_tesserae)
    #     image_values = np.asarray(image_means)

    if True:
        images, image_values = itools.get_resize_source_imgs(
            src_imgs_json, tessera_res, max_source_imgs, colour_space,
            save_tesserae)

        # Let's have a look at one of the mosaic images
        if disp_img_progress:
            itools.plt_img_from_arr(images[random.randrange(0, len(images)-1)])

        # Convert list to np array
        # images_array = itools.img_list_to_arr(images)

        # Get mean of colour values for each image
        # This will store the mean of each colour channel for each mosaic image
        # image_values = itools.get_colour_mean(images_array)


    # Set KDTree for image_values
    tree = ttools.set_tree(image_values)

    # Find the best match for each 'pixel' of the template
    image_idx = ttools.find_best_match(mosaic_res, mos_template, tree,
                                       min(max_source_imgs, 40))
    
    # Generate the mosaic
    canvas = itools.generate_mosaic(mosaic_res, tessera_res, images, image_idx)
    canvas.show()

    if mosaic_fp_out:
        canvas.save(mosaic_fp_out)


if __name__ == "__main__":
    # Construct argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--mosaic_fp_in',
                        help="Mosaic image file path and name",
                        type=str,
                        required=True)

    parser.add_argument('--source_mylio_json',
                        help="Text file with json copied from Mylio " +
                        "console for source images",
                        type=str,
                        required=False)
            
    parser.add_argument('--saved_tesserae',
                        help="File path to saved tesserae from preivous run" +
                        "of this script",
                        required=False,
                        type=str)
    
    parser.add_argument('--mosaic_fp_out',
                        help ="File path and name to save output mosaic to",
                        required=False)

    help = "How many source images for the width of the mosaic; "
    help += "height determined by mosaic image aspect ratio"
    parser.add_argument('--mosaic_tesserae_width',
                        help=help, required=False,
                        default=100, type=int)

    help = "Resolution for source images in output mosiac "
    help += "format `height width` e.g. `40 40`"
    parser.add_argument('--tessera_res',
                        help=help, required=False,
                        default=(40, 40), nargs='+', type=int)

    parser.add_argument('--disp_img_progress',
                        help="Display images as we progress through the script.",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('--max_source_imgs',
                        help="Optional, maximum number of source images to " +
                        "get, else use all",
                        type=int)

    parser.add_argument('--colour_space',
                        help="Optional, select colour space to use; options" +
                        "are default 'RGB' or 'HSV'",
                        type=str,
                        default='RGB')
    
    parser.add_argument('--save_tesserae',
                        help="File path to save tesserae to",
                        type=str,
                        required=False)
    
    args = parser.parse_args()

    # Call main script
    main(args.mosaic_fp_in,
        args.source_mylio_json,
        args.saved_tesserae,
        args.mosaic_fp_out,
        args.mosaic_tesserae_width,
        tuple(args.tessera_res),
        args.disp_img_progress,
        args.max_source_imgs,
        args.colour_space,
        args.save_tesserae)