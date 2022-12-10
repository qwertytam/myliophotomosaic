import image_tools as itools
import mylio
import tree_tools as ttools
import random
import argparse
import sys

def main(mo_img : str,
        source_imgs : str,
        out_file : str = None,
        mo_iw : int = 100,
        src_res : tuple = (40, 40),
        disp_img_progress : bool = False,
        max_imgs : int = None,
        colour_space : str = 'RGB',
        src_save : str = None):

    # Load our main image
    face_im_arr = itools.load_img_as_arr(mo_img, colour_space)
    if disp_img_progress:
        itools.plt_img_from_arr(face_im_arr)

    # Store width and height of main image
    (height, width) = itools.get_img_arr_hw(face_im_arr)

    # Calculate target mosiac resolution
    ar = itools.get_img_arr_ar(face_im_arr)
    mo_res = (int(mo_iw*ar) ,mo_iw)

    # Create a template for the mosaic by index slicing the image, using the
    # step for rows and columns to divide the resolution
    mos_template = face_im_arr[::(height//mo_res[0]), ::(width//mo_res[1])]
    if disp_img_progress:
        itools.plt_img_from_arr(mos_template)

    # Number of images required to fill the mosaic
    num_imgs = mos_template[:,:, -1].size

    # Load in json for source image info from Mylio
    src_imgs_json = mylio.load_mylio_json(source_imgs)
    if max_imgs is None:
        max_imgs = len(src_imgs_json)

    # Create a list of all images as np arrays
    # Set size for mosaic images, loop through images and resize using
    # resize_image() function
    images = itools.get_resize_source_imgs(src_imgs_json, src_res,
                                           max_imgs, colour_space, src_save)
    
    # Let's have a look at one of the mosaic images
    if disp_img_progress:
        itools.plt_img_from_arr(images[random.randrange(0, len(images)-1)])

    # Convert list to np array
    images_array = itools.img_list_to_arr(images)

    # Get mean of colour values for each image
    # This will store the mean of each colour channel for each mosaic image
    image_values = itools.get_colour_mean(images_array)

    # Set KDTree for image_values
    tree = ttools.set_tree(image_values)

    # Find the best match for each 'pixel' of the template
    image_idx = ttools.find_best_match(mo_res, mos_template, tree, min(max_imgs, 40))
    
    # Generate the mosaic
    canvas = itools.generate_mosaic(mo_res, src_res, images, image_idx)
    canvas.show()

    if out_file:
        canvas.save(out_file)


if __name__ == "__main__":
    # Construct argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--mo_img",
                        help="Mosaic image file path and name",
                        required=True)

    parser.add_argument("--source_imgs",
                        help ="Text file with json copied from Mylio console for source images",
                        required=True)

    parser.add_argument("--out_file",
                        help ="File path and name to save output mosaic to",
                        required=False)

    help = "How many source images for the width of the mosaic; "
    help += "height determined by mosaic image aspect ratio"
    parser.add_argument("--mo_iw", help=help, required=False,
                        default=100, type=int)

    help = "Resolution for source images in output mosiac "
    help += "format `height width` e.g. `40 40`"
    parser.add_argument("--src_res", help=help, required=False,
                        default=(40, 40), nargs='+', type=int)

    parser.add_argument("--disp_img_progress",
                        help="Display images as we progress through the script.",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--max_imgs",
                        help="Optional, maximum number of source images to get, else use all",
                        type=int)

    parser.add_argument("--colour_space",
                        help="Optional, select colour space to use; options are default 'RGB' or 'HSV'",
                        type=str,
                        default='RGB')
    
    parser.add_argument("--src_save",
                        type=str)
    
    args = parser.parse_args()

    # Call main script
    main(args.mo_img,
        args.source_imgs,
        args.out_file,
        args.mo_iw,
        tuple(args.src_res),
        args.disp_img_progress,
        args.max_imgs,
        args.colour_space,
        args.src_save)