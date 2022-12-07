import image_tools as itools
import mylio
import tree_tools as ttools
import random
import argparse

def main(main_img : str,
        source_imgs : str,
        out_file : str = None,
        target_res_main : tuple = (100, 100),
        target_res_src : tuple = (40, 40),
        disp_img_progress : bool = False,
        max_imgs : int = 100):

    # Load our main image
    face_im_arr = itools.load_img_as_arr(main_img)
    if disp_img_progress:
        itools.plt_img_from_arr(face_im_arr)

    # Store width and height of main image
    height = face_im_arr.shape[0]
    width = face_im_arr.shape[1]

    # Create a template for the mosaic by index slicing the image, using the
    # step for rows and columns to divide the resolution
    mos_template = face_im_arr[::(height//target_res_main[0]), ::(width//target_res_main[1])]
    if disp_img_progress:
        itools.plt_img_from_arr(mos_template)

    # Number of images required to fill the mosaic
    num_imgs = mos_template[:,:, -1].size

    # Load in json for source image info from Mylio
    src_imgs_json = mylio.load_mylio_json(source_imgs)

    # Create a list of all images as np arrays
    # Set size for mosaic images, loop through images and resize using resize_image() function
    images = itools.get__resize_source_imgs(src_imgs_json, target_res_src, max_imgs)

    # Let's have a look at one of the mosaic images
    if disp_img_progress:
        itools.plt_img_from_arr(images[random.randrange(0, len(images)-1)])

    # Convert list to np array
    images_array = itools.img_list_to_arr(images)

    # Get mean of RGB values for each image
    # This will store the mean Red, Green and Blue values of each mosaic image
    # image_values = np.apply_over_axes(np.mean, images_array, [1,2]).reshape(len(images),3)
    image_values = itools.get_rgb_mean(images_array)

    # Set KDTree for image_values
    tree = ttools.set_tree(image_values)

    # Find the best match for each 'pixel' of the template
    image_idx = ttools.find_best_match(target_res_main, mos_template, tree, min(max_imgs, 40))
    
    # Generate the mosaic
    canvas = itools.generate_mosaic(target_res_main, target_res_src, images, image_idx)
    canvas.show()

    if out_file:
        canvas.save(out_file)


if __name__ == "__main__":
    # Construct argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--main_img",
                        help="Main image file path and name",
                        required=True)

    parser.add_argument("--source_imgs",
                        help ="Text file with json copied from Mylio console",
                        required=True)

    parser.add_argument("--out_file",
                        help ="File path and name to save output to",
                        required=False)

    help = "target resolution as number of images to make up final image in "
    help += "format `height width` e.g. `100 100`"
    parser.add_argument("--target_res_main", help=help, required=False,
                        default=(100, 100), nargs='+', type=int)

    help = "resolution for mosaic images in "
    help += "format `height width` e.g. `40 40`"
    parser.add_argument("--target_res_src", help=help, required=False,
                        default=(40, 40), nargs='+', type=int)

    parser.add_argument("--disp_img_progress",
                        help="Display images as we progress through the script.",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--max_imgs",
                        help="Maximum number of source images to get",
                        default=100,
                        type=int,
                        required=False)

    args = parser.parse_args()

    # Call main script
    main(args.main_img,
        args.source_imgs,
        args.out_file,
        tuple(args.target_res_main),
        tuple(args.target_res_src),
        args.disp_img_progress,
        args.max_imgs)