import json
import re


def load_mylio_json(source : str) -> json:
    """Gets json image data from Mylio output text file.
    
    Loads json data dump from Mylio stored in a plain text file. See repo
    setup / installation instructions for how to get data out of Mylio.

    Args:
        source: File path and name to text file containing data

    Returns:
        A json object of the data in the file along with an additional entry
        'FullPath' for the full file path, name and extension
    
    Raises:
        None
    """
    with open(source, 'r') as f:
        ft = f.read()
        # Use regex to look for json data and assume data we want is the first
        # match we find
        ft_json = re.findall(r'\[.*\]', ft)[0]
        src_imgs_json = json.loads(ft_json)

    for src in src_imgs_json:
        ext = ''
        if src['NonRaw'] != 'null':
            ext = src['NonRaw']
        else:
            ext = src['Raw']
        src['FullPath'] = src['LocalName'] + src['LocalFileNameNoExt'] + '.' + ext

    return src_imgs_json