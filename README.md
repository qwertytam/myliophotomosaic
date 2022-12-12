# myliophotomosaic - making image mosaics from Mylio photo library
Python script(s) to create a photo mosaic. Photos sourced from Mylio photo
library management software.

## Contents
* [General Info](#general-info)
* [Requires](#requires)
* [Setup](#setup)
* [References](#references)

## General Info
The aim of this project is to create an image using a number of other images,
i.e. a mosaic. The image to be recreated, is pixelated to provide a template for
the mosaic images (one pixel = one image), then loop through each pixel in the
template and find the image with the closest RGB value. The images would then be
laid out side by side to recreate the image.

## Requires
See pip file.

## Setup/Installation
1. Clone the repo into your local environment
2. Make sure you have pipenv installed and run the following commands within the
project root folder
```
pipenv shell
pipenv install pipfile
```
This will initiate a pipenv virtual environment and use the pipfile to install
package dependencies

## To Use
1. Select photos in Mylio using keyboard and/or mouse to use in the Mosaic
2. Show the console in Mylio under `Help > Console`
3. Create a SQL view to return full paths for you (copy this all into one line).
This is for Windows - for Mac, replace the ‘\’ with a ‘/’.
```
old_sql CREATE VIEW localdirs as WITH RECURSIVE FoldersAndChildren(Id, UniqueHash, LocalName, FolderName) AS(VALUES(0, X'', '', '') UNION SELECT DISTINCT Folder.id, Folder.uniqueHash parentHash, FoldersAndChildren.localname || coalesce(nullif(Folder.localRootOrTemporaryPath, ''), Folder.localName) || '\', coalesce(nullif(Folder.localRootOrTemporaryPath, ''), Folder.localName) from Folder join FoldersAndChildren on FoldersAndChildren.UniqueHash = Folder.ParentFolderHash) SELECT * from FoldersAndChildren where Id <> 0
```
4. You can try this out using `SELECT * from localdirs limit 20`
5. If you need to redo the view for some reason, run `sql drop view localdirs`
to delete the previous one
6. Then run the following on some selected media to see if this data looks right:
```
selected > select LocalName, FileNameNoExt, LocalFileNameNoExt, MediaField(files, 1, 'format') Raw, MediaField(files, 2, 'format') NonRaw, MediaField(files, 3, 'format') Bundle, MediaField(files, 4, 'format') Display, MediaField(files, 4, 'format') XMP from media inner join localdirs on media.containingFolderHash = localdirs.uniqueHash where media.uniqueHash in ($_)
```
7.  If this data looks right, then run the same query as above, but add: `> json -v` at the end - with a space before that. You can just press the up arrow to edit the previous command, then scroll to the end. Now that will output a json array containing objects with several properties - FileNameNoExt, LocalFileNameNoExt, LocalName, RAW, NonRAW, DisplayImage, XMP & Bundle extensions.
8. The whole thing to get to your clipboard (apart from creating the view) is:
```
cls

selected > select LocalName, FileNameNoExt, LocalFileNameNoExt, MediaField(files, 1, 'format') Raw, MediaField(files, 2, 'format') NonRaw, MediaField(files, 3, 'format') Bundle, MediaField(files, 4, 'format') Display, MediaField(files, 4, 'format') XMP from media inner join localdirs on media.containingFolderHash = localdirs.uniqueHash where media.uniqueHash in ($_) > json -v

copy
```
9. Run script using `python main.py`, specifying at a minimum `--mosaic_fp_in`
and `--source`.

Example usage:
1. Reading in source images from Mylio json and saving tesserae to `./tesserae/`
```
python main.py --source mylio ./mosaic_templates/images.txt --tessera_res 40 40 --mosaic_tesserae_width 150 --mosaic_fp_in ./mosaic_templates/mo_tmp.jpg --mosaic_fp_out ./mosaics/mosaic.jpg --colour_space HSV --save_tesserae ./tesserae/
2. Reading in tesserae from previously run script
```
python main.py --source json ./tesserae/tesserae.json --tessera_res 40 40 --mosaic_tesserae_width 150 --mosaic_fp_in ./mosaic_templates/mo_tmp.jpg --mosaic_fp_out ./mosaics/mosaic.jpg --colour_space HSV 
```

## To dos
1. Ability to process raw images. Ideal solution is to use the `rawpy` library,
however local machine for some reason doesn't find the `rawpy` library using
`pip`.

## References & Inspiration:
- [Mylio Support Forum Topic](https://forum.mylio.com/t/list-of-file-path-and-names-for-search-or-filter-results/6093/4)
- [facemo - making image mosaics](https://github.com/MstrFunkBass/facemo)
- [Google Python style guide](https://google.github.io/styleguide/pyguide.html)

## Thanks
- To Mylio users and support for providing invaluable support for SQL and other tips