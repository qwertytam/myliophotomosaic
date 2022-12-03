# myliophotomosaic

Python script(s) to create a photo mosaic. Photos sourced from Mylio photo library management software.

## To Use
1. Select photos in Mylio using keyboard and/or mouse to use in the Mosaic
2. Show the console in Mylio under `Help > Console`
3. Create a SQL view to return full paths for you (copy this all into one line). This is for Windows - for Mac, replace the ‘\’ with a ‘/’.
```
old_sql CREATE VIEW localdirs as WITH RECURSIVE FoldersAndChildren(Id, UniqueHash, LocalName, FolderName) AS(VALUES(0, X'', '', '') UNION SELECT DISTINCT Folder.id, Folder.uniqueHash parentHash, FoldersAndChildren.localname || coalesce(nullif(Folder.localRootOrTemporaryPath, ''), Folder.localName) || '\', coalesce(nullif(Folder.localRootOrTemporaryPath, ''), Folder.localName) from Folder join FoldersAndChildren on FoldersAndChildren.UniqueHash = Folder.ParentFolderHash) SELECT * from FoldersAndChildren where Id <> 0
```
4. You can try this out using `SELECT * from localdirs limit 20`
5. If you need to redo the view for some reason, run `sql drop view localdirs` to delete the previous one
6. Then run the following on some selected media to see if this data looks right:
```
selected > select LocalName, filenamenoext, MediaField(files, 1, 'format') Raw, MediaField(files, 2, 'format') NonRaw, MediaField(files, 3, 'format') Bundle, MediaField(files, 4, 'format') Display, MediaField(files, 4, 'format') XMP from media inner join localdirs on media.containingFolderHash = localdirs.uniqueHash where media.uniqueHash in ($_)
```
7.  If this data looks right, then run the same query as above, but add: `> json -v` at the end - with a space before that. You can just press the up arrow to edit the previous command, then scroll to the end. Now that will output a json array containing objects with several properties - FileNameNoExt, LocalName, RAW, NonRAW, DisplayImage, XMP & Bundle extensions.
8. The whole thing to get to your clipboard (apart from creating the view) is:
```
cls

selected > select LocalName, filenamenoext, MediaField(files, 1, 'format') Raw, MediaField(files, 2, 'format') NonRaw, MediaField(files, 3, 'format') Bundle, MediaField(files, 4, 'format') Display, MediaField(files, 4, 'format') XMP from media inner join localdirs on media.containingFolderHash = localdirs.uniqueHash where media.uniqueHash in ($_) > json -v

copy
```
9. Run script using...

Reference: [Mylio Support Forum Topic](https://forum.mylio.com/t/list-of-file-path-and-names-for-search-or-filter-results/6093/4)
