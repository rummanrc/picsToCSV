# picsToCSV
Convert image files (seperated By Folders) to a CSV file, 
where the rows contain the flattened image arrays and each column contains a pixel value.
By default the resolution to be converted to is set to 28x28
The folder names are assumed to be 100,101,102... and so on. But if you want custom names, you can create an array and loop over it.
The class labels are set according to the folder serial. 
The first folder entries are labelled as 0, second are labelled as 1, third as 2 and so on.


This is helpful if you have multiple folders containing images (divided according to their respective labels),
and you need them to be converted to one single CSV file.

