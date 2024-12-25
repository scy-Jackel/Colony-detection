I checked the pre-trained dataset of the cellpose library, and found that the modality of this dataset is somewhat different (see cellpose.jpg), so I made two sets of methods, one based on the cellpose library and the other based on my own thinking.

cellpose:
The original data is in the data folder. First run Colony_detection.py to get the segmentation mask and output it to the cellpose folder, but the mask background here is black and the foreground is light. Run reverse.py to get the flipped mask and output it to the seg_cellpose folder. Finally, run merge_cellpose.py to get the composite image and output it to merge_cellpose.

My own method:
The original data is in the data folder. First run segment.py, use a threshold of 70 to remove background noise, then use a 11*11 sliding window to detect the mean in the window, exclude some smaller noise, and then continue to detect the mean in the window through a 50*50 sliding window, retain only larger colonies, calculate the area of ​​each connected area, filter out areas with too small an area, and then get the mask image and output it to the seg_result folder. After that, run denoise.py to detect the range of the petri dish, extract the area outside the petri dish and store the result in the denoise folder. Finally, run merge.py to combine the original image with the segmentation mask into a fused image and output it to the merge folder.

Note:
The folder data is the original image.
The folder seg_cellpose is the segmented image obtained by the cellpose library.
The folder merge_cellpose is the fused image after the cellpose library is segmented and combined with the original image.
The folder denoise is the segmented image obtained by my method.
The folder merge is the fused image after the segmentation and combined with the original image by my method.
The paths in the python file all use relative paths, so you can run it directly without modifying the path.
