
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

## Import Packages


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import copy
%matplotlib inline
```

## Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x7fb3a93d5128>




![png](fina_result_code_files/fina_result_code_6_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

  
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    rline_li = []    
    lline_li = []        
    img_x_center = img.shape[1] / 2 # x coordinate of center of image    

    slope_thr1 = 0.5
    slope_thr2 = 0.1
    
    for line in lines:
         
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
          
            if x2-x1 == 0:
                slope = 999. # practically infinite slope
            else:    
                slope = (y2-y1)/(x2-x1)   
                     
            if slope>0 and slope_thr1 < abs(slope) and x1 > img_x_center and x2 > img_x_center:
                rline_li.append(line)
                
            elif slope<0 and slope_thr1 < abs(slope) and x1 < img_x_center and x2 < img_x_center:                   
                lline_li.append(line)
                
    
    rlines_x,rlines_y,llines_x,llines_y = [],[],[],[]
    for line in rline_li:
        line = line[0]
        rlines_x += [line[0],line[2]]  # indices 0,2
        rlines_y += [line[1],line[3]] # indeices 1,3
    
    for line in lline_li:
        line = line[0]
        llines_x += [line[0],line[2]] # indices 0,2
        llines_y += [line[1],line[3]] # indeices 1,3   
    
    # check if there are lines for the fitting
    empty_right = False
    empty_left = False    
    
    trap_height = 0.4
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)      
  
    if len(rlines_x)>0 and len(rlines_y)>0:
        right_m, right_b = np.polyfit(rlines_x, rlines_y, 1)       
        right_x1 = (y1 - right_b) / right_m
        right_x2 = (y2 - right_b) / right_m
        
        right_x1 = int(right_x1)
        right_x2 = int(right_x2)
     
    else:
        empty_right = True
        
    if len(rlines_x)>0 and len(rlines_y)>0:
        left_m, left_b = np.polyfit(llines_x, llines_y, 1)  
        left_x1 = (y1 - left_b) / left_m
        left_x2 = (y2 - left_b) / left_m
        
        left_x1 = int(left_x1)
        left_x2 = int(left_x2)    
    

    else:
        empty_left = True 

            
    y1 = int(y1)
    y2 = int(y2)
    
    if not empty_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if not empty_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
        
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Test Images

Build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
image_location_li = os.listdir("test_images/")

```

## Build a Lane Finding Pipeline



Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.


```python
# TODO: Build your pipeline that will draw lane lines on the test_images

# open all images from the list and store them in a dictionary
image_dic,image_init_dic = {},{}

#parameters
kernel_size = 5
low_threshold = 50
high_threshold = 150

rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 4    # maximum gap in pixels between connectable line segments

ignore_mask_color = 255   
image_str = "solidWhiteCurve.jpg"
    
img = mpimg.imread("test_images/"+image_str)
initial_img = mpimg.imread("test_images/"+image_str)
    
img = grayscale(img)
img = gaussian_blur(img, kernel_size)
img = canny(img,low_threshold,high_threshold)
    
    # Cut areas we are not interested in
mask = np.zeros_like(img)
imshape = img.shape    
vertices = np.array([[(50,imshape[0]), (460, 300) , (imshape[1]-460, 300) , (imshape[1]-50 ,imshape[0])]] , dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
img = cv2.bitwise_and(img, mask)
    
lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
 
img = hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
img = weighted_img(img,initial_img ,1, 1, 0.)
  
#print(len(lines))
#print(image_dic) 

plt.imshow(img)

```

    Slope: 0.576811594203
    Slope kept: 0.576811594203
    Slope: 0.553571428571
    Slope kept: 0.553571428571
    Slope: -0.784615384615
    Slope kept: -0.784615384615
    Slope: -0.833333333333
    Slope kept: -0.833333333333
    Slope: -0.78125
    Slope kept: -0.78125
    Slope: -0.666666666667
    Slope kept: -0.666666666667
    Slope: -0.830508474576
    Slope kept: -0.830508474576
    Slope: -0.764705882353
    Slope kept: -0.764705882353
    Slope: -0.727272727273
    Slope kept: -0.727272727273
    Slope: -0.133333333333
    Slope: 0.585106382979
    Slope kept: 0.585106382979
    Slope: -0.0833333333333
    Slope: -0.714285714286
    Slope kept: -0.714285714286
    Slope: 0.586206896552
    Slope kept: 0.586206896552
    Slope: 0.6
    Slope kept: 0.6
    Slope: -0.0952380952381
    Slope: 0.615384615385
    Slope kept: 0.615384615385
    Slope: 0.0
    Slope: -0.5
    Slope: -0.833333333333
    Slope kept: -0.833333333333
    Slope: -0.785714285714
    Slope kept: -0.785714285714
    Slope: -0.833333333333
    Slope kept: -0.833333333333
    Slope: 0.619047619048
    Slope kept: 0.619047619048
    Slope: 0.0
    Slope: 0.552941176471
    Slope kept: 0.552941176471
    Slope: -0.770833333333
    Slope kept: -0.770833333333
    Slope: 0.576923076923
    Slope kept: 0.576923076923





    <matplotlib.image.AxesImage at 0x7fb36fbaf550>




![png](fina_result_code_files/fina_result_code_16_2.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    
    
    img = image
    initial_img = image
    
    img = grayscale(img)
    img = gaussian_blur(img, kernel_size)
    img = canny(img,low_threshold,high_threshold)
    
    # Cut areas we are not interested in
    mask = np.zeros_like(img)
    imshape = img.shape    
    vertices = np.array([[(50,imshape[0]), (460, 300) , (imshape[1]-460, 300) , (imshape[1]-50 ,imshape[0])]] , dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    img = cv2.bitwise_and(img, mask)
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
 
    img = hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
    img = weighted_img(img,initial_img ,1, 1, 0.)
    
    result = img    
    return result
```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
print(clip1)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    <moviepy.video.io.VideoFileClip.VideoFileClip object at 0x7fb3700a5978>
    [MoviePy] >>>> Building video test_videos_output/solidWhiteRight.mp4
    [MoviePy] Writing video test_videos_output/solidWhiteRight.mp4


    100%|█████████▉| 221/222 [00:06<00:00, 29.42it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidWhiteRight.mp4 
    
    CPU times: user 4.34 s, sys: 272 ms, total: 4.61 s
    Wall time: 7.8 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidWhiteRight.mp4">
</video>




## Improve the draw_lines() function

**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**

**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/solidYellowLeft.mp4
    [MoviePy] Writing video test_videos_output/solidYellowLeft.mp4


    100%|█████████▉| 681/682 [00:22<00:00, 30.68it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidYellowLeft.mp4 
    
    CPU times: user 14.2 s, sys: 888 ms, total: 15.1 s
    Wall time: 23.2 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidYellowLeft.mp4">
</video>




## Writeup and Submission

If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
##clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/challenge.mp4
    [MoviePy] Writing video test_videos_output/challenge.mp4


     99%|█████████▉| 125/126 [00:08<00:00, 11.43it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/challenge.mp4 
    
    CPU times: user 8.94 s, sys: 300 ms, total: 9.24 s
    Wall time: 11.5 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/challenge.mp4">
</video>





```python

```
