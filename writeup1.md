# **Finding Lane Lines on the Road** 

## Writeup Template

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied the
gaussian blur to the image witht the default parameters. Then the canny edge detection algorithm
is also applied. Then I cut the areas where we are not interested in detecting the lines. Finally
the hough lines algorithm is applied and the lines in a selected image region are found. After this
the original image is drawn with the found (extrapolated) on top. 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by
first finding the right and left lines. I also removed the lines which were too flat to be compatible
with the white lines by removing lines with a slope less that 0.5. Also selected lines according
to their position in the image. After these selections the line (points) were gives to the polyfit
function and with it was possible to find the slope and insercted of the average line position.
With the slope and intersect it is possible to find the x points on the left and right lines.
Then finding the y points is trivial. 


### 2. Identify potential shortcomings with your current pipeline

When trying the challenge I saw that the drawn lines are very unstable.  


### 3. Suggest possible improvements to your pipeline

Maybe the parameters could be optimized in order to have more stable lines in the challenge video.
Also making the extrapolation slightly "smaller" such that we can take into account curves.  