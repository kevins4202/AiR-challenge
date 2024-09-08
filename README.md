1. Static image processing\
For the first challenge, I tried multiple approaches constituting a combination of blurring, eroding, dilating,
 masking, canny image detection, and contouring. I knew that I had to use contouring to detect the area of the
shapes, but I found that canny edge detection was better for detecting the shapes. So, my final solution was to
first blur the image three times to smooth the image initially, then masking the grass (manually selecting the upper
and lower bounds). I then used canny edge detection to find the edges, and contouring to refine them and make them
better defined. I then calculated the centers of the shapes using center of mass.\
I know this algorithm was extremely inefficient because I used many algorithms that had to go over the whole image
such as masking, and it took a long time to refine the algorithm to get it to work with the video.\

3. Video 1\


4. Video 2\
