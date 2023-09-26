# planet_quality_estimation
Several machine learning projects outperforming commercial astronomical software
## Summary
This repository contains three Jupyter notebooks that focus on a dataset of (self-acquired) planet images. Our main goals are: 
- exploring and understanding the dataset
- building **Machine Learning models** able to rate the quality of the planet images, mimicking the extant astronomical software
- most importantly, **outperforming the commercial software** at ranking the quality of the images in terms of **aesthetic quality**, which results in **superior processed images**

**All the goals are achieved, in two very different ways.**

The whole project involves common Data Science and Machine Learning techniques, particularly:
- Linear Regression
- Ridge Regression
- Logistic Regression
- Convolutional Neural Networks and Transfer Learning



![processed_bigger](https://github.com/JGallegoPerez/wrong_forecasts/assets/89183135/817cf7be-d8bf-4209-9dac-3f8069f70d9a)

*(Processed image of Mars)*

## Motivation

Processing astronomical images is an arduous process. One of the first steps consists in **stacking** numerous images of the same target (*subframes*, 
in the astrophotography lingo), which significantly reduces the signal-to-noise ratio. Stacking usually involves a small number of subframes in amateur 
Deep-Sky Object (*DSO*) astrophotography; given that the subframe exposures are very long (typically, around 2-10 min), a good (and realistic) night would 
rarely yield us more than 40-50 usable subframes. In this case, selecting the best subframes is easily accomplished with just some visual inspection and by 
enabling the quality-rating options available in most commercial stacking applications.

However, in contrast to DSO astrophotography, *planetary* astrophotography involves many hundreds or thousands of subframes, taken with exposures as 
short as just a few milliseconds. Here too, commercial applications can ease the stacking process by automatically rating and including in the stack 
only the subframes with the highest quality (typically, 1-20% of the subframes in a video file; or around 500-5000 subframes). 

*In this project, we will use a subset of over 50000 images of planet Mars, which I extracted from a 5-min astronomical video recording that I personally collected.*

As we see, selecting the subframes with the highest quality possible is crucial to obtain a good final processed image. 


### Automatic quality estimation
Commercial applications, such as PIPP and AutoStakkert, do a great job at sorting the subframes by quality. They rely on methods such as estimating **local 
contrast** values. For example, according to the PIPP team (https://sites.google.com/site/astropipp/pipp-manual/quality-options):
*For the default and original quality algorithms, the quality estimation is based on calculating a summation of local contrast values (sum of squares of 
the difference between adjacent pixels) for a series of subsampled images. Higher calculated values indicate higher quality images.*

Generally, such quality algorithms tend to select good subframes; indeed in a much better way than just randomly choosing a certain percentage. However, if we 
inspect the top-rated images one by one, we may notice that some of them might not appeal to the eye and we may disagree with the ordering. For example, from 
the Mars images dataset we will work with, PIPP's default algorithm ranked the following two images one after the other, with the image on the left given a 
higher quality score than the image on the right: 

   ![wrong_ranking](https://github.com/JGallegoPerez/wrong_forecasts/assets/89183135/4a08b3a3-9fd8-44fc-9ead-1d8c60974986)

*The image on the left was given a higher quality score*

### Selecting subframes by aesthetic quality

Thus, we might disagree with such quality rankings, and, while not necessarily dismissing objective overall quality assessments, we may want to also 
consider our own subjective judgments. For instance, I may want to choose for my stack subframes that have been both automatically rated as good images,
but that *also* satisfy some personal aesthetic requirements: I may want the planet to appear quite round, I may want the image to be sharp enough 
at least in the regions of the disk that I'm mostly interested in, etc. For such a flexible and personal choosing, it would seem we have no choice but to hand-
pick our own images. Hundreds or thousands, one by one... Stacks of such images yield better final results and can also be tailored to maximize the aesthetic 
quality of specific regions of the picture. For instance, If I'm working with Jupiter subframes and the *Great Red Spot (GRS)* is my favorite region on the target, I could select only the 
subframes where the GRS "looks good". After stacking those hand-picked images, I would find that my effort had been rewarded, even if by a small margin. 

However, hand-picking planetary subframes is tremendously time-consuming. Even though I tried to create some solutions to speed up the process (see my repository: 
https://github.com/JGallegoPerez/subjective_qual_estimator), it still takes me several hours to accomplish the task. 

**It would be great if we had a system that could rate the subframes' quality in the same way we humans do.**

## The present project

It suprises me how much effort is devoted in the amateur astrophotography community to process Deep Sky Images (including advanced AI techniques, nowadays), 
while the available software to process planet images does not appear as developed. I found this a "low hanging fruit" on which a great deal of improvement can 
still be achieved. Specifically, I felt that there remained a lot of room for improvement on the image quality sorting algorithms, such as the Local Contrast algorithm.
I have experimented with various machine learning techniques to try to find better quality sorting solutions, which I present in this project. 

I present my work across three Jupyter notebooks, written in Python. This approach enhances the ability to manipulate and experiment with both data and code.
They don't necessarily need to be run sequentially, although the first notebook offers a more thorough exploration of the dataset. 

Across the three Jupyter notebooks, the objectives are as follows:
1. To investigate and gain insights into the dataset.
2. To develop Machine Learning models capable of assessing the quality of planetary images, emulating existing astronomical software.
3. Of utmost significance is surpassing commercial software in evaluating image quality from an aesthetic perspective, leading to the production of superior processed images.


### Notebook 1: **Linear and Ridge Regression**

Given that quality scores are represented as a continuous variable, we aimed to construct linear and Ridge regression models to imitate the functionality of the Local Contrast (LC) algorithm, 
which we planned to investigate in a quantitative manner. The question was whether these models could outperform the LC algorithm and align more closely with human visual perception.

Interestingly, our Ridge regression model closely replicated the quality ranking behavior of the PIPP software (LC algorithm), but it didn't exhibit superior image ranking in terms of aesthetic appeal.

However, we conducted an unconventional experiment by creating a linear model and introducing a coefficient matrix derived from a processed image of Mars. Surprisingly, this unorthodox 
approach led to the model surpassing the LC algorithm in ranking image aesthetic quality.

It's worth noting that this solution, while effective, lacks elegance and emerged from quick code prototyping. It may have proven successful simply because it calculates and sorts pixel-wise 
distances between the "ideal" or "ground truth" image and each subframe. But I anticipate that other more rigorous methods (for example, based on Mean Squared Error) that also evaluate pixel distances could yield similar results.

Below, we can observe two images of Mars, both stacked and sharpened with identical settings. On the left, LC contrast was used for image quality sorting, while on the right, the 
hereby described approach was employed:


   ![pipp_lin](https://github.com/JGallegoPerez/wrong_forecasts/assets/89183135/746ba0f6-cf15-4815-a46e-822d0177d4aa)

*Local Constrast vs. Linear Model*

### Notebook 2: **Logistic Regression**

The idea behind this approach was to train a logistic model using subframes that have the highest and lowest LC quality scores, aiming to enable the model to effectively 
distinguish between images labeled as "high" or "low" quality. Just like in the first notebook, we intended to evaluate the model both quantitatively and qualitatively (subjective aesthetic judgment), by comparing it to the LC algorithm.

The logistic model underwent training using subframes categorized as very high and very low quality. Consequently, it demonstrated a notable capability to differentiate 
between "good" and "bad" images. Furthermore, it exhibited a moderate level of correlation with images from a more representative dataset, implying its capacity to predict 
image quality in a manner similar to PIPP's LC algorithm. However, the logistic model did not outperform the LC algorithm when it came to ranking images based on aesthetic quality.



### Notebook 3: **Convolutional Neural Network and Transfer Learning**

We employed transfer learning with MobileNet, a pre-trained Convolutional Neural Network (CNN). We enhanced MobileNet by adding and training additional layers. Our goal 
was to train the model on two classes: low-quality and high-quality Mars subframes. Just like in the previous notebooks, the model was compared to the LC algorithm quantitatively and qualitatively (subjective aesthetic judgment)

Interestingly, when we extensively train the CNN, it achieves moderate correlation with PIPP's Local Contrast algorithm in terms of quantitative image quality. However, 
it doesn't consistently offer superior aesthetics. Surprisingly, when we significantly undertrain the network, it produces what I consider better images. The undertrained model's success may be attributed to 
transfer learning. MobileNet comes pre-loaded with knowledge from a vast dataset of over 14 million images spanning 20,000 classes. When we fully train the extended CNN, it might override this prior knowledge, 
sticking too closely to the LC algorithm. Conversely, by not letting the extra layers fit too much to the dataset, we may be allowing the network to tap into the image knowledge embedded in MobileNet's layers.

Below, we can observe two images of Mars, both stacked and sharpened with identical settings. On the left, LC contrast was used for image quality sorting, while on the right, 
the CNN model was employed:

   ![pipp_cnn](https://github.com/JGallegoPerez/wrong_forecasts/assets/89183135/8df6839a-0a6f-4c3a-90db-877f7a8e54e5)

*Local Contrast vs. CNN Model*

I'd like to emphasize that we maintained consistent image processing settings for three distinct images: one processed solely with PIPP, another generated by a linear model 
in the initial notebook, and the final one mentioned here. This consistency enables a fair comparison among them. All three images are built upon stacks of the top 2000 images 
from a substantial dataset comprising over 50,000 images. The last Convolutional Neural Network (CNN) model displayed a remarkable ability of consistent aesthetic quality ranking. 
That is, when we examine images sorted by quality within a specific range, they consistently exhibit a very similar degree of aesthetic quality. This contrasts sharply with the Local Contrast ranking, 
which lacks this consistency. This implies that employing the CNN model allows us to stack more images effectively. It can discover numerous good images throughout the entire dataset, 
even those that the Local Contrast algorithm may not have rated as high quality. Consequently, there are fewer poor-quality images interspersed within the top-ranking images and we can select more images. 
And incorporating more high-quality images in the stack leads to an increased high-to-noise ratio, which results in superior final processed images.

In fact, I achieved my finest Mars image (the one used to create the synthetic coefficient matrix in the first notebook) by initially using this model for image quality ranking, before stacking 
(see processed image below, again).

   ![processed_bigger](https://github.com/JGallegoPerez/wrong_forecasts/assets/89183135/817cf7be-d8bf-4209-9dac-3f8069f70d9a)

*(Processed image of Mars)*

## Instructions

Download the images as a zipped file and unzip the folder. The images can be downloaded from this repository:
https://figshare.com/articles/figure/Mars_subframes/2419077
(It doesn't matter where the images folder is saved; each Jupyter notebook will contain instructions to specify the folder path). 

Clone the repository or save all files into the same directory. That includes not only the Jupyter notebooks.

Install the necessary dependencies from *requirements.txt*. 

Run the notebooks. 

## Acknowledgments

Special thanks to Takazumi Matsumoto for his collaboration at the image acquisition and for his support on the present project.  
