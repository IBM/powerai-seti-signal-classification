<a href="https://www.cognitiveclass.ai"><img src = "https://ibm.box.com/shared/static/qo20b88v1hbjztubt06609ovs85q8fau.png" align = left height="50"></a>
<br>

# Search for Extra Terrestrial Intelligence (SETI)


### Project overview:
Each night, using the Allen Telescope Array (ATA) in northern California, the SETI Institute scans the sky at various radio frequencies, observing star systems with known exoplanets, searching for faint but persistent signals. The current signal detection system is programmed to search only for particular kinds of signals: narrow-band carrier waves. However, the detection system sometimes triggers on signals that are not narrow-band signals  (with unknown efficiency) and are also not explicitly-known radio frequency interference (RFI). There seems to be various categories of these kinds of events that have been observed in the past.

Our goal is to classify these accurately in real-time. This may allow the signal detection system to make better observational decisions, increase the efficiency of the nightly scans, and allow for explicit detection of these other signal types.

For more information refer to [SETI hackathon page](https://github.com/setiQuest/ML4SETI/).  

When you’ve completed this pattern, you will understand how to:
-	Convert signal data into image data
-	Build and train a convolutional neural networks
-	Display and share results in Jupyter Notebooks
This pattern will assist application developers who need to efficiently build powerful deep learning applications and use GPUs to train the model quickly. 

## Notebooks:
This repository includes 3 parts:
### 1. Preparing dataset
  * **Converting images to binary files using Numpy (SETI_img_to-binary.ipynb)**
      * In this notebook we read the Basic 4 dataset and convert signals into a binary file. Also, we split data into train/test datasets. 
  * **Optional: Converting images to binary files using Spark (SETI_img_to_binary_spark.ipynb)**  
      * In this notebook we read the Basic 4 dataset through Spark, and convert signals into a binary file. It is an optional notebook and you dont have to run it if you have already converted the images to binary files.
### 2. Classification
 * **Classification of images using CCN on Single GPU (SETI_CNN_Tf_SingleGpu.ipynb)**
     * In this Notebook, we will use the famous [SETI Dataset](https://github.com/setiQuest/ML4SETI/) to build a Convolutional Neural Networks capable to perform signals classification. CNN will say, with some associated error, what type of signal is the presented input.
     * In our case, as we are running this notebook on [IBM PowerAI](http://cocl.us/SETI-NIMBIX-PowerAI), you have access to multi GPU, but we use one of the GPUs in this notebook, for the sake of simplicity.
 * **Optional: Classification of images using CCN on Multi GPU (SETI_CNN_Tf_MultiGpu.ipynb)** 
     * This Notebook, builds a Convolutional Neural Networks, but using multi GPUs. You will use IBM PowerAI with multiple GPU to train the model in parallel manner.
     * You can run this notebook in case you have access to an environment with multiple GPUs.
### 3. Prediction
* **Use the trained model for prediciton (SETI_prediction.ipynb)**
     * In this notebook you can load a pre-trained model and predict the signal class.

### Performance
Convolutional Neural Network involves a lot of matrix and vector multiplications that can parallelized, so GPUs can overperform, because GPUs were designed to handle these kind of matrix operations in parallel!

### Why GPU overperforms?
A single core CPU takes a matrix operation in serial, one element at a time. But, a single GPU could have hundreds or thousands of cores, while a CPU typically has no more than a few cores.


### How to use GPU with TensorFlow?
It is important to notice that if both CPU and GPU are available on the machine that you are running the notebook, and if a TensorFlow operation has both CPU and GPU implementations, the GPU devices will be given priority when the operation is assigned to a device. 



### Benchmark:
- SETI_single_gpu_train.py achieves ~72% accuracy after 3k epochs of data (75K steps).
- Speed: With batch_size 128.  
- __Notice:__ The model is not optimized to reach to its highest accuracy, you can achieve better results tuning the parameters.

<table border="1" style="box-sizing: border-box; border-spacing: 30px; background-color: transparent; color: #333333; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 12px;">
<tbody style="box-sizing: border-box;">
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">CPU Architecture</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">CPU cores&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">Memory&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">GPU&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">Step time (sec/batch)&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">&nbsp;Accuracy</span></td>
</tr>
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;">POWER8</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">40</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">256 GB</td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">1 x Tesla K80</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">~0.127 </td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">~72% at 75K steps  (3 hours)</td>
</tr>
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;" >POWER8</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">32</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">128 GB</td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">1 x Tesla P100 w/NVLink np8g4</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">~0.035 </td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">~72% at 75K steps  (1 hour)</td>
</tr>


</tbody>
</table>


- SETI_multi_gpu_train.py achieves ~72% accuracy after 75K steps.
- Speed: With batch_size 128.  
- __Notice:__ The model is not optimized to reach to highest accuracy, and you can achieve better results tuning the parameters.

<table border="1" style="box-sizing: border-box; border-spacing: 30px; background-color: transparent; color: #333333; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 12px;">
<tbody style="box-sizing: border-box;">
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">CPU Architecture</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">CPU cores&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">Memory&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">GPU&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">Step time (sec/batch)&nbsp;</span></td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;"><span style="box-sizing: border-box; font-weight: bold;">&nbsp;Accuracy</span></td>
</tr>
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;">POWER8</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">160</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">1 TB</td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">4 x Tesla K80</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">~0.066 </td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">~72% at 75K steps  (83 minutes)</td>
</tr>
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;" >POWER8</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">64</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">256 GB</td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">2 x Tesla P100 w/NVLink np8g4</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">~0.033 </td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">~72% at 75K steps  (40 minutes) </td>
</tr>
<tr style="box-sizing: border-box;">
<td style="box-sizing: border-box; padding: 3px; text-align:left;">POWER8</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">128</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">512 GB</td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">4 x Tesla P100 w/NVLink np8g4</td>
<td style="box-sizing: border-box; padding: 3px; text-align:center;">~0.017 </td>
<td style="box-sizing: border-box; padding: 3px; text-align:left;">~72% at 75K steps  (20 minutes)</td>
</tr>
</tbody>
</table>




