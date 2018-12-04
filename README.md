![ibm_ad](https://ibm.box.com/shared/static/qo20b88v1hbjztubt06609ovs85q8fau.png)

# Search for extra terrestrial intelligence (SETI) with Tensorflow on PowerAI

Each night, using the Allen Telescope Array (ATA) in northern California, the SETI Institute scans the sky at various radio frequencies, observing star systems with known exoplanets, searching for faint but persistent signals. The current signal detection system is programmed to search only for particular kinds of signals: narrow-band carrier waves. However, the detection system sometimes triggers on signals that are not narrow-band signals (with unknown efficiency) and are also not explicitly-known radio frequency interference (RFI). There seems to be various categories of these kinds of events that have been observed in the past.

Our goal is to classify these accurately in real-time. This may allow the signal detection system to make better observational decisions, increase the efficiency of the nightly scans, and allow for explicit detection of these other signal types.

For more information refer to [SETI hackathon page](https://github.com/setiQuest/ML4SETI/).

When you’ve completed this pattern, you will understand how to:

- Convert signal data into image data
- Build and train a convolutional neural network
- Display and share results in Jupyter Notebooks

This pattern will assist application developers who need to efficiently build powerful deep learning applications and use GPUs to train the model quickly.

<!-- TODO:
![flow](doc/source/images/architecture.png)

## Flow

1. The developer loads the provided notebook, which is run on a PowerAI system.
2. A provided example dataset demonstrates a use case of recognizing images of houses with pools vs. houses without pools.
3. The notebook uses the TensorFlow Inception model and retraining example to leverage the existing model and build a custom classifier.
4. The notebook compares results using the original model and the new model.
 -->

<!-- TODO:
# Watch the Video

[![](http://img.youtube.com/vi/S3jdmKrERrE/0.jpg)](https://www.youtube.com/watch?v=S3jdmKrERrE)
 -->

# Steps

Follow these steps to setup and run this code pattern. The steps are
described in detail below.

1. [Get 24-hours of free access to the PowerAI platform](#1-get-24-hours-of-free-access-to-the-powerai-platform)
1. [Access and start the Jupyter notebooks](#2-access-and-start-the-jupyter-notebooks)
1. [Run the notebooks](#3-run-the-notebooks)
1. [Analyze the results](#4-analyze-the-results)
1. [Save and share](#5-save-and-share)
1. [End your trial](#6-end-your-trial)

## 1. Get 24-hours of free access to the PowerAI platform

IBM has partnered with Nimbix to provide cognitive developers a trial account that provides 24-hours of free processing time on the PowerAI platform. Follow these steps to register for access to Nimbix to try the PowerAI code patterns and explore the platform.

* Go [here](https://www.ibm.com/account/reg/us-en/login?formid=urx-19543) and follow the instructions to register for your free trial.

* Use the welcome page (or confirmation email) to determine when your container is "ACTIVE" and collect the following information:
  * **IP Address** (might be fully qualified domain name)
  * **User Id**
  * **Password**

* Take the **IP Address** (FQDN) and use your local browser to go to `https://<IP Address>`.

* Login with the **User Id** and **Password**.

  ![welcome](https://raw.githubusercontent.com/IBM/pattern-utils/master/powerai/welcomepage.png)

## 2. Access and start the Jupyter notebooks

* Get a new terminal window by clicking on the ```New``` pull-down and selecting ``Terminal``.

  ![powerai-notebook-terminal](https://raw.githubusercontent.com/IBM/pattern-utils/master/powerai/powerai-notebook-terminal.png)

* When using the free trial, a 4-hour timeout will cause you to lose data that is not in the `/data` directory. Create a `patterns` directory under `/data` and a symbolic link for that directory under `/usr/local/samples/` as follows:

  ```commandline
  mkdir /data/patterns
  ln -s /data/patterns /usr/local/samples/
  ```

* Use `git clone` to download the example notebook, dataset, and retraining library into `/data/patterns`:

  ```commandline
  cd /data/patterns
  git clone https://github.com/IBM/powerai-seti-signal-classification
  ```

  ![clone](doc/source/images/clone.png)

* Once done, you can exit the terminal and return to the notebook browser. Use the `Files` tab. From the root folder, click on `patterns`, `powerai-seti-signal-classification`, and then `notebooks`.

  ![powerai-notebook-open](doc/source/images/powerai-notebook-open.png)

* If your container is paused (after 4 hours) and you resume it, your data will still be under `/data`. Recreate the symbolic link for it to show up in the Jupyter files tree.

  ```commandline
  ln -s /data/patterns /usr/local/samples/
  ```
## 3. Run the notebooks

### Introduction to running notebooks

When a notebook is executed, what is actually happening is that each code cell in
the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag
format is `In [x]:`. Depending on the state of the notebook, the `x` can be:

* A blank, this indicates that the cell has never been executed.
* A number, this number represents the relative order this code step was executed.
* A `*`, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

* One cell at a time.
  * Select the cell, and then press the `Play` button in the toolbar.
* Batch mode, in sequential order.
  * From the `Cell` menu bar, there are several options available. For example, you
    can `Run All` cells in your notebook, or you can `Run All Below`, that will
    start executing from the first cell under the currently selected cell, and then
    continue executing all cells that follow.

### The SETI notebooks

To complete the code pattern, run the training and prediction notebooks in this order:

1. seti_cnn_tf_training.ipynb
2. seti_predition.ipynb

#### Preparing the dataset

First, we read the Basic 4 dataset, converted signals into images, and saved them in MNIST format. Now our signal classification problem has become an image classification problem. We split the MNIST data into train and test datasets and stored the results.

The training and prediction notebooks use our stored results, so you can skip the data preparation step.
You can review an example notebook from the data preparation step [here](examples/seti_signal_to_mnist_format.ipynb).

#### Training the model

In the `seti_cnn_tf_training.ipynb` notebook, we use the famous SETI Dataset to build a convolutional neural network (CNN) able to perform signal classification. The CNN will determine, with some associated error, what type of signal is presented.

Run this notebook. The PowerAI free trial does not include GPUs, so training is slower. We'll use a lower number of epochs to finish faster, but this will hurt accuracy.

To create a production model for a problem of this size, you would want to run a higher number of epochs using PowerAI with one or more GPUs.

#### Prediction

In the `seti_prediction.ipynb` notebook, we will use the trained model to predict the signal class. Run this notebook after the training notebook has completed.

![](doc/source/images/powerai-notebook-run.png)

## 4. Analyze the results

### Performance
Convelutional Neural Network involves a lot of matrix and vector multiplications that can parallelized, so GPUs can overperform, because GPUs were designed to handle these kind of matrix operations in parallel!

### Why GPU overperforms?
A single core CPU takes a matrix operation in serial, one element at a time. But, a single GPU could have hundreds or thousands of cores, while a CPU typically has no more than a few cores.


### How to use GPU with TensorFlow?
It is important to notice that if both CPU and GPU are available on the machine that you are running the noebook, and if a TensorFlow operation has both CPU and GPU implementations, the GPU devices will be given priority when the operation is assigned to a device. 



### Benchmark:
- SETI_single_gpu_train.py achieves ~72% accuracy after 3k epochs of data (75K steps).
- Speed: With batch_size 128.  
- __Notice:__ The model is not optimized to reach to its highest accuracy, you can achive better results tuning the parameters.

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
- __Notice:__ The model is not optimized to reach to highest accuracy, and you can achive better results tuning the parameters.

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

## 5. Save and share

### How to save your work

Because this notebook is running temporarily on a Nimbix
Cloud server, use the following options to save your work:

Under the `File` menu, there are options to:

* `Download as...` will download the notebook to your local system.
* `Print Preview` will allow you to print the current state of the
  notebook.

## 6. End your trial

When you are done with your work, please cancel your subscription by visiting the `Manage` link on the **My Products and Services** page.