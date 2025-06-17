# ParallelWaveGAN Hardware Accelerator #

This directory contains work done both through weekly codefests, and in general, on the final project for ECE 410. The work for this project was done in collaboration with Google's Gemini LLM, and the full conversation for works done within can be found [here](https://g.co/gemini/share/235e15774d0e).

## Motivation ##
Though audio generation has made leaps in progress in recent years, recent developments tend to be focused on either software generation for music or hardware implementations that are focused on voice generation; there appears to be a large gap in hardware use of machine learning for musical purposes. As such, I was hoping to look for an audio generation algorithm that could be deployed on an FPGA module for use with Eurorack modular synthesizers. The ultimate goal is to create a self-contained Eurorack module that takes in inputs in the form of control voltage, and uses a machine learning algorithm to output audio samples that can be converted into output voltage.

## Algorithm Choice ##
Research began by looking through modern audio generation algorithms for a viable candidate that would fit our hardware needs; Gemini's "Deep Research" model was particularly helpful in this quest, and the associated conversation can be found [here](https://g.co/gemini/share/6be085720a74). The first algorithm we looked at is Google's Wavenet<sup>1</sup>, one the earliest approaches to ML audio generation. This algorithm served as the inspiration to many of the other algorithms we would eventually look at and features a Dilated Causal Convolutional Network to create the next sample based on a large window of previously generated samples along with a conditional input; as this algorithm was originally intended for voice synthesis, the conditional input is typically composed of text words for the network to convert into audio representations of these words. Dilated, in this context, refers to their technique for increasingly accessing samples that are wider apart as we move up through the network layers; this allows the netork to be influenced by a much larger window of samples and inputs without compromising computational efficiency. This algorithm was deemed unfit for our purposes due to the limitations on real-time performance imposed by its causal (sequential) nature; in order to precict the next sample, the network needs access to the sample immediately preceeding it. 

On the other end of the spectrum are current state-of the art models that rely on transformers in order to turn a description of a song into a full clip representative of the characteristics within that prompt; AudioCraft<sup>2</sup> by Meta is one such example. This is a powerful method, but lacks the direct control that we were seeking for our model; rather than creating entire tracks or songs, the goal was to create waveshapes or granular samples of sound to allow for more direct musical explorations.

The algorithm that we ended up settling on was name ParallelWaveGAN<sup>3</sup> (PWG). This algorithm is based off of the original WaveNet structure, but adds a few key improvements that are critical for our use case. Firstly, the parallel nature of this algorithm means that we are creating multiple samples every time-step, allowing us to achieve the real-time performance necessary for deployment as a hardware synthesis module. Secondly, the GAN design allows us to perform the more complicated training computations (discriminator) on a general purpose computer and deploy the simpler output network (generator) on our hardware. Given the timeline of this course project, only the generator network was focused on with the discriminator left for future work.

## Profiling ##
With our algorithm chosen, we can begin to analyze the portion of this algorithm that we can accelerate. To this end, we began by creating a naive Python implementation of the PWG generator network; code found within ```Week_2```. Without training data available yet, we tested this script by initalizing the weights randomly and verifying that the output sample was "believable" as an audio sample.

![Naive Generator Output](./Week_2/numpy_generated_audio.png) 

Though the waveform appears to be Once we confirmed that our naive model was performing as we expected, we could begin to profile the workload in the hopes of finding the biggest bottleneck. Firstly, we used ```cProfile``` to identify which of our function calls was taking up the most run-time. As expected, our ```conv_1D``` function was taking up the bulk or our calculation time; this makes sense as the ```conv_1D``` function is used for all of our hidden layers, whereas our other equally expensive convolution functions are only used a single time. 

![cProfile Output](./docs/cProfile.png)

We can further narrow down the most troublesum operations by using ```line_profiler``` within this function to confirm our suspicions regarding what specific line within this function is taking up the most time. As before, the results were expected and point to a nested loop of MAC operations that are both individually costly and repeated a vast number of times for every feed-forward pass. 

![line_profiler Output](./docs/Line_Profile.png)

## Design ##
Since the bottleneck was determined to be MAC operations, we can begin work on designing our hardware solution. MAC operations are a common bottleneck in many machine learning workloads, and there are many possible solutions to accelerate them on custom hardware; for our design we chose to address the problem by using an output-stationary systollic array to allow the multple MAC operations required for each layer in parallel. Systolic arrays are inspired by the structure of biological hearts, and feature an array of pipelined Processing Elements (PEs) where data pulses through the array and is processed as it moves through each PE; activation values can be fed from left to right, while weights can be fed in from top to bottom. Output-stationary refers to the fact that we are feeding in both of our inputs to the array (activations and weights) while each PE retains the completed MAC value until it is accessed after all data has been processed.

![Systolic Array Structure. *Source: telesens.co*](https://telesens.co/wp-content/uploads/2018/09/img_5ba82bde783ad-296x300.png)

The software/hardware boundary did end up changing through the eventual rescoping of the project, but our original plan was to have the software feed the raw input activation and weight vectors to the hardware through PCIe; SPI was determined to be too limiting for the large bandwidth we would require for real-time performance.

## Implementation Journey ##
Our journey through this project was rife with complications, primarily related to the complexity of the algorithm and the difficulty encountered in verify the intricate timing that the control path requires. Though we did find inital success in the foundational modules, our troubles with verification and control complexity led to a need to re-evaluate our scope and attempt to push some of the complexities to the more capable software processing. Though this ultimately proved unfruitful, we produced parameterized and verified modules for the main processing elements with unit tests for each, with the major obstacle continuing to be the FSM; specifically the difficulty in verifying this module.

### Main Modules ###
#### Systolic Node ####
Our systolic array was designed bottom-up with a single SystemVerilog systollic node (PE), featuring a built in saturating accumulator that stores our calculated MAC value, along with the input-output pipeline for activations, weights, and valid bits to ensure that the data is properly being propagated through the network. This module was unit-tested with a focus on timing, value correctness, and appropriate response to valid bits before moving on to the full array.

#### Systolic Array ####
The systolic array, as implied in its name, is simply an array made up of our systolic nodes with packed input/output vectors and control signals that feed to every node in the array. As the individual nodes had been verified already, the unit test for this module was focused on ensuring that our results were accurate for a reduced matrix-matrix multiplication. 

#### FIFO ####
In order to ensure that our data stream is consistent and well timed, we need to buffer the inputs from the PCIe module into a module that can stream a single input value on every clock cycle; this ensures that we're leveraging the full bandwidth capabilities of PCIe while balancing our need for sequential inputs. To accomplish this, a basic FIFO module was developed and verified with the goal of eventually connecting it to our PCIe module or possibly general memory module controlled by the FSM.

#### PCIe Port ####
*Work in progress*. Due to the high bandwidth requirements for real-time performance, we opted to use PCIe rather than SPI for our SW/HW interface. Unfortunately, troubles with the FSM early on led to this module being pushed back for the time being. 

#### Control FSM ####
This module is the brain of our generator network. Originally, this FSM was responsible for maintining the required timing between all of our modules, as well as a limited amount of data pre-processing (input dilation), and finally some tiling if a layer's calculations was too large for our array. While the module was designed and developed, verifying correct functionality proved to be somewhat beyond my current capabilities as an undergraduate ECE student given the timeline for the project. As such, a change in our SW/HW boundary and rescoping of the project goal was necessary. 

### Re-evaluating Scope and HW/SW Boundary ###

