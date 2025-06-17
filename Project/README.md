# ParallelWaveGAN Hardware Accelerator #

This directory contains work done both through weekly codefests, and in general, on the final project for ECE 410. The work for this project was done in collaboration with Google's Gemini LLM, and the full conversation for works done within can be found [here](https://g.co/gemini/share/235e15774d0e).

## Motivation ##
Though audio generation has made leaps in progress in recent years, recent developments tend to be focused on either software generation for music or hardware implementations that are focused on voice generation; there appears to be a large gap in hardware use of machine learning for musical purposes. As such, I was hoping to look for an audio generation algorithm that could be deployed on an FPGA module for use with Eurorack modular synthesizers. The ultimate goal is to create a self-contained Eurorack module that takes in inputs in the form of control voltage, and uses a machine learning algorithm to output audio samples that can be converted into output voltage.

## Algorithm Choice ##
Research began by looking through modern audio generation algorithms for a viable candidate that would fit our hardware needs; Gemini's "Deep Research" model was particularly helpful in this quest, and the associated conversation can be found [here](https://g.co/gemini/share/6be085720a74). The first algorithm we looked at is Google's Wavenet<sup>1</sup>, one the earliest approaches to ML audio generation. This algorithm served as the inspiration to many of the other algorithms we would eventually look at and features a Dilated Causal Convolutional Network to create the next sample based on a large window of previously generated samples along with a conditional input; as this algorithm was originally intended for voice synthesis, the conditional input is typically composed of text words for the network to convert into audio representations of these words. Dilated, in this context, refers to their technique for increasingly accessing samples that are wider apart as we move up through the network layers; this allows the network to be influenced by a much larger window of samples and inputs without compromising computational efficiency. This algorithm was deemed unfit for our purposes due to the limitations on real-time performance imposed by its causal (sequential) nature; in order to predict the next sample, the network needs access to the sample immediately preceding it. 

On the other end of the spectrum are current state-of the art models that rely on transformers in order to turn a description of a song into a full clip representative of the characteristics within that prompt; AudioCraft<sup>2</sup> by Meta is one such example. This is a powerful method, but lacks the direct control that we were seeking for our model; rather than creating entire tracks or songs, the goal was to create waveshapes or granular samples of sound to allow for more direct musical explorations.

The algorithm that we ended up settling on was name ParallelWaveGAN<sup>3</sup> (PWG). This algorithm is based off of the original WaveNet structure, but adds a few key improvements that are critical for our use case. Firstly, the parallel nature of this algorithm means that we are creating multiple samples every time-step, allowing us to achieve the real-time performance necessary for deployment as a hardware synthesis module. Secondly, the GAN design allows us to perform the more complicated training computations (discriminator) on a general purpose computer and deploy the simpler output network (generator) on our hardware. Given the timeline of this course project, only the generator network was focused on with the discriminator left for future work.

## Profiling ##
With our algorithm chosen, we can begin to analyze the portion of this algorithm that we can accelerate. To this end, we began by creating a naive Python implementation of the PWG generator network; code found within ```Week_2```. Without training data available yet, we tested this script by initializing the weights randomly and verifying that the output sample was "believable" as an audio sample.

![Naive Generator Output](./Week_2/numpy_generated_audio.png) 

Though the waveform appears to be jagged and dissimiliar to the softer waves one would expect from audio, this is somewhat expected due to the random state of our weights in this naive approach. Once we confirmed that our naive model was performing as we expected, we could begin to profile the workload in the hopes of finding the biggest bottleneck. Firstly, we used ```cProfile``` to identify which of our function calls was taking up the most run-time. As expected, our ```conv_1D``` function was taking up the bulk or our calculation time; this makes sense as the ```conv_1D``` function is used for all of our hidden layers, whereas our other equally expensive convolution functions are only used a single time. 

![cProfile Output](./docs/cProfile.png)

We can further narrow down the most troublesome operations by using ```line_profiler``` within this function to confirm our suspicions regarding what specific line within this function is taking up the most time. As before, the results were expected and point to a nested loop of MAC operations that are both individually costly and repeated a vast number of times for every feed-forward pass. 

![line_profiler Output](./docs/Line_Profile.png)

## Design ##
Since the bottleneck was determined to be MAC operations, we can begin work on designing our hardware solution. MAC operations are a common bottleneck in many machine learning workloads, and there are many possible solutions to accelerate them on custom hardware; for our design we chose to address the problem by using an output-stationary systolic array to allow the multiple MAC operations required for each layer in parallel. Systolic arrays are inspired by the structure of biological hearts, and feature an array of pipelined Processing Elements (PEs) where data pulses through the array and is processed as it moves through each PE; activation values can be fed from left to right, while weights can be fed in from top to bottom. Output-stationary refers to the fact that we are feeding in both of our inputs to the array (activations and weights) while each PE retains the completed MAC value until it is accessed after all data has been processed.

![Systolic Array Structure. *Source: telesens.co*](https://telesens.co/wp-content/uploads/2018/09/img_5ba82bde783ad-296x300.png)

The software/hardware boundary did end up changing through the eventual rescoping of the project, but our original plan was to have the software feed the raw input activation and weight vectors to the hardware through PCIe; SPI was determined to be too limiting for the large bandwidth we would require for real-time performance.

## Implementation Journey ##
Our journey through this project was rife with complications, primarily related to the complexity of the algorithm and the difficulty encountered in verify the intricate timing that the control path requires. Though we did find initial success in the foundational modules, our troubles with verification and control complexity led to a need to re-evaluate our scope and attempt to push some of the complexities to the more capable software processing. Though this ultimately proved unfruitful, we produced parameterized and verified modules for the main processing elements with unit tests for each, with the major obstacle continuing to be the FSM; specifically the difficulty in verifying this module.
### Main Modules ###
#### Systolic Node ####
Our systolic array was designed bottom-up with a single SystemVerilog systolic node (PE), featuring a built in saturating accumulator that stores our calculated MAC value, along with the input-output pipeline for activations, weights, and valid bits to ensure that the data is properly being propagated through the network. This module was unit-tested with a focus on timing, value correctness, and appropriate response to valid bits before moving on to the full array.
#### Systolic Array ####
The systolic array, as implied in its name, is simply an array made up of our systolic nodes with packed input/output vectors and control signals that feed to every node in the array. As the individual nodes had been verified already, the unit test for this module was focused on ensuring that our results were accurate for a reduced matrix-matrix multiplication. 
#### FIFO ####
In order to ensure that our data stream is consistent and well timed, we need to buffer the inputs from the PCIe module into a module that can stream a single input value on every clock cycle; this ensures that we're leveraging the full bandwidth capabilities of PCIe while balancing our need for sequential inputs. To accomplish this, a basic FIFO module was developed and verified with the goal of eventually connecting it to our PCIe module or possibly general memory module controlled by the FSM.
#### PCIe Port ####
*Work in progress*. Due to the high bandwidth requirements for real-time performance, we opted to use PCIe rather than SPI for our SW/HW interface. Unfortunately, troubles with the FSM early on led to this module being pushed back for the time being. 
#### Control FSM ####
This module is the brain of our generator network. Originally, this FSM was responsible for maintaining the required timing between all of our modules, as well as a limited amount of data pre-processing (input dilation), and finally some tiling if a layer's calculations was too large for our array. While the module was designed and developed, verifying correct functionality proved to be somewhat beyond my current capabilities as an undergraduate ECE student given the timeline for the project. As such, a change in our SW/HW boundary and rescoping of the project goal was necessary. 

### Re-evaluating Scope ###
Faced with a seemingly impossible task for my current skills, even with LLM collaboration, we were forced to re-evaluate the project goal with the aim of reducing the scope and simplifying the control path.
#### Achievable Goal ####
Though our initial plan was to deploy this module on an FPGA with space constraints; we had to admit that there was an element of tunnel-vision in pursuing this goal. By instead assuming a custom chiplet, we give ourselves the freedom to have as large an array as we need for our largest layer (128x64). This significantly reduces the burden of our FSM by removing the need to perform any tiling.
#### HW/SW Boundary #### 
Additionally, the dilation calculations were an ambitious attempt to have the final be entirely self-contained. As with the previous goal, this was likely an unnecessary constraint since our goal for this project was primarily to optimize for a machine learning workload; a goal that could be accomplished with incremental iteration by starting with a more limited HW boundary before looking to expand it in future iterations. This further reduces the load on our FSM, as it no longer has to perform the complicated per-layer dilation calculations before feeding the data into our systolic array. 
#### FSM Redux ####
*Work in Progress*. With these new freedoms in mind, we set out to tackle the FSM problem once more. The result was a much more simplified FSM that should theoretically be easier to verify. Unfortunately, at this point the project was beginning to reach the end of our allotted time; though I firmly believe that verification of this module is now within my capabilities, the time remaining was not sufficient to succesfully and thoroughly verify correct functionality.

## Future Work ## 
1. **Completion of modules and verification**: Due to the running out of time, and our difficulties with verification, there is still work remaining on this first iteration of the project; primarily in completing the PCIe module, and verifying PCIe and FSM module functionality.
2. **Benchmarking hardware accelerator**: Once work has been completed on all of our modules, work can begin on testing our completed module and comparing performance results between our hardware implementation and our naive software implementation. Hopefully, the results show a successful speed up and we can move forward into future iterations focused on expanding the HW boundary as well as containing the module to be deployed on an FPGA.
3. **Software discriminator for full PWG implementation**: With a fully functional and hardware-contained generator network, we can begin work on developing the software discriminator that will train the weights needed for our network. This will involve collecting diverse data from real Eurorack synthesizers, through a very intentional collection plan, in order to teach the network to respond to the CV inputs in the same way as the target synth waveform.

## Conclusion ##
This project was an ambitious attempt to accelerate the ParallelWaveGAN algorithm for a real-time Eurorack synthesizer module. While initial profiling and design successfully identified a systolic array to solve the core MAC bottleneck, the implementation journey was stalled by the unexpected complexity of verifying the control FSM. This central challenge ultimately forced a strategic rescoping of the project's goals.

We have successfully developed and verified the core computational hardware, establishing a solid foundation for the accelerator; however, the final verification of the simplified FSM and the integration of the PCIe interface remain as the primary obstacles. Though the original vision is not yet complete, this work represents a crucial first step, and the lessons learned have provided an invaluable roadmap for the future completion of this novel synthesis tool.
___
## References ##
[1] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio," arXiv preprint arXiv:1609.03499, 2016. [Online]. Available: https://arxiv.org/abs/1609.03499.

[2] A. Nag, J. Casebeer, K. Gandhi, S. Sapra, J. Gardner, D. Parikh, and G. Singh, "AudioGen: Textually Guided Audio Generation," arXiv preprint arXiv:2209.15352, 2022. [Online]. Available: https://arxiv.org/abs/2209.15352.

[3] R. Yamamoto, E. Song, and J.-M. Kim, "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with a multi-resolution spectrogram," in 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 6174-6178.
