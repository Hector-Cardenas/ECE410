# Week 2 Challenges # 
For Week 2, we were tasked with exploring the use of perceptrons to solve binary functions. As I've had some experience with perceptrons before, my main intentions with these challenges was to play with "vibe coding", using Gemini 2.5, to create the code with only prompt guidance and minimal editing of the provided code.

The full conversation can be found [here](https://g.co/gemini/share/49894c7d5f92). While the final version of the code can be found within the directories of the same name.

## Challenge 6 ## 
Gemini handled Challenge 6 quite handedly, with the only concern being that I prompted it with a very open question; in the future, I would expect to receive better results right off the bat if I was more specific about what I was looking for (e.g. sigmoid activation function).

## Challenge 7 ##
With the perceptron implementation appearing correct for both NAND and XOR, we could then add the visualization of the decision plot through a GIF.

### NAND ###
NAND was fairly straightforward, as expected, and our resulting perceptron's evolution can be seen here:
![NAND perceptron](./Challenge_6/nand_perceptron_learning.gif)

### XOR ###
Knowing that a single perceptron is incapable, we expected XOR to not converge. Where I was surprised, however, is that the boundary did not appear to oscillate after a certain amount of epochs; this later turned out to be a result of taking GIF frames only after an epoch has updated weights for all inputs, so there was oscillation that was "unseen". XOR learning can be seen here:
![XOR perceptron](./Challenge_6/XOR_perceptron_learning.gif)

