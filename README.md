# Deep Learning Optimizers -- Complete Beginner-Friendly Notes

A simple, interview-friendly guide to **Gradient Descent, Loss
Functions, Gradients, Learning Rate, and Deep Learning Optimizers**.

> **Goal:** Understand what each concept means, how the pieces work
> together, and how the major optimizers differ --- without getting lost
> in complicated mathematics.

------------------------------------------------------------------------

## Table of Contents

1.  [What is a Model?](#1-what-is-a-model)
2.  [What is a Loss Function?](#2-what-is-a-loss-function)
3.  [What is a Gradient?](#3-what-is-a-gradient)
4.  [What is Gradient Descent?](#4-what-is-gradient-descent)
5.  [What is Learning Rate?](#5-what-is-learning-rate)
6.  [What is an Optimizer?](#6-what-is-an-optimizer)
7.  [How Training Works: Step by
    Step](#7-how-training-works-step-by-step)
8.  [Types of Gradient Descent](#8-types-of-gradient-descent)
9.  [All Important Optimizers](#9-all-important-optimizers)
10. [Important Optimizer
    Comparisons](#10-important-optimizer-comparisons)
11. [Quick Revision Table](#11-quick-revision-table)
12. [Common Interview Questions](#12-common-interview-questions)
13. [Glossary](#13-glossary)

------------------------------------------------------------------------

## 1. What is a Model?

A **machine learning model** is a system that learns patterns from
training data and uses those patterns to make predictions.

**Example:** Give a model information about houses --- such as area,
location, and number of rooms --- and it can learn to predict house
prices.

During training, the model adjusts its internal values, called **weights
and biases**, to improve its predictions.

## 2. What is a Loss Function?

**Definition:** A loss function measures how far a model's prediction is
from the actual answer.

In simple words, it tells us **how wrong the model's prediction is**.

**Example:**

-   Actual house price: ₹50 lakh
-   Predicted house price: ₹60 lakh
-   Prediction error: ₹10 lakh

A loss function converts prediction errors into a numerical value. In
general, a lower loss means better predictions on the data being
evaluated.

**Remember:** Loss function = Measures prediction error.

## 3. What is a Gradient?

**Definition:** A gradient tells us how the loss changes when a model
parameter (such as a weight) changes.

In simple words, it tells the optimizer **which direction and how
strongly to adjust a weight** to reduce the loss.

-   Positive gradient: increasing the parameter increases the loss
    locally.
-   Negative gradient: increasing the parameter decreases the loss
    locally.
-   Gradient near zero: the loss is changing very little in that
    direction.

Gradients are calculated using a process called **backpropagation** in
neural networks.

**Remember:** Gradient = Direction and rate of change of loss.

## 4. What is Gradient Descent?

**Definition:** Gradient Descent is an optimization method that
repeatedly updates model parameters in the direction that reduces the
loss.

In simple words, it helps a model **learn from its mistakes by adjusting
its weights**.

### Real-life example

Imagine standing on a hill and wanting to reach the lowest point:

1.  Check which direction slopes downward.
2.  Take a step in that direction.
3.  Check the slope again.
4.  Repeat until you reach a low point.

This is similar to how Gradient Descent reduces a model's loss.

### Basic update formula

\[ w\_{`\text{new}`{=tex}} = w\_{`\text{old}`{=tex}} -
`\alpha `{=tex}`\times `{=tex}`\frac{\partial L}{\partial w}`{=tex} \]

Where:

-   (w\_{`\text{new}`{=tex}}): updated weight
-   (w\_{`\text{old}`{=tex}}): current weight
-   (`\alpha`{=tex}): learning rate
-   (L): loss function
-   (`\frac{\partial L}{\partial w}`{=tex}): gradient of the loss with
    respect to the weight

The minus sign means we move in the direction opposite to the gradient
to reduce the loss.

**Remember:** Gradient Descent = Repeatedly move weights towards lower
loss.

## 5. What is Learning Rate?

**Definition:** Learning rate is a hyperparameter that controls how big
a step the optimizer takes when updating model weights.

In simple words, it controls **how quickly or slowly the model learns**.

### Example

Imagine walking down a hill:

-   **Very small learning rate:** You take tiny steps. Learning may be
    slow.
-   **Very large learning rate:** You take huge steps and may jump past
    the lowest point.
-   **Suitable learning rate:** You take steps that help you move
    towards the low-loss region efficiently.

  Learning rate   What can happen?
  --------------- ------------------------------------------------------
  Too small       Training can be very slow.
  Too large       Training may become unstable or overshoot a minimum.
  Suitable        Training can make steady progress.

A learning rate is usually chosen or tuned by the person training the
model. Some optimizers adapt the effective learning rate separately for
different parameters.

**Remember:** Learning Rate = Step size.

## 6. What is an Optimizer?

**Definition:** An optimizer is an algorithm that updates a model's
weights and biases using gradients to reduce the loss during training.

In simple words, an optimizer decides **how the model should adjust its
weights to learn better**.

### Why do we need an optimizer?

A model may initially make incorrect predictions. The optimizer helps it
improve by updating its parameters during training.

### Difference between Loss Function, Gradient Descent, and Optimizer

  -----------------------------------------------------------------------
  Concept                             Simple meaning
  ----------------------------------- -----------------------------------
  Loss Function                       Measures how wrong the prediction
                                      is.

  Gradient                            Tells how the loss changes when a
                                      parameter changes.

  Backpropagation                     Calculates gradients in a neural
                                      network.

  Gradient Descent                    A method for updating parameters to
                                      reduce loss.

  Optimizer                           The algorithm that carries out
                                      parameter updates; many optimizers
                                      are based on gradient descent.

  Learning Rate                       Controls the size of an update
                                      step.
  -----------------------------------------------------------------------

> Gradient Descent is a basic optimization method. Optimizers such as
> SGD, Momentum, RMSProp, and Adam use different strategies to update
> parameters.

## 7. How Training Works: Step by Step

Here is the typical training loop for a neural network:

1.  **Input data:** Provide training examples to the model.
2.  **Forward propagation:** The model makes predictions.
3.  **Calculate loss:** Compare predictions with the actual answers.
4.  **Backpropagation:** Calculate gradients showing how parameters
    affect the loss.
5.  **Optimizer update:** Use the gradients to update weights and
    biases.
6.  **Repeat:** Continue over batches and epochs to improve the model.

### Training flow

``` text
Training Data
     |
     v
Model makes predictions
     |
     v
Calculate Loss
     |
     v
Backpropagation calculates Gradients
     |
     v
Optimizer updates Weights and Biases
     |
     v
Repeat training
```

**Important terms:**

-   **Epoch:** One complete pass through the entire training dataset.
-   **Batch:** A group of training examples processed together.
-   **Iteration / Step:** One parameter update by the optimizer. The
    exact number of steps in an epoch depends on the batch strategy.

## 8. Types of Gradient Descent

The main difference between these methods is **how much training data is
used to calculate each update**.

  -------------------------------------------------------------------------------------
  Method         Simplest     How it works Main          Main limitation    Memory
                 definition                advantage                        trick
  -------------- ------------ ------------ ------------- ------------------ -----------
  **Batch        Uses the     Calculates   Stable        Can be slow and    All data
  Gradient       entire       the gradient gradient      memory-intensive   together
  Descent**      training     using all    estimate.     for large          
                 dataset for  training                   datasets.          
                 one update.  examples.                                     

  **Stochastic   Uses one     Calculates   Frequent,     Updates can be     One data
  Gradient       training     an update    inexpensive   noisy and          point
  Descent        example per  from one     updates.      fluctuate.         
  (SGD)**        update.      randomly                                      
                              selected                                      
                              example.                                      

  **Mini-Batch   Uses a small Splits data  Efficient     Batch size needs   Small
  Gradient       group of     into batches balance of    to be selected.    groups
  Descent**      examples per and updates  speed and                        
                 update.      after each   stability.                       
                              batch.                                        
  -------------------------------------------------------------------------------------

**Example:** If a dataset has 100 training examples and the mini-batch
size is 10, there are 10 updates in one epoch (assuming all examples are
used once).

> Note: In many deep learning libraries, the optimizer called `SGD` is
> commonly used with mini-batches. The name SGD is also used more
> generally for stochastic gradient descent.

## 9. All Important Optimizers

### 9.1 Momentum

**Definition:** Momentum uses information from previous gradients along
with the current gradient to make updates smoother and often faster.

**Simple explanation:** It builds momentum in directions where gradients
consistently point the same way.

**Analogy:** A ball rolling downhill gains speed as it continues moving.

-   **Main benefit:** Can speed up learning in consistent directions and
    reduce oscillations.
-   **Remember:** Previous gradient memory.

### 9.2 Nesterov Accelerated Gradient (NAG)

**Definition:** NAG is a momentum-based optimizer that calculates the
gradient at an approximate look-ahead position before making the update.

**Simple explanation:** It looks ahead before deciding how to adjust the
direction.

-   **Main benefit:** Can make more informed corrections than standard
    Momentum.
-   **Remember:** Momentum + Look Ahead.

### 9.3 AdaGrad

**Full form:** Adaptive Gradient Algorithm.

**Definition:** AdaGrad adjusts the learning rate for each parameter
using the accumulation of all its past squared gradients.

**Simple explanation:** It remembers all past squared gradients when
adjusting learning rates.

-   **Main benefit:** Can work well with sparse features and infrequent
    updates.
-   **Main limitation:** Its effective learning rates can become very
    small over time.
-   **Remember:** All past gradients.

### 9.4 RMSProp

**Full form:** Root Mean Square Propagation.

**Definition:** RMSProp adjusts the learning rate for each parameter
using an exponentially weighted moving average of recent squared
gradients.

**Simple explanation:** It gives more importance to recent squared
gradients and gradually reduces the influence of older ones.

-   **Main benefit:** Helps prevent the learning rate from shrinking as
    aggressively as it can with AdaGrad.
-   **Remember:** Recent gradients.

### 9.5 AdaDelta

**Definition:** AdaDelta adapts parameter updates using moving averages
of past squared gradients and past squared updates.

**Simple explanation:** It uses recent gradient and update information
to control how much parameters change.

-   **Main idea:** Adaptive updates without relying on one fixed global
    learning-rate scale in the same way as basic Gradient Descent.
-   **Remember:** Adaptive updates.

### 9.6 Adam

**Full form:** Adaptive Moment Estimation.

**Definition:** Adam combines moving averages of gradients and squared
gradients to adapt parameter updates.

**Simple explanation:** It combines the main ideas of Momentum and
RMSProp.

-   **First moment:** Moving average of gradients (momentum-like
    information).
-   **Second moment:** Moving average of squared gradients (adaptive
    step-size information).
-   **Main benefit:** Often works well across many deep learning tasks
    and is a popular starting choice.
-   **Common starting learning rate:** 0.001 is a frequently used
    default, not a rule.
-   **Remember:** Momentum + RMSProp.

### 9.7 AdamW

**Definition:** AdamW is an Adam variant that applies weight decay
separately from Adam's gradient-based update.

**Simple explanation:** It is Adam with decoupled weight decay, which
can help control excessively large weights and improve generalization.

-   **Main benefit:** Often useful when weight decay is important,
    including many Transformer training setups.
-   **Remember:** Adam + Decoupled Weight Decay.

### 9.8 Nadam

**Full form:** Nesterov-Accelerated Adaptive Moment Estimation.

**Definition:** Nadam combines Adam with the Nesterov look-ahead
momentum idea.

**Simple explanation:** It adds a Nesterov-style look-ahead approach to
Adam.

-   **Main idea:** Adaptive moment estimation + Nesterov momentum.
-   **Remember:** Adam + NAG idea.

## 10. Important Optimizer Comparisons

These are especially useful for interviews.

### 10.1 Momentum vs NAG

  -----------------------------------------------------------------------
  Momentum                            NAG
  ----------------------------------- -----------------------------------
  Uses the current gradient with      Uses a look-ahead position to
  accumulated momentum.               calculate the gradient.

  Builds speed using past gradient    Can correct direction by looking
  information.                        ahead.

  Memory trick: Previous gradient.    Memory trick: Look ahead.
  -----------------------------------------------------------------------

### 10.2 AdaGrad vs RMSProp

  -----------------------------------------------------------------------
  Feature                 AdaGrad                 RMSProp
  ----------------------- ----------------------- -----------------------
  Gradient information    Accumulates all past    Uses an exponentially
                          squared gradients.      weighted moving average
                                                  of squared gradients.

  Older gradients         Remain accumulated.     Their influence
                                                  gradually decreases.

  Learning-rate behavior  Can become very small   Helps avoid excessive
                          over time.              shrinking of effective
                                                  learning rates.

  Memory trick            All past gradients.     Recent gradients.
  -----------------------------------------------------------------------

**Easy way to remember:**

-   AdaGrad = All past squared gradients.
-   RMSProp = Recent squared gradients.

### 10.3 RMSProp vs Adam

  -----------------------------------------------------------------------
  RMSProp                             Adam
  ----------------------------------- -----------------------------------
  Uses a moving average of squared    Uses moving averages of gradients
  gradients.                          and squared gradients.

  Adapts parameter step sizes.        Combines momentum-like behavior
                                      with adaptive step sizes.

  Memory trick: Recent squared        Memory trick: Momentum + RMSProp.
  gradients.                          
  -----------------------------------------------------------------------

### 10.4 Adam vs AdamW

  -----------------------------------------------------------------------
  Adam                                AdamW
  ----------------------------------- -----------------------------------
  Uses adaptive moment estimates.     Uses adaptive moment estimates too.

  Weight decay, if used, may be       Applies weight decay separately
  coupled to the gradient update in   from the gradient-based update.
  common implementations.             

  Regularization behavior depends on  Provides decoupled weight decay
  implementation and configuration.   directly.
  -----------------------------------------------------------------------

### 10.5 Adam vs Nadam

  -----------------------------------------------------------------------
  Adam                                Nadam
  ----------------------------------- -----------------------------------
  Uses Adam's momentum-like first     Adds a Nesterov-style look-ahead
  moment and adaptive second moment.  momentum idea.

  No Nesterov look-ahead component in Includes Nesterov-style momentum.
  standard Adam.                      
  -----------------------------------------------------------------------

## 11. Quick Revision Table

Use this table for a quick review before an interview.

  -----------------------------------------------------------------------
  Optimizer         Simplest          Main idea         Remember it as
                    definition                          
  ----------------- ----------------- ----------------- -----------------
  **Batch Gradient  Updates using the All examples for  All data
  Descent**         entire dataset.   each update.      

  **SGD**           Updates using one One example per   One data point
                    example at a      update.           
                    time.                               

  **Mini-Batch      Updates using a   A batch per       Small group
  Gradient          small group of    update.           
  Descent**         examples.                           

  **Momentum**      Uses previous     Builds speed in   Previous gradient
                    gradients with    consistent        memory
                    the current       directions.       
                    gradient.                           

  **NAG**           Momentum with a   Looks ahead       Momentum + Look
                    look-ahead        before correcting Ahead
                    gradient.         direction.        

  **AdaGrad**       Adjusts learning  Accumulated       All past
                    rates using all   squared           gradients
                    past squared      gradients.        
                    gradients.                          

  **RMSProp**       Adjusts learning  Moving average of Recent gradients
                    rates using       squared           
                    recent squared    gradients.        
                    gradients.                          

  **AdaDelta**      Adapts updates    Moving averages   Adaptive updates
                    using recent      of gradients and  
                    gradients and     updates.          
                    updates.                            

  **Adam**          Combines Momentum Moving averages   Momentum +
                    and RMSProp       of gradients and  RMSProp
                    ideas.            squared           
                                      gradients.        

  **AdamW**         Adam with         Separates weight  Adam + Weight
                    decoupled weight  decay from        Decay
                    decay.            gradient update.  

  **Nadam**         Adam with         Adaptive          Adam + NAG
                    Nesterov-style    moments +         
                    momentum.         look-ahead        
                                      momentum.         
  -----------------------------------------------------------------------

### Optimizer family map

``` text
Gradient-based optimization
|
|-- Basic Gradient Descent methods
|   |-- Batch Gradient Descent
|   |-- Stochastic Gradient Descent (SGD)
|   `-- Mini-Batch Gradient Descent
|
|-- Momentum-based ideas
|   |-- Momentum
|   `-- Nesterov Accelerated Gradient (NAG)
|
|-- Adaptive learning-rate methods
|   |-- AdaGrad
|   |-- RMSProp
|   `-- AdaDelta
|
`-- Combined / advanced methods
    |-- Adam       (Momentum-like + RMSProp-like ideas)
    |-- AdamW      (Adam + decoupled weight decay)
    `-- Nadam      (Adam + Nesterov-style momentum)
```

This is a conceptual grouping. Some optimizers combine ideas from more
than one family.

## 12. Common Interview Questions

### Q1. What is Gradient Descent?

Gradient Descent is an optimization method that repeatedly updates model
parameters in the direction that reduces the loss.

### Q2. What is an optimizer?

An optimizer is an algorithm that updates model weights and biases using
gradients to reduce the loss during training.

### Q3. What is learning rate?

Learning rate is a hyperparameter that controls the size of each
parameter update.

### Q4. What is the difference between Momentum and NAG?

Momentum uses the current gradient along with accumulated momentum. NAG
calculates the gradient at an approximate look-ahead position.

### Q5. What is the difference between AdaGrad and RMSProp?

AdaGrad accumulates all past squared gradients, which can make its
effective learning rates very small. RMSProp uses a moving average of
recent squared gradients.

### Q6. What is Adam?

Adam stands for Adaptive Moment Estimation. It uses moving averages of
gradients and squared gradients, combining Momentum-like and
RMSProp-like ideas.

### Q7. What is the difference between Adam and AdamW?

AdamW decouples weight decay from the gradient-based update, unlike the
coupled form of weight decay used in some Adam implementations.

### Q8. Which optimizer is commonly used in deep learning?

Adam is a popular starting choice because it often works well across
many tasks. AdamW is also widely used, especially in Transformer
training. The best optimizer depends on the model, dataset, and training
setup.

## 13. Glossary

  -----------------------------------------------------------------------
  Term                                Simple meaning
  ----------------------------------- -----------------------------------
  **Weight**                          A learnable value that controls the
                                      influence of an input.

  **Bias**                            A learnable value added to a
                                      model's weighted input.

  **Parameter**                       A value learned by the model, such
                                      as a weight or bias.

  **Hyperparameter**                  A setting chosen before or during
                                      training, such as learning rate or
                                      batch size.

  **Loss**                            A numerical measure of prediction
                                      error.

  **Gradient**                        Direction and rate of change of
                                      loss with respect to parameters.

  **Backpropagation**                 The process used to calculate
                                      gradients in neural networks.

  **Epoch**                           One complete pass through the
                                      training dataset.

  **Batch size**                      Number of examples processed
                                      together in one batch.

  **Iteration / Step**                One optimizer update.

  **Convergence**                     The training process settling
                                      towards a point where improvement
                                      becomes small.

  **Oscillation**                     Updates moving back and forth
                                      instead of smoothly towards a
                                      low-loss region.

  **Local minimum**                   A point with lower loss than nearby
                                      points, but not necessarily the
                                      lowest possible loss.

  **Global minimum**                  The lowest loss across the entire
                                      search space.

  **Vanishing gradients**             Gradients become extremely small,
                                      making earlier layers learn very
                                      slowly.

  **Exploding gradients**             Gradients become extremely large,
                                      causing unstable updates.

  **Gradient clipping**               Limits gradient size to help
                                      prevent excessively large updates.

  **Weight decay**                    A regularization technique that
                                      discourages weights from becoming
                                      excessively large.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Final Memory Tricks

-   **Gradient Descent:** Move towards lower loss.
-   **Learning Rate:** Size of the step.
-   **Optimizer:** Decides how weights are updated.
-   **Batch GD:** All data.
-   **SGD:** One data point.
-   **Mini-Batch GD:** Small group.
-   **Momentum:** Previous gradient memory.
-   **NAG:** Momentum + Look Ahead.
-   **AdaGrad:** All past squared gradients.
-   **RMSProp:** Recent squared gradients.
-   **AdaDelta:** Adaptive updates.
-   **Adam:** Momentum + RMSProp ideas.
-   **AdamW:** Adam + Decoupled Weight Decay.
-   **Nadam:** Adam + Nesterov-style momentum.

> **Best learning strategy:** First understand the simple idea behind
> each optimizer. Then learn its formula and practical implementation.
> Understanding the difference is more useful than memorizing names
> alone.

------------------------------------------------------------------------

*Beginner-friendly study notes for Machine Learning and Deep Learning
interview preparation.*
