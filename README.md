# Text-summarization

## Outline

### Essential Deep Learning topics for text summarization
* tokenization
* Normalization
* TF/IDF
* stopwords
* Jaccard Similarity
* Cosine Similarity
* Hadamard Product
* one hot vector
* Sigmoid Activation Function
* ReLU Activation Function
* tanh activation function
* softmax function
* Recurrent neural network (RNN)
* Gated Recurrent Unit(GRU)
* LSTM
* Feed forward neural network
* Bidirectional RNN (BRNN)
* Convolution Neural Network(CNN)
* Encoder-Decoder Model
* Attention Model
* Seq to Seq model
* Transformer
* Teacher Forcing
* Beam Search

### Keyword Extraction
* Rules Based Keyword Extraction
* Using support vector machine(SVM)
* Rake Algorithm

### Word Embedding
* Bag of words
* TF-IDF

### Word2vec
* Skip gram
* Cbow
* Loss function
* Difference between CBOW and skip-gram model
* Glove
* Fasttext

### Sentence Embedding
* Doc2vec
* PVDM(Distributed Memory version of Paragraph Vector)
* PVDOBW( Distributed Bag of Words version of Paragraph Vector)
* SentenceBERT
* part1: Sentence Encoder(LSTM and GRU) (Infersent)
* part2: Sentence Encoder (BiLSTM with max/mean pooling) (Infersent)
* part3: Sentence Encoder (Self-attentive network) (Infersent)
* part4: Sentence Encoder( Hierarchical ConvNet) (Infersent)
* Natural Language Inference Classifier(NLI)
* Universal Sentence Encoder

### Models
* Graph Base Extractive Summarization
* Feature Base Extractive Summarization
* Topic Base Extractive Summarization
* BSTM base Extractive Summarization
* Grammar Base Extractive summarization
* Neural network base Extractive Summarization
* Encoder Decoder Abstractive Model with RNN
* Abstractive Summarization with Attentive Recurrent Neural Networks
* Neural Attention Model
* Sequence-to-sequence RNNs and Beyond
* Summarization with Pointer-Generator Networks
* LSTM and CNN base abstractive model
* Selective Encoding for Abstractive Sentence Summarization
* A Deep Reinforced Model for Abstractive Summarization
* Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting
* Query Focused Abstractive Summarization
* Fine-tune BERT for Extractive Summarization
* Pretraining-Based  Natural  Language  Generation for Text Summa-rization(Abstractive)


<details>
  <summary> :point_right: TF/IDF</summary>
TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
The full form of TF is Term Frequency (TF).The full form of IDF is Inverse Document Frequency.

**TF** = no. of times term occurrences in a document / total number of words in a document

**IDF** = log base e (total number of documents / number of documents which are having term )

There are lots of library in every language for measuring TF-IDF.
In NLP, TF/IDF is a necessary step for information retrieval and keyword Extraction. In-Text summarization, as we need to find information from sentences Tf/IDF does that for us. It was much simple and faster than any other model.
</details>

<details>
  <summary> :point_right: Stopwords</summary>
Stopwords are the words in any language which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence.

Suppose, a sentence **"Dhaka is the capital city of bangladesh"** . after removing stop words it looks like ** Dhaka capital city bangladesh** . See, it doesn't change the meaning.
So why we need it?

    1. Stopwords are rapidly used in sentences. So for getting usefull information, this always create problems.
    2. Without stopwords, our model computational power will be better.
    3. without stopwords, we can get better accuracy.


In english language nltk library marked **127 words** as a stop words.

```
i, me, my, myself, we, our, ours, ourselves, you, your, yours, yourself,yourselves,
he, him, his, himself, she, her, hers, herself, it, its, itself, they, them,their, 
theirs, themselves, what, which, who, whom, this, that, these, those, am, is,are, 
was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, 
the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, 
against, between, into, through, during, before, after, above, below, to, from, 
up, down, in, out, on, off, over, under, again, further, then, once, here, there,
when, where, why, how, all, any, both, each, few, more, most, other, some, such,
no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don,
should, now
```
we can customize it.

</details>


<details>
  <summary> :point_right: Jaccard Similarity</summary>

The **Jaccard index**, also known as **Intersection over Union** and **the Jaccard similarity coefficient** (originally given the French name coefficient de communauté by Paul Jaccard), is a statistic used for gauging the similarity and diversity of sample sets.

![equation](https://latex.codecogs.com/svg.latex?J(A,B)=%20\frac{|A\cap%20B|}{|A\cup%20B|}%20=%20\frac{|A\cap%20B|}{|A|+|B|-|A\cap%20B|)


![](https://latex.codecogs.com/svg.latex?d_j(A,B)%20=%201-J(A,B)=\frac{|A\cup%20B|-|A\cap%20B|}{|A\cup%20B|})

Here, ![](https://latex.codecogs.com/svg.latex?A%20\cap%20B) represents the number of common words between the set A and B;
![](https://latex.codecogs.com/svg.latex?A%20\cup%20B) represents total number of distinct words in the set A and B.
Why we need it?

Finding the relation between two sentences or vectors.
</details>

<details>
  <summary> :point_right: Cosine Similarity</summary>

As like as jaccard Similarity, Cosine similarity also works for finding relation of vectors.
Given two vectors of attributes, A and B, the cosine similarity, ![](https://latex.codecogs.com/svg.latex?cos(\theta)) , is represented using a dot product and magnitude as

![](https://latex.codecogs.com/svg.latex?similarity%20=%20\cos%20(\theta)%20=%20\frac{A.B}{||A||\hspace{2pt}||B||}%20=%20\frac{\sum^n_{i=1}%20A_i%20B_i}{\sqrt{\sum^n_{i=1}%20A^2_i}\sqrt{\sum^n_{i=1}%20B^2_i}})

where A_i and B_i are components of vector A and B respectively.

</details>


<details>
  <summary> :point_right: Hadamard Product</summary>

In mathematics, the Hadamard product (also known as the element-wise, entrywise or Schur product) is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands, where each element i, j is the product of elements i, j of the original two matrices.

![](img/sum42.png)

</details>


<details>
  <summary> :point_right: ReLU Activation Function</summary>
ReLU stands for rectified linear unit, and is a type of activation function. It is widely used in Activation Function.

![](img/sum26.png)
why do we use ReLU, if we already have sigmoid?


    1. it takes less time to train or run.
    2. It is very faster than other models.  Cause there is no complex equation. It doesn’t have the vanishing gradient problem suffered by other activation functions like sigmoid or tanh.

But in relu, if the number is negative it will make number 0. So, if all number is negative, it don't activated the function. And it's called Dying ReLU.

ReLU has so many version. **Leaky ReLU** is mostly used in deep learning. Instead of making 0, it makes negative value to multiple of a constant factor.
</details>


<details>
  <summary> :point_right: tanh activation function</summary>

In sigmoid function, every value transform 0 to 1. In tanh function instead of 0 to 1, it makes -1 to 1.

![](img/gg.jpg)

</details>

<details>
  <summary> :point_right: softmax function</summary>

The softmax function squashes the outputs of each unit to be between 0 and 1, just like a sigmoid function. But it also divides each output such that the total sum of the outputs is equal to 1.

![](img/sum23.jpg)

The output of the softmax function is equivalent to a categorical probability distribution, it tells you the probability that any of the classes are true.

Mathematically the softmax function is shown below, where z is a vector of the inputs to the output layer (if you have 10 output units, then there are 10 elements in z). And again, j indexes the output units, so j = 1, 2, ..., K.

![](https://latex.codecogs.com/svg.latex?\sigma%20(z)_j%20=%20\frac{e^{z_j}}{\sum^K_{k=1}e^{z_k}})

Suppose, we have an image where '4' digit is written on it. Now we want character recognition from that image. And we do all process for recognition of the character and make a vector. Now after using softmax function in our result it looks like,

![](img/sum24.png)

So softmax function is basically making the vector into the probabilistic vector.
</details>

<details>
  <summary> :point_right: Recurrent neural network (RNN)</summary>

Recurrent Neural Networks (RNNs) are widely used for data with some kind of sequential structure. For instance, time series data has an intrinsic ordering based on time. Sentences are also sequential, "I love dogs" has a different meaning than "Dogs I love." So if you are working with sequential model, RNN might be the solution for that.\\
A simple architecture for RNN

![](img/sum27.png)

1. **Vector h :** is the output of the hidden state after the activation function has been applied to the hidden nodes. 
   
2. **Matrices Wx, Wy, Wh:** — are the weights of the RNN architecture which are shared throughout the entire network. The model weights of Wx at t=1 are the exact same as the weights of Wx at t=2 and every other time step.
   
3. **Vector Xi:** is the input to each hidden state where i=1, 2,…, n for each element in the input sequence.

We can make divide full architecture with two part.
1. Forward Propagation
2. Backpropagation through time

**Forward Propagation:**

In Forward Propagation, we calculate everything. Like, output,hidden state,loss function,gradient etc.

![](img/sum28.png)
Here,
* **k** is the dimension of the input vector Xi
  
* **d** is the number of hidden nodes

Loss function is calculated by multi-class cross entropy loss function.

![](https://latex.codecogs.com/svg.latex?L_t(y_t,\hat%20y_t)%20=%20-y_t%20log(\hat%20y_t))

OverAll loss function is calculated by below equation.

![](https://latex.codecogs.com/svg.latex?L_{total}(y,\hat%20y)%20=%20-\sum^n_{t=1}%20y_t%20\log(\hat%20y_t))

For gradient calculation of ![](https://latex.codecogs.com/svg.latex?W_y),

![](https://latex.codecogs.com/svg.latex?\frac{\partial%20L_{total}}{\partial%20W_Y}%20=%20\frac{\partial%20L_1}{\partial%20\hat%20y_1}%20\frac{\partial%20\hat%20y_1}{\partial%20z_1}\frac{\partial%20z_1}{\partial%20W_Y}%20+%20\frac{\partial%20L_2}{\partial%20\hat%20y_2}%20\frac{\partial%20\hat%20y_2}{\partial%20z_2}\frac{\partial%20z_2}{\partial%20W_Y}%20+%20\frac{\partial%20L_3}{\partial%20\hat%20y_3}%20\frac{\partial%20\hat%20y_3}{\partial%20z_3}\frac{\partial%20z_3}{\partial%20W_Y}%20=%20\sum^n_{t=1}%20\frac{\partial%20L_t}{\partial%20W_Y})

For gradient Calculation of ![](https://latex.codecogs.com/svg.latex?W_x),

![](https://latex.codecogs.com/svg.latex?\frac{\partial%20L_{total}}{\partial%20W_X}%20=%20\sum^n_{t=1}%20\sum^n_{k=0}%20\frac{\partial%20L_t}{\partial%20\hat%20y_t}%20\frac{\partial%20\hat%20y_t}{\partial%20z_t}\frac{\partial%20z_t}{\partial%20h_t}\frac{\partial%20h_t}{\partial%20h_k}\frac{\partial%20h_k}{\partial%20W_x})

**BackPropagation through time:**

In backward Propagation, It's updated ![](https://latex.codecogs.com/svg.latex?W_x) , ![](https://latex.codecogs.com/svg.latex?W_y) ,![](https://latex.codecogs.com/svg.latex?W_h) value based on gradients.

So, overall process look like,

1. Initialize weight matrices Wx, Wy, Wh randomly

2. Forward propagation to compute predictions

3. Compute the loss

4.Backpropagation to compute gradients

5. Update weights based on gradients

6. Repeat steps 2–5

A problem that RNNs face, which is also common in other deep neural nets, is the **vanishing gradient problem** . Vanishing gradients make it difficult for the model to learn long-term dependencies. For example, if an RNN was given this sentence,

```
The brown and black dog,which was playing with cat, was a German shepherd
```
in this sentence, when we want to predict 'shepherd', it needs the input brown. But 'brown' is too far from 'shepherd'. A long term dependency occurs. And it creates gradient loss. Cause gradient follows chain rule.
However, there have been advancements in RNNs such as gated recurrent units (GRUs) and long short term memory (LSTMs) that have been able to deal with the problem of vanishing gradients.

</details>


<details>
  <summary> :point_right: Gated Recurrent Unit(GRU)</summary>

To solve the vanishing gradient problem of a standard RNN, GRU uses, so-called, update gate and reset gate. Basically, these are two vectors which decide what information should be passed to the output. The special thing about them is that they can be trained to keep information from long ago, without washing it through time or remove information which is irrelevant to the prediction.
![](img/sum29.png)

**Update Gate:**
We start with calculating the update gate $Z_t$ for time step t using the,

![](https://latex.codecogs.com/svg.latex?z_t%20=%20\sigma%20(W^{(z)}%20x_t%20+%20U^{(z)}%20h_{t-1}))


The update gate helps the model to determine how much of the past information (from previous time steps) needs to be passed along to the future.

**Reset Gate:** this gate is used from the model to decide how much of the past information to forget. To calculate it, we use:

![](https://latex.codecogs.com/svg.latex?r_t%20=\sigma%20(W^{(r)}x_t%20+%20U^{(r)}h_{t-1}))


**Current memory content:**
This is a big process, because output depends on it. We can divide it with 4 section.

![](https://latex.codecogs.com/svg.latex?h_t%20=\tanh%20(Wx_t%20+%20r_t%20\odot%20Uh_{t-1}))

1. Multiply the input ![](https://latex.codecogs.com/svg.latex?x_t) with a weight W and ![](https://latex.codecogs.com/svg.latex?h_{t-1}) with a weight U.

2. Calculate the Hadamard (element-wise) product between the reset gate ![](https://latex.codecogs.com/svg.latex?r_t) and ![](https://latex.codecogs.com/svg.latex?Uh_{t-1}). That will determine what to remove from the previous time steps.
3. Sum up the results of step 1 and 2.
4. Apply the nonlinear activation function tanh.

**Final memory at current time step:** This is the output state. Here we computes output from previous computation.

![](https://latex.codecogs.com/svg.latex?h_t%20=%20z_t%20\odot%20h_{t-1}%20+%20(1-z_t)\odot%20h_t)

1. Apply element-wise multiplication to the update gate ![](https://latex.codecogs.com/svg.latex?z_t) and ![](https://latex.codecogs.com/svg.latex?h_{t-1}) .
2. Apply element-wise multiplication to ![](https://latex.codecogs.com/svg.latex?1-z_t) and ![](https://latex.codecogs.com/svg.latex?h`_t) .
3. Sum the results from step 1 and 2.

</details>


<details>
  <summary> :point_right: LSTM</summary>
Long Short Term Memory are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter \& Schmidhuber (1997) [20], and were refined and popularized by many people in following work. Lstm architecture is similar to GRU.

![](img/sum30.png)

We can divide whole architecture with 4 section,

**1. Forget gate( ![](https://latex.codecogs.com/svg.latex?f_t) ):**
The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ![](https://latex.codecogs.com/svg.latex?h_(t-1)) and ![](https://latex.codecogs.com/svg.latex?x_t), and outputs a number between 0 and 1 for each number in the cell state ![](https://latex.codecogs.com/svg.latex?C_{t-1}). 
**1** represents “completely keep this” while a **0** represents “completely get rid of this.”

![](https://latex.codecogs.com/svg.latex?f_t=\sigma%20(x_tU^f%20+%20h_{t-1}W^f))

**2. input gate (![](https://latex.codecogs.com/svg.latex?i_t) ):** The next step is to decide what new information we’re going to store in
the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” 
decides which values we’ll update. Next, a tanh layer creates a vector of new candidate 
values, ![](https://latex.codecogs.com/svg.latex?\tilde%20C_t), 
that could be added to the state. In the next step, we’ll combine these two to create an 
update to the state.

![](https://latex.codecogs.com/svg.latex?i_t=\sigma%20(x_tU^i%20+%20h_{t-1}W^i))

![](https://latex.codecogs.com/svg.latex?\tilde%20C_t=\tanh%20(x_tU^g%20+%20h_{t-1}W^g))

**3. update old state:** It’s now time to update the old cell state, ![](https://latex.codecogs.com/svg.latex?C_{t-1}), into the new cell state ![](https://latex.codecogs.com/svg.latex?C_t). The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by ![](https://latex.codecogs.com/svg.latex?f_t), forgetting the things we decided to forget earlier. Then we add ![](https://latex.codecogs.com/svg.latex?i_t%20\ast%20\tilde%20C_t). This is the new candidate values, scaled by how much we decided to update each state value.

![](https://latex.codecogs.com/svg.latex?C_t=\sigma%20(f_t*C_{t-1}%20+%20i_t*%20\tilde%20C_t))

**4. output Gate:** Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between - and 1 and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

![](https://latex.codecogs.com/svg.latex?o_t=\sigma%20(x_tU^o%20+%20h_{t-1}W^o))

![](https://latex.codecogs.com/svg.latex?h_t=\tanh%20(C_t)*%20o_t)

</details>


<details>
  <summary> :point_right: Feed forward neural network</summary>

Feed forward neural network is a simple one directional neural network. In RNN there a backward propagation but here only forward propagation. But in this neural network there is a lots of hidden layer for decision making. for computing hidden layer, in this network sigmoid function is used. output layer is only 1 and 0 activation function. Thats why this network is simplest neural network.

![](img/sum50.png)
</details>

<details>
  <summary> :point_right: Bidirectional RNN (BRNN)</summary>

In 1997 Schuster & mike proposed bidirectional model of RNN [26].

![](img/sum35.png)

The idea is to split the state neurons of a regular RNN in a part that is responsible for the positive time direction (forward states) and a part for the negative time direction (backward states). Outputs from forward states are not connected to inputs of backward states, and vice versa.

**FORWARD PASS:** \
Run all input data for one time slice through 
![](https://latex.codecogs.com/svg.latex?1\leq%20t\leq%20T) the BRNN and determine all
predicted outputs.
1. Do forward pass just for forward states (from t=1
to t=T ) and backward states (from t=T to t=1).
2. Do forward pass for output neurons.

**BACKWARD PASS:** \
Calculate the part of the objective function derivative
for the time slice ![](https://latex.codecogs.com/svg.latex?1\leq%20t\leq%20T) used in the forward pass.
1. Do backward pass for output neurons.
2. Do backward pass just for forward states (from t=T
to t=1 ) and backward states (from t=1
to t=T).
</details>

<details>
  <summary> :point_right: Convolution Neural Network(CNN)</summary>

CNN was introduced by LeCun at 1998 [27]. Images recognition, images classifications. Objects detections, recognition faces etc., are some of the areas where CNNs are widely used.

![](img/sum37.jpeg)

We already know what is **Relu** Layer.

**Padding:** \
Sometimes filter does not fit perfectly fit the input image. We have two options for padding:

1. Pad the picture with zeros (zero-padding) so that it fits
   
2. Drop the part of the image where the filter did not fit. This is called valid padding which keeps only valid part of the image.

**pooling:**
Pooling layers section would reduce the number of parameters when the images are too large. Spatial pooling also called subsampling or downsampling which reduces the dimensionality of each map but retains important information. Spatial pooling can be of different types:

1. Max Pooling
2. Average Pooling
3. Sum Pooling

Max Pooling is the most common among others.Max pooling takes the largest element from the rectified feature map. Taking the largest element could also take the average pooling. Sum of all elements in the feature map call as sum pooling.

![](img/sum36.png)


**Flattening:** An image is nothing but a matrix of pixel values. 
In Neural network it is difficult to work with 3-dimentional matrix. 
So we make 2D matrix to 1D matrix by flattened for feeding the next layer.

![](img/sum38.png)


**Fully Connected Layer:** \
The layer we call as FC layer, we flattened our matrix into vector and feed it into a 
fully connected layer like a neural network. Every node connected to others hidden layer node.

**Process:** 

1. Provide input image into convolution layer
2. Choose parameters, apply filters with strides, padding if requires. Perform convolution on the image and apply ReLU activation to the matrix.
3. Perform pooling to reduce dimensionality size
4. Add as many convolutional layers until satisfied
5. Flatten the output and feed into a fully connected layer (FC Layer)
6. Output the class using an activation function (Logistic Regression with cost functions) and classifies images.


</details>

<details>
  <summary> :point_right: Encoder-Decoder Model</summary>

The encoder-decoder model is composed of encoder and decoder like its name. The encoder converts an input document to a latent representation (vector), and the decoder generates a summary by using it.
It was develop by google AI [21] and also implemeted it in the google play service. In our message section, this technique is used. So we already know what is it.

![](img/sum31.png)

In Encoder part, machine will learn context sequentially. We can use any sequential model 
here for learning purpose. Like, RNN,LSTM,CNN,Seq to seq, GRU etc.
In Decoder part, machine Decode what he learn and it depends on what we want. 
It can be translation,summary,question-answer etc.

In NLP, Encoder Decoder model is hugely used by researcher Cause of good accuracy.
In summarization We will see how to summary a text by encoder decoder.

</details>




<details>
  <summary> :point_right: Attention Model</summary>

Suppose we want to translate English to bangla.

![](img/sum33.png)

In this figure, You can see when we translate 'Hasina', We don't need to focus 'Bangladesh'. When we decode something, we don't need to focus all the word. Only we need to focus those word who was closely connected. Here 'Attention' mechanism comes.

![](img/sum34.png)

In This figure, we have a bidirectional RNN for encoder. Now when we decode, 
there is a value comes from every hidden state. Because we need to know which word
we need to focus. This model is proposed by Bahdanau at 2014 [25].

In this architecture, they define each conditional probability,

![](https://latex.codecogs.com/svg.latex?p(y_i|y_1,...,y_{i-1},X)=g(y_{i-1},s_i,c_i))

where ![](https://latex.codecogs.com/svg.latex?s_i) is an RNN hidden state for time i, computed by

![](https://latex.codecogs.com/svg.latex?s_i=f(s_{i-1},y_{i-1},c_i))

The context vector ![](https://latex.codecogs.com/svg.latex?c_i) is, then, computed as a weighted sum of these annotations ![](https://latex.codecogs.com/svg.latex?h_i)

![](https://latex.codecogs.com/svg.latex?c_i=\sum^{T_x}_{j=1}\alpha_{ij}h_j)

The weight ![](https://latex.codecogs.com/svg.latex?\alpha_{ij}) of each annotation ![](https://latex.codecogs.com/svg.latex?h_j) is computed by

![](https://latex.codecogs.com/svg.latex?\alpha_{ij}=\frac{\exp(e_{ij})}{\sum^{T_x}_{k=1}\exp(e_{ik})})

where,

![](https://latex.codecogs.com/svg.latex?e_{ij}=a(s_{i-1},h_j))

![](https://latex.codecogs.com/svg.latex?h_j) is the hidden state of RNN Model


</details>



<details>
  <summary> :point_right: Seq to Seq model</summary>

This model is one of the famous model in NLP. Why? It can solve different types of difficult task that we can't do with other language. This model is also very fast.
This model was introduced by google at 2014 [30]. This model also based on encoder-decoder architecture.
But we need to give input sequentially and output layer will give us output sequentially. So, there is no big difference.

We can use different types model in encoder and decoder here. Like,

1. RNN or Bidirectional RNN
2. LSTM or Bidirectional LSTM
3. GRU or Bidirectional GRU

</details>


<details>
  <summary> :point_right: Transformer</summary>

Transfer learning or transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. How? why is this model much important and famous?  This model was introduced by Vaswani at 2017 in his famous paper "Attention is all you need" [37]. They used multi-head attention instead of single-head attention.

![](img/sum51.png)

From this figure, we can guess how this architecture works.\
We already know about **"Embedding", "Feed Forward neural network",
"Softmax","Normalization"**. Now we will know about **"Multi-Head Attention"** and **"Positional Encoding"**.

**Positional Embedding:** Since this model contains no recurrence and no convolution,
in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position
of the
tokens in the sequence. To this end, they add "positional encodings" to the input embeddings 
at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension ![](https://latex.codecogs.com/svg.latex?d_{model})
as the embeddings, so that the two can be summed. They used sine and cosine function for different frequency.

![](https://latex.codecogs.com/svg.latex?PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}}))

![](https://latex.codecogs.com/svg.latex?PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}}))

**Multi-Head Attention:** In single attention, if the input consists of
queries and keys of dimension ![](https://latex.codecogs.com/svg.latex?d_k), and values of dimension ![](https://latex.codecogs.com/svg.latex?d_v), We compute the dot products of the query with all keys, divide each by 
![](https://latex.codecogs.com/svg.latex?\sqrt{d_k}), and apply a softmax function to obtain the weights on the
values.

![](https://latex.codecogs.com/svg.latex?Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V)

In Multi-Head Attention model, instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries,
we found it beneficial to linearly project the queries, keys and values h times with different, learned
linear projections to ![](https://latex.codecogs.com/svg.latex?d_k), ![](https://latex.codecogs.com/svg.latex?d_k) and ![](https://latex.codecogs.com/svg.latex?d_v) dimensions, respectively.
Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this

![](https://latex.codecogs.com/svg.latex?MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O)

where,

![](https://latex.codecogs.com/svg.latex?head_i=Attention(QW^Q_i,KW^k_i,VW^V_i))

We have to fixed head size.

**Self Attention:** Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.Self-attention allows the model to look at the other words in the input sequence to get a better understanding of a certain word in the sequence. Now, let’s see how we can calculate self-attention.

1. First, we need to create three vectors from each of the encoder’s input vectors: 
    Query Vector,
Key Vector,
Value Vector.
2. Next, we will calculate self-attention for every word in the input sequence
3. Consider this phrase – “Legends Never die”. To calculate the self-attention for the first word “Legends”, we will calculate scores for all the words in the phrase with respect to “legends”. This score determines the importance of other words when we are encoding a certain word in an input sequence

![](img/sum54.png)


**Encoder:** The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position wise fully 
connected feed-forward network. We employ a residual connection [10] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension ![](https://latex.codecogs.com/svg.latex?d_{model}) = 512.

**Decoder:** The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.


</details>


<details>
  <summary> :point_right: Teacher Forcing</summary>

Teacher forcing is a strategy for training recurrent neural networks that uses model output from a prior time step as an input. 
Teacher forcing works by using the actual or expected output from the training dataset at the current time step y(t) as 
input in the next time step X(t+1), rather than the output generated by the network.

Lets take an example,

"Dhaka is the capital city of bangladesh". We want to feed this sentence in RNN. We must add [start] and [end] 
token in the sentence. 

now the sentence looks like, "[Start] Dhaka is the capital city of bangladesh [End]".

When we give [Start] token, if it gives "The" as a output, and if we continue this process,
whole sequence will be wrong. The model is in the off track and is going to get punished for every subsequent 
word it generates. This makes learning slower and the model unstable.

Here Teacher Forcing comes, When we calculate error and found we are in off track 
we will replace it with others word. It makes faster the model. 

Lots of teacher forcing approach were invented in recent years, we can use any of it.

</details>


<details>
  <summary> :point_right: Beam Search</summary>

Let's Think about, after calculating a output vector, how we can predict correct sentence. 
As like as Teacher forcing, If one word is incorrect, entire sequence can be incorrect. 
After using softmax function, If we take always highest value , it may be give wrong value. In greedy search, 
always take highest softmax value.

Instead of taking only maximum value we can take few value with highest softmax value. 
Suppose our Beam=3, We will take 3 value from output vector and make a bfs tree with it.

![](img/sum55.jpg)


After taking each of them, we will calculate conditional probability with them in each time step.
And we will take maximum probabilistic Sequence for our output.

</details>

### Keyword Extraction techniques

<details>
  <summary> :point_right: Introduction</summary>
Keyword extraction is tasked with the automatic identification of terms that best describe the subject of a document. Key phrases, key terms, key segments or just keywords are the terminology which is used for defining the terms that represent the most relevant information contained in the document.(source wiki). If you get good keyword from text, your answer will be more accurate.
Maximum extractive and abstractive models depend on keyword Extraction.

</details>


<details>
  <summary> :point_right: Rules Based Keyword Extraction</summary>

Rules-based Keyword extraction basically conditional or pattern type.
In 2008  Khoury, Richard, Fakhreddine Karray, and Mohamed S. Kamel published a paper about a rules-based model in keyword extraction using parts of speech hierarchy [1]. In this model, they describe, parts of speech have an important role in keyword extraction. They made a supervised learning model and a training corpus.

1. words are split according to their parts of speech tag. 
2. words are divided into different categories.
3. rules and condition check here which looks like a tree, every edge have a different condition.
4. if the word goes to the leaf then we will take it as a keyword.

</details>


<details>
  <summary> :point_right: Using support vector machine(SVM)</summary>

In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with 
associated learning algorithms that analyze data used for classification and regression analysis.
In 2006, Zheng and khuo made a keyword extraction model using SVM [2].

1. **Pre-processing:** For a document, we conduct the sentence split, word tokenization and POS (part-of-speech) tagging by using GATE. Next, employ Linker to analyze the dependency relationships between words in each sentence. After that, we employ tri-gram to create candidate phrases and then filter the phrases whose frequencies are below a predefined threshold. We also exclude the common words in the stop-words list. We conduct gentle stemming by using WordNet. Specifically, we only stem the plural noun, gerund, and passive infinitive by their dictionary form. Finally, we obtain a set of ‘keyword candidate’ for later processing.
2. **Feature extraction:** The input is a bag of words/phrases in a document. We make use of both local context information(POS Feature, Linkage Features, Contextual TFIDF Feature) and global context information(TFIDF Feature, First Occurrence Feature, Position Features) of a word/phrase in a document to define its features. The output is the feature vectors, and each vector corresponds to a word/phrase.
3. **Learning:** The input is a collection of feature vector by step (2). We construct a SVM model that can identify the keyword. In the SVM model, we view a word/phrase as an example, the words/phrases labeled with ‘keyword’ as positive examples, and the other words/phrases as negative examples. We use the labeled data to train the SVM model in advance.
4. **Extraction:** The input is a document. We employ the preprocessing and the feature extraction on it, and obtain a set of feature vectors. We then predict whether or not a word/phrase is a keyword by using the SVM model from step (3). Finally, the output is the extracted keywords for that document.

</details>

<details>
  <summary> :point_right: Rake Algorithm</summary>
In 2010 Stuart Rose invented new techniques in keyword extraction name Rapid automatic keyword extraction(RAKE).
It was one of the popular algorithms used in finding important keywords. In-text Summarization, this algorithm much popular in keyword extraction.

**Candidate keywords:**  Split the words by tokenizing them. Remove the stop word and delimiters.
All other word is candidate words. Be careful about the position. Cause position matters when we rank them.

**Keyword scores:** Keyword Scores were calculated from the co-occurrence graph.

Suppose we have some word token after stop word remove.
```
Compatibility – systems – linear constraints – set – natural numbers – Criteria –
compatibility – system – linear Diophantine equations – strict inequations – nonstrict
inequations – Upper bounds – components – minimal set – solutions – algorithms –
minimal generating sets – solutions – systems – criteria – corresponding algorithms –
constructing – minimal supporting set – solving – systems – systems
```

For co-occurrence graph we need sorted, unique word. now it looks like,

```
algorithms – bounds – compatibility – components – constraints – constructing – corresponding –
criteria – diophantine – equations – generating – inequations – linear – minimal – natural –
nonstrict – numbers – set – sets – solving – strict – supporting – system – systems – upper
```

In co-occurrence we check for every i,j cell. if i'th row and j'th coloum are related or co-related,
we will increase count of this cell. Table 2.1 and Table 2.2 define us how it works.
![](img/table1.png) 

![](img/table2.png)

**Extracted keywords** After candidate keywords are scored, the top T scoring candidates are selected as keywords for the document. We compute T as one-third the number of words
in the graph.
</details>


### Word embedding


<details>
  <summary> :point_right: Bag of words</summary>

The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR).
In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding
grammar and even word order but keeping multiplicity.The bag-of-words model is commonly used in methods of document 
classification where the (frequency of) occurrence of each word is used as a feature for training a classifier.
An early reference to "bag of words" in a linguistic context can be found in Zellig Harris's 1954 article on Distributional Structure.[source: wiki]
Bag of words process has two steps.

**Tokenization:**  First we tokenize whole sentences.
**Vectors Creation:**
	after doing the first step, we will make a vector for each sentence.
	Dividing sentences into words and creating a list with all unique words and also in alphabetical order. \\
Let's take an example with three-sentence,
```
 John likes to watch movies.
 Mary likes movies too.
 Mary also likes to watch football games.
```
Now after tokenization, our word list look like,

```
[also,football,games,john,likes,mary,movies,to,too,watch]
```

After that we need to compute sparse matrix. This sparse matrix represent the bag of words.
In sparse matrix every cell represent, **occurence of j'th word in i'th sentence**. 

![](img/sum1.jpg)
</details>


<details>
  <summary> :point_right: TF-IDF</summary>

TF-IDF value will be increased based on frequency of the word in a document. Like Bag of Words in this technique also we can not get any semantic meaning for words. \\
But this technique is mostly used for document classification and also 
successfully used by search engines like Google, as a ranking factor for content.

Suppose we have three sentences.
```
A. This pasta is very tasty and affordable.
B. This pasta is not tasty and is affordable.
C. This pasta is very very delicious.
```
Let's consider each sentence as a document. Here also our first task is tokenization (dividing sentences into words or tokens) and then taking unique words.

![](img/table3.png)

</details>


<details>
  <summary> :point_right: Word2vec</summary>

The Word2Vec model is used for learning vector representations of words called “word embeddings”. In the previous methods, there was no semantic meaning from words of corpus. 
Word2vec model takes input as a large size of the corpus and produces output to vector space. This vector space size may be in hundred of dimensionality. Each word vector will be placed on this vector space.
word2vec uses 2 types of methods. There are

1. Skip-gram
2. CBOW (Continuous Bag of Words)


![](img/sum2.png)

Here one more thing we have to discuss that is window size. window size basically 1-gram,bi-gram,tri-gream,N-gram model.

</details>

<details>
  <summary> :point_right: Skip gram (word2vec p1)</summary>
In this method , take the center word from the window size words as an input and context words (neighbour words) as outputs. Word2vec models predict the context words of a center word using skip-gram method. Skip-gram works well with a small dataset and identifies rare words really well.
In 2013 Mikolov first time told about this method [3].

![](img/sum3.png)

According to Fig,

1. Both the input word **wi** and the output word **wj** are one-hot encoded into binary vectors **x** and **y** of size V.
2. First, the multiplication of the binary vector x and the word embedding matrix **W** of size 
   **V×N** gives us the embedding vector of the input word **wi** the i-th row of the matrix 
   **W**.
3. This newly discovered embedding vector of dimension N forms the hidden layer.
4. The multiplication of the hidden layer and the word context matrix 
   **W'** of size **N×V** produces the output one-hot encoded vector **y**.
5. The output context matrix **W'** encodes the meanings of words as context, 
   different from the embedding matrix W. NOTE: Despite the name, **W'** is independent of 
   **W**, not a transpose or inverse or whatsoever.

</details>

<details>
  <summary> :point_right: Cbow (word2vec p2)</summary>
The Continuous Bag-of-Words (CBOW) is another similar model for learning word vectors. It predicts the target word (i.e. “swing”) 
from source context words (i.e., “sentence should the sword”).
Because there are multiple contextual words, we average their corresponding word vectors, constructed 
by the multiplication of the input vector and the matrix \emph{W}. Because the averaging stage smoothes over a lot
of the distributional information, some people believe the CBOW model is better for small dataset.

![](img/sum4.png)

</details>


<details>
  <summary> :point_right: Loss function (word2vec p3)</summary>
Both the skip-gram model and the CBOW model should be trained to minimize a well-designed loss/objective function. 
There are several loss functions we can incorporate to train these language models. like,

1. Full Softmax
2. Hierarchical Softmax
3. Cross Entropy
4. Noise Contrastive Estimation (NCE)
5. Negative Sampling (NEG)
</details>


<details>
  <summary> :point_right: Difference between CBOW and skip-gram model(word2vec p4)</summary>

</details>



<details>
  <summary> :point_right: Glove</summary>
GloVe stands for global vectors for word representation. It is an unsupervised learning algorithm 
developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus. 
The resulting embeddings show interesting linear substructures of the word in vector space.
GloVe is essentially a log-bilinear model with a weighted least-squares objective. The main intuition underlying the model 
is the simple observation that ratios of word-word co-occurrence probabilities have the potential for encoding some form of meaning. In Glove we also create sparse matrix.
It was directly involved in the probability matrix.

For example, consider the co-occurrence probabilities for target words ice and steam with various probe words from the vocabulary.
Here are some actual probabilities from a 6 billion word corpus:

![](img/sum7.png)

</details>


<details>
  <summary> :point_right: Fasttext</summary>

Compared to other word embedding methods, FastText (Mikolov et al., 2016) is a new approach which can generate competitive results.[4] [5] This technique is much popular.

![](img/sum6.jpg)

According to the main paper, the model is a simple neural network with only one layer.
The bag-of-words representation of the text is first fed into a lookup layer, 
where the embeddings are fetched for every single word. Then, those word embeddings are 
averaged, so as to obtain a single averaged embedding for the whole text. At the hidden layer
we end up with **(n\_words x dim)** number of parameters, where dim is the size of the 
embeddings and n\_words is the vocabulary size. After the averaging, we only have a single 
vector which is then fed to a linear classifier: we apply the softmax over a linear transformation of the output of the input layer. The linear transformation is a matrix with {\bf (dim x n\_output)}, where {\bf n\_output} is the number output classes. In the original paper, the final log-likelihood is:

![](https://latex.codecogs.com/svg.latex?-\frac{1}{N}\sum^N_{i=n}y_n\log(f(BAx_n)))


where,
1. ![](https://latex.codecogs.com/svg.latex?x_n) is the original one-hot-encoded representation of a word (or n-gram feature),
2. A is the look-up matrix that retrieves the word embedding,
3. B is the linear output transformation,
4. f is the softmax function

</details>

### Sentence Embedding

<details>
  <summary> :point_right: Introduction</summary>

Instead of dealing with words, we can directly work with the sentence. how? 
Sentence embedding is the answer. Sentence embedding also makes semantic information 
in vector.

Suppose we have two sentences, 'I don’t like crowded places',
'However, I like one of the world’s busiest cities, New York'. 
Here, How can we make the machine draw the inference between ‘crowded places’
and ‘busy cities’?

Word embedding can't do that. But we can do it by sentence embedding.
There are lots of techniques in recent years. We will talk about a few of them.

1. Doc2Vec
2. SentenceBERT
3. InferSent
4. Universal Sentence Encoder

</details>


<details>
  <summary> :point_right: Doc2vec</summary>

An extension of Word2Vec, the Doc2Vec embedding is one of the most popular techniques out there.
Introduced in 2014, it is an unsupervised algorithm and adds on to the Word2Vec model by introducing another ‘paragraph vector’ by T mikolov [6]. In 2015 it was improved by Andrew [7]. Also, there are 2 ways to add the paragraph vector to the model. 
one is PVDM and another PVDOBW.
</details>


<details>
  <summary> :point_right: PVDM(Distributed Memory version of Paragraph Vector)</summary>

We assign a paragraph vector sentence while sharing word vectors among all sentences. Then we either average or concatenate the (paragraph vector and words vector) to get the final sentence representation.
If you notice, it is an extension of the Continuous Bag-of-Word type of Word2Vec where we predict the next word given a set of words. It is just that in PVDM, we predict the next sentence given a set of sentences.
![](img/sum8.png)
</details>

<details>
  <summary> :point_right: PVDOBW( Distributed Bag of Words version of Paragraph Vector)</summary>

PVDOBW is the extention of Skip-gram model in Word2vec. Here, we just sample random words from the sentence and make the model predict which sentence it came from(a classification task).

![](img/sum9.png)

PVDM is more than enough for most tasks.But we can use both parallelly.
</details>

<details>
  <summary> :point_right: SentenceBERT</summary>

Currently, the leader among the pack, SentenceBERT was introduced by Reimers in 2018 [8] 
and immediately took the pole position for Sentence Embeddings. At the heart of this 
BERT-based model, there are 4 key concepts:

1. Attention
2. Transformers
3. BERT
4. Siamese Network


we will discuss BERT,transformers,Attention model later.

Sentence-BERT uses a Siamese network like architecture to provide 2 
sentences as an input. These 2 sentences are then passed to BERT models and a pooling layer 
to generate their embeddings. Then use the embeddings for the pair of sentences as inputs to 
calculate the cosine similarity.

![](img/sum10.png)

Suppose we have some sentences.

1. I ate dinner. 
2. We had a three-course meal.
3. Brad came to dinner with us.
4. He loves fish tacos.",
5. In the end, we all felt like we ate too much.
6. We all agreed; it was a magnificent evening.

Now, if we want to find the similarity between these sentences with 
**"I had pizza and pasta"**. SentenceBert will give us.

![](img/table4.png)

</details>

<details>
  <summary> :point_right: InferSent</summary>

InferSent is a sentence embeddings method that provides semantic sentence representations 
invented by A Conneau in 2017 [9]. It is trained on natural language inference data and 
generalizes well to many different tasks.

In this model,they examine standard recurrent models such as LSTMs and
GRUs, for which we investigate mean and maxpooling over the hidden representations; 

The architecture consists of 2 parts:

1. One is the sentence encoder that takes word vectors and encodes sentences into vectors
2. Another, NLI classifier that takes the encoded vectors in and outputs a class among entailment, contradiction and neutral.

![](img/sum11.jpg)

In the author paper, four types of sentence encoder were discussed.

</details>
<details>
  <summary> :point_right: InferSent part1: Sentence Encoder(LSTM and GRU)</summary>

First, and simplest, encoders apply recurrent neural networks using either LSTM or GRU modules, as in sequence to sequence encoders. For a sequence of T words (w1, . . . , wT ), the network computes a set of T hidden representations h1, . . . , hT , with ht = -> LSTM(w1, . . . , wT ) (or using GRU units instead). A sentence is represented by the last hidden vector, hT.

</details>

<details>
  <summary> :point_right: InferSent part2: Sentence Encoder (BiLSTM with max/mean pooling)</summary>

It’s a bi-directional LSTM network which computes n-vectors for n-words and each vector is a 
concatenation of output from a forward LSTM and a backward LSTM that read the sentence in 
opposite direction. Then a max/mean pool is applied to each of the concatenated vectors to 
form the fixed-length final vector.
![](img/sum12.png)

</details>

<details>
  <summary> :point_right: InferSent part3: Sentence Encoder (Self-attentive network)</summary>

The self-attentive sentence encoder uses an attention mechanism over the hidden states of a BiLSTM to generate a representation u of an input sentence. The attention mechanism is defined as,

![](https://latex.codecogs.com/svg.latex?\bar%20h_i=%20\tanh(Wh_i+b_w))

![](https://latex.codecogs.com/svg.latex?\alpha_i=%20\frac{e^{\bar%20h^T_i%20u_w}}{\sum_i%20e^{\bar%20h^T_i%20u_w}})

![](https://latex.codecogs.com/svg.latex?u=%20\sum_t%20\alpha_i%20h_i)

Where ![](https://latex.codecogs.com/svg.latex?{h_1,.....,h_T}) are the output hidden 
vectors of BiLSTM. These are fed to an affine transformation ![](https://latex.codecogs.com/svg.latex?(W,b_w)) 
which outputs a set of keys ![](https://latex.codecogs.com/svg.latex?(\bar%20h_1,....,\bar%20h_T)).
The ![](https://latex.codecogs.com/svg.latex?\{\alpha_i\}) represent the score of similarity 
between the keys and learned context query vector ![](https://latex.codecogs.com/svg.latex?u_w). 
These weights are used to produce the final representation u, 
Which is a linear combination of the hidden vectors.

![](img/sum13.png)

</details>

<details>
  <summary> :point_right: InferSent part4: Sentence Encoder( Hierarchical ConvNet)</summary>

One of the currently best performing models on classification tasks is a convolutional architecture termed AdaSent, which concatenates different representations of the sentences at a different level of abstractions. The hierarchical convolutional network introduced in the paper is inspired by that and comprise 4 convolutional layers. At every layer, max-pooling of the feature maps is done to obtain representation. The final sentence embedding is represented by a concatenation of the 4 max-pooled representations.

![](img/sum14.jpg)
</details>

<details>
  <summary> :point_right: InferSent part5: Natural Language Inference Classifier(NLI)</summary>

This section discusses the inference classifier network which takes these sentence embeddings and predicts the output label.

![](img/sum15.png)
After the sentence vectors are fed as input to this model, 3 matching methods are applied to extract relations between the text, u and hypothesis, v –

1. concatenation of the two representations (u, v)
2. element-wise product (u * v)
3. and, absolute element-wise difference (|u - v|)

The resulting vector captures information from both the text, u and the hypothesis, v, and 
is fed into a 3-class classifier( entailment, contradiction and neutral) consisting of 
multiple fully connected layers followed by a softmax layer.
</details>

<details>
  <summary> :point_right: Universal Sentence Encoder</summary>

One of the most well-performing sentence embedding techniques right now is the Universal 
Sentence Encoder. And it should come as no surprise from anybody that it has been proposed by Google. The key feature here is 
that we can use it for Multi-task learning.
This means that the sentence embeddings we generate can be used for multiple tasks 
like sentiment analysis, text classification,
sentence similarity, etc, and the results of these asks are then fed back to the model
to get even better sentence vectors that before.

This technique was invented by google and authored Cer and Daniel in 2018 [10]. 

![](img/sum16.png)

In this model, they use two types of encoder. We can use any of them.

1. Transformer
2.  Deep Averaging Network (DAN)

Both of these models are capable of taking a word or a sentence as input and generating
embeddings for the same. The following is the basic flow:

1. Tokenize the sentences after converting them to lowercase
2. Depending on the type of encoder, the sentence gets converted to a 512-dimensional vector

    1. If we use the transformer, it is similar to the encoder module of the transformer architecture and uses the self-attention mechanism.
    2. The DAN option computes the unigram and bigram embeddings first and then averages them to get a single embedding. This is then passed to a deep neural network to get a final sentence embedding of 512 dimensions.
    
3. These sentence embeddings are then used for various unsupervised and supervised tasks like Skipthoughts, NLI, etc. The trained model is then again reused to generate a new 512 dimension sentence embedding.

![](img/sum17.png)

Similarity of two vector:

![](https://latex.codecogs.com/svg.latex?sim(u,v)%20=%20(1-%20\frac{\arccos%20\frac{u.v}{||u||\hspace{2pt}||v||}}{\pi}))
</details>

### End of Pre-Requirements study
<details>
  <summary> :point_right: Ok i learn about these topics but can't understand how they use these?</summary>

Yeah. It's really tough to understand, how they used these topics in Text summarization models. In my opinion, Maximum models used these topics as a sub-portion. So we need to read a paper sequentially and learn which portion we will do with these topics. 
</details>

## Types of Text Summarization


<details>
<summary> :point_right: Types of Text Summarization</summary>

We can divide all summarization model into three categories.
1.  Extractive Summarization 
2.  Abstractive Summarization
3.  Combined Approach

</details>


<details>
<summary> :point_right: Extractive Summarization</summary>
Extractive summarization means identifying important sections of the text and generating them verbatim producing a subset of the sentences from the original text.
Here, content is extracted from the original data, but the extracted 
content is not modified in any way.

So many Extractive summarization techniques were invented after 2004. Like,

1.  LexRank (Erkan and Radev, 2004) 
2.  TextRank (Mihalcea and Tarau, 2004) 
3.  RegSum system (Hong and Nenkova, 2014)
4.  TextTeaser(Jolo Balbin,2013)
5.  Summarization by LSA(Ferda and Ilyas,2011)
6.  Quasi-Synchronous Grammar base(Woodsend and Yansong Feng,2010)
7.  Neural network Base extractive
8.  An attention-based encoder-decoder(Cheng and Lapata (2016))
</details>


<details>
<summary> :point_right: Abstractive Summarization</summary>

Abstractive summarization is much advanced from extractive and also much good accuracy. It is more fluent and natural summaries.The abstractive model takes advantage of deep learning. In this time so many new models coming with high accuracy.

1.  Encoder-decoder Model
2.  CNN and RNN model
3.  LSTM model
4.  Traning model like MLE
5.  Seq to seq model
6.  Summarization using sentence Compression
7.  Abstractive Text Summarization using Reinforcement Learning
8.  Summarization using GRU or BGRU
9.  Attention base model
10.  Pointer base model
11.  Transformer base model like bart

Many Models was invented in recent years using these techniques.

</details>

<details>
<summary> :point_right: Extractive and Abstractive Combined Approach</summary>

As our data like news, articles, blogs, papers, etc increasing day by day. Also out content is much bigger than the old one. The abstractive models take lots of time in computing. The extractive models are much faster. That's why nowadays researchers make a combined way.


1.  Focusing importance sentences and use abstractive model (Abigail See, Liu, Christopher,2017)
2.  Query Focused Abstractive Summarization (Tal Baumel, Matan Eyal, Michael Elhadad,2018)
3.  Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting (chen and bishal,2018)
</details>

## Text Summarization models

<details>
<summary> :point_right: Graph Base Extractive Summarization</summary>

Graph Base Extractive Summarization is the oldest technique.
In 2004, Mihalcea and Rada discussed graph base model textrank. 
This technique was based on PageRank.

![](img/page_rank.png)

In TextRank, We just make relations with sentences by Some processing. we link 
sentence to sentence.
Every Sentence is treated like a node. We can make the graph with word unit.

**graph clustering** is the major process here.
An overall Graph bases model structure:

![](img/sum18.jpg)

</details>


<details>
<summary> :point_right: Feature Base Extractive Summarization</summary>

In Feature base Model, Sentence are ranked by some features.


1. Position of the sentence in input document
2. Presence of the verb in the sentence
3. Length of the sentence
4. Term frequency(TF)
5. Named entity tag NE
6. Font style

Basically these features are used in feature based model was developed by 
Jagadeesh in 2005 [12]. Feature Base Extraction model is better than any other 
extraction model. Cause, we can add deep learning feature or another good feature for 
ranking sentences.
![](img/sum19.jpg)

</details>


<details>
<summary> :point_right: Topic Base Extractive Summarization</summary>

The topic base model calculates the topic of the document and evaluate each sentences by what kinds of topics are 
included (the "main" topic is highly evaluated when scoring the sentence).
**Latent Semantic Analysis (LSA) is usually used to detect the topic.** It's based on SVD (Singular Value Decomposition). 
A good representation about text summarization with LSA was given by Ozsoy(2011) [13].
In this paper they used three step.

**Step 1(Input Matrix Creation):** \
The way in which an input matrix is created is very important for summarization, since it affects the
resultingmatrices calculated with SVD. As already mentioned, SVD is a complex algorithm and its complexity 
increases withthe size of input matrix, which degrades the performance. In order to reduce the matrix size, 
rows of the matrix, i.e. thewords, can be reduced by approaches like removing stop words, using the roots of words 
only, using phrases instead ofwords and so on. Also, cell values of matrix can change the results of SVD. There are 
different approaches to filling outthe cell values. These approaches are as follows:

1. **Frequency of word:** the cell is filled in with the frequency of the word in the sentence.
2. **Binary representation:** the cell is filled in with 0/1 depending on the existence of a word in the sentence.
3. **Tf-idf (Term Frequency-Inverse Document Frequency):** the cell is filled in with the tf-idf value of the word.A higher tf-idf value means that the word is more frequent in the sentence but less frequent in the wholedocument. A higher value also indicates that the word is much more representative for that sentence thanothers.
4. **Log entropy:** the cell is filled in with the log-entropy value of the word, which gives information on how infor-mative the word is in the sentence.
5. **Root type:** the cell is filled in with the frequency of the word if its root type is a noun, otherwise the cell valueis set to 0.
6. **Modified Tf-idf:** this approach is proposed in Ozsoy et al. [3], in order to eliminate noise from the input matrix.The cell values are set to tf-idf scores first, and then the words that have scores less than or equal to the averageof the row are set to 0.


**Step 2(Singular Value Decomposition (SVD)):** \
SVD is an algebraic method that can model relationships among words/phrases and sen-tences.
In this method, the given input matrix Ais decomposed into three new matrices as follows:

![](https://latex.codecogs.com/svg.latex?A%20=%20U%20\sum%20V^T)

where A is the input matrix (m * n); U is words ×extracted concepts (m * n); 
Prepresents scaling values, diagonal descending matrix (n * n); and V is sentences ×extracted concepts (n * n).

**Step 3(Sentence Selection):** using the results of SVD, different algorithms are used to select important sentences. 
Some of them are discussed in below paper:

1.  Gong and Liu (2001) [14]
2.  Steinberger, Josef, and Karel Jezek (2004) [15]
3.  Murray et al. (2005) [16]
4.  Cross Method by Steinberger \& Jezek(2008) [17]
5.  topic method

</details>


<details>
<summary> :point_right: BSTM base Extractive Summarization</summary>

BSTm means Bayesian Sentence-based Topic Models. In topic modeling of text documents, 
the goal is to infer the words related to a certain topic and the topics discussed in a certain document, 
based on the prior analysis of a corpus of documents. It is possible with the help of Bayesian inference that calculates 
the probability of an event based on a combination of common sense assumptions and the outcomes of previous related events.
First, They took a single Document $d$. Then We calculate parameters by,

![](https://latex.codecogs.com/svg.latex?p(W^n_1|%20\theta_d)%20=%20\Pi^n_{i=1}%20p(W_i|\theta_d))

where ![](https://latex.codecogs.com/svg.latex?\theta_d) denotes the model parameter for document d, ![](https://latex.codecogs.com/svg.latex?W^n_1)
denotes the sequence of words, i.e. the content of the document. W is the
vocabulary.
Now we need to calculate  ![](https://latex.codecogs.com/svg.latex?p(W_i|\theta_d)). We will Calculate it based on different topic.

![](https://latex.codecogs.com/svg.latex?p(W_i|\theta_d)%20=%20\sum_{T_i%20\in%20\tau}%20p(W_i|T_i)p(T_i|\theta_d))

where T is the set of topics. Here, we assume
that given a topic, generating words is independent
from the document, i.e.

![](https://latex.codecogs.com/svg.latex?p(W_i|T_i,\theta_d)%20=%20p(W_i|T_i))

Instead of freely choosing topic unigram models, we further assume that topic unigram models
are mixtures of some existing base unigram models, i.e.

![](https://latex.codecogs.com/svg.latex?p(W_i|T_i)=\sum_{s\in%20S}%20p(W_i|S_i%20=%20s)p(S_i=s|T_i))

where S is the set of base unigram models. Here,
we use sentence language models as the base models. One benefit of this assumption is that each
topic is represented by meaningful sentences, instead of directly by keywords.
</details>


<details>
<summary> :point_right: Grammar Base Extractive summarization</summary>

The grammar base model parses the text and constructs a grammatical structure, then select/reorder substructures.
In 2010, Woodsend and Kristian told in their paper how grammatical structure helps us to build summarization [18]. 
They build every sentences into a grammatical structure and reduces length. 

![](img/sum20.png)

In this figure, you can see how they reduces 'Saudi dissident Osama bin Laden' to bin laden.
Another example "in the disputed territory of East Timor" to "in East Timor". 
Last one is "the extradition of Kurdish leader Ocalan" to "Ocalan's extradition".
</details>


<details>
<summary> :point_right: Neural network base Extractive Summarization</summary>

In 2016, Nallapati and Ramesh proposed an extractive model by Recurrent Neural Network(RNN) and named it **"SummaRuNNer"** [19]

![](img/sum21.png)

SummerRunner basically a two-layer RNN based sequence classifier. The bottom layer operates at word level within each sentence,
while the top layer runs over sentences. Double-pointed arrows indicate a bi-directional RNN. The top layer with 1’s and 0’s is the
sigmoid activation based classification layer that decides whether
or not each sentence belongs to the summary. The decision at
each sentence depends on the content richness of the sentence, its
salience with respect to the document, its novelty with respect to
the accumulated summary representation and other positional features

</details>


<details>
<summary> :point_right: Encoder Decoder Abstractive Model with RNN</summary>

You can see the Dictionary Section, What is RNN and Encoder Decoder Model. We will now discuss Encoder Decoder Model 
with RNN. Cho and Kyunghyun in 2014 [22] showed us, how we can encode and decode sequential context by RNN.
We can learn whole input by RNN and. After encode we will get a hidden layer. From this Hidden Layer we will 
decode the Output. RNN is good for unsupervised machine Learning. In 2009 Wang Proposed it [23].

![](img/sum32.png)

RNN did a great job in encoder decoder model. Lots of encoder-decoder model with RNN was build in last few years.
we will see them later.

</details>


<details>
<summary> :point_right: Abstractive Summarization with Attentive Recurrent Neural Networks</summary>

This model is another modified version of Encoder decoder model with rnn. Difference between two of them is in Encoder machanism.
This model was proposed by Alexander in 2016 [24]. In this model they used Conditional RNN.
Let x denote the input sentence consisting of a
sequence of M words x = ![](https://latex.codecogs.com/svg.latex?x_1,...,x_M)], where each word ![](https://latex.codecogs.com/svg.latex?x_i)
is part of vocabulary V, of size
|V| = V . Our task is to generate a target sequence
y = ![](https://latex.codecogs.com/svg.latex?y_1,...,y_N) ], of N words, where N < M,
such that the meaning of x is preserved: y = argmax(y) ![](https://latex.codecogs.com/svg.latex?P(y|x)), where y is a random variable denoting a sequence of N words.
Typically the conditional probability is modeled by a parametric function with parameters ![](https://latex.codecogs.com/svg.latex?\theta):
![](https://latex.codecogs.com/svg.latex?P(y|x)%20=%20P(y|x;%20\theta)). Training involves finding the
![](https://latex.codecogs.com/svg.latex?\theta) which maximizes the conditional probability of
sentence-summary pairs in the training corpus.

![](https://latex.codecogs.com/svg.latex?P(y|x;\theta)%20=%20\Pi^N_{t=1}p(y_t|\{y_1,...,y_{t-1}\},x;\theta))

{\bf Recurrent Decoder: }
The above conditional is modeled using an RNN:

![](https://latex.codecogs.com/svg.latex?P(y_t|\{y_1,...,y_{t-1}\},x;\theta)%20=%20P_t=g\theta_1(h_t,c_t))

where ![](https://latex.codecogs.com/svg.latex?h_t) is the hidden state of the RNN:

![](https://latex.codecogs.com/svg.latex?h_t=g\theta_1(y_{t-1},h_{t-1},c_t))

Final form of RNN Model is,

![](https://latex.codecogs.com/svg.latex?h_t%20=\sigma(W_1y_{t-1}+W_2h_{t-1}+W_3c_t))

![](https://latex.codecogs.com/svg.latex?P_t=\rho(W_4h_t+W_5c_t))

where ![](https://latex.codecogs.com/svg.latex?\sigma) is the sigmoid function and ![](https://latex.codecogs.com/svg.latex?\rho) is the softmax function. So this is not like LSTM Long process.

**Attentive Encoder:** \
 for an input sentence x we denote by ![](https://latex.codecogs.com/svg.latex?x_i) the d
dimensional learnable embedding of the i-th word ( ![](https://latex.codecogs.com/svg.latex?x_i%20\epsilon%20R^d)
). In addition the position i of the word
![](https://latex.codecogs.com/svg.latex?x_i)
is also associated with a learnable embedding ![](https://latex.codecogs.com/svg.latex?l_i)
of size d ![](https://latex.codecogs.com/svg.latex?l_i%20\epsilon%20R^d)
). Then the full embedding for
i-th word in x is given by ![](https://latex.codecogs.com/svg.latex?a_i%20=%20x_i%20+%20l_i)
. Let us denote by ![](https://latex.codecogs.com/svg.latex?B^k%20\epsilon) ![](https://latex.codecogs.com/svg.latex?R^(q*d)))
a learnable weight matrix
which is used to convolve over the full embeddings
of consecutive words. Let there be d such matrices

![](https://latex.codecogs.com/svg.latex?k%20\epsilon%20{1,...,d}) The output of convolution is given
by:

![](https://latex.codecogs.com/svg.latex?z_{ik}%20=\sum^{q/2}_{h=-q/2}a_{i+h}.b^k_{q/2+h})

where ![](https://latex.codecogs.com/svg.latex?b^k_j) is the j-th column of the matrix ![](https://latex.codecogs.com/svg.latex?B^k). Thus
the d dimensional aggregate embedding vector ![](https://latex.codecogs.com/svg.latex?z_i) is
defined as ![](https://latex.codecogs.com/svg.latex?z_i) = [![](https://latex.codecogs.com/svg.latex?z_i1), . . . , ![](https://latex.codecogs.com/svg.latex?z_id)]. Note that each word
![](https://latex.codecogs.com/svg.latex?x_i) in the input sequence is associated with one aggregate embedding vector ![](https://latex.codecogs.com/svg.latex?z_i).
The vectors ![](https://latex.codecogs.com/svg.latex?z_i) can be
seen as a representation of the word which captures
the position in which it occurs in the sentence and
also the context in which it appears in the sentence.
In our experiments the width q of the convolution
matrix ![](https://latex.codecogs.com/svg.latex?B^k) was set to 5. To account for words at the
boundaries of x we first pad the sequence on both
sides with dummy words before computing the aggregate vectors ![](https://latex.codecogs.com/svg.latex?z_i) ’s.
Given these aggregate vectors of words, we compute the context vector ![](https://latex.codecogs.com/svg.latex?c_t) (the encoder output) as:

![](https://latex.codecogs.com/svg.latex?c_t=\sum^M_{j=1}\alpha_{j,t-1}x_j)

![](https://latex.codecogs.com/svg.latex?\alpha_{j,t-1}) are computed as,

![](https://latex.codecogs.com/svg.latex?\alpha_{j,t-1}=\frac{\exp(z_j.h_{t-1})}{\sum^M_{i=1}\exp(z_i.h_{t-1})})

</details>


<details>
<summary> :point_right: Neural Attention Model</summary>

Neural Attention model is the combined of **attention model and BRNN**. This model was indtoduced by Alexander at 2017 [28].

![](img/sum39.png)

In this model, We use attention based encoder for finding important sentences.

![](https://latex.codecogs.com/svg.latex?enc(x,y_c)%20=%20p^T\bar{x})

![](https://latex.codecogs.com/svg.latex?p%20\propto%20\exp(\tilde%20xP\tilde%20y^\prime_c))

![](https://latex.codecogs.com/svg.latex?\tilde%20x=[Fx_1,...,Fx_M])

![](https://latex.codecogs.com/svg.latex?\tilde{y}^\prime_c%20=[Gy_{i-C+1},...,Gy_i])

![](https://latex.codecogs.com/svg.latex?\forall_i%20\bar{x}_i%20=\sum^{i+Q}_{q=i-Q}\tilde{x}_i/Q)

![](img/sum40.jpg)

Here,

1. x is the input sequence, ![](https://latex.codecogs.com/svg.latex?y_c) is the output sequence of fixed window size C. 
   enc is the encoder.
2. P is a new weight matrix parameter mapping between the context embedding
and input embedding, and Q is a smoothing window(some fixed size of matrix). 
3. F is a word embedding matrix


</details>


<details>
<summary> :point_right: Sequence-to-sequence RNNs and Beyond</summary>

This model was introduced by Nallapati at 2016. In this model they focused on all type of seq to seq model problem. example:

1. In sequence to sequence model we need to train large vocabulary. But here we will train a subset of vocabulary.\\
Large Vocab Trick, instead of softmax over the entire vocabulary, softmax over the vocabulary from the source document as well as the top frequent words in the vocabulary
2. In other model, For finding important keyword , we need to train vocabulary (embedding layer). But here, they focused on TF-IDF , Pos tag, Position, NER etc. what is the benefit? of course much faster than neural network.
![](img/sum44.png)
3. In seq to seq model, we don't handle unknown or rare word. But in this model we handle it by pointer or copy from original text.
4. For long term dependency, they used here attention mechanism.
</details>


<details>
<summary> :point_right: Summarization with Pointer-Generator Networks</summary>

This model is similar to Seq-to-Seq model with RNNs and beyond. This model was proposed by see at 2017 [32]. This model also reduced Seq to Seq model problems.

![](img/sum43.png)

In this model they Pointed two problem in seq to seq attention model.

1. **Unknown and rare word problem:** When we found unknown word but important,
   we marked it as 'UNK'. For that reason we didn't get better result in seq to seq model. 
   In this model they used pointer generator for overcome this problem. How?
In pointer generator network we will directly pass unknown word. It allows both copying words via pointing, and generating words from a fixed vocabulary.

2. **repeating problem:** Repetition is a common problem for sequence-to-sequence models.
   example: "germany beat germany beat germany beat.." may be produced in seq to seq model. 
They used a coverage vector to overcome this problem. Coverage vector ![](https://latex.codecogs.com/svg.latex?C^t) basically sum of attention distributions over all previous decoder timesteps. For every present timestamp, they used it to change output state.
   
</details>


<details>
<summary> :point_right: LSTM and CNN  base abstractive model</summary>

In 2019, Song proposed a Phrase encoder [29]. This encoder is based on Lstm-CNN. 
This encoder takes Phrases instead of words.The phrases are composed of several
words to express the overall meaning, and it is a key concept in the field of phrase filling
and phrase extraction.Phrases can be roughly divided into three major types: subject phrases, 
relational phrases and object phrases.  for example in "Mary wants to go home", the subject 
phrase is "Mary", the relational phrase is "wants to go" and the object phrase is "home".

what is the difference between others model and this model? This model is the extended 
of Lstm based model. Here in input sequence they didn't give words. In this model, they used 
phrase encoder to encode words to phrase.

So this full model we will divide into three parts.

![](img/sum41.png)

**Phrase encoder:** \
We opted for a convolutional neural network model
for representing phrases for two reasons. Firstly, single layer CNNs can be trained 
effectively (without any long-term dependencies in the model) and secondly, CNNs have been
successfully used for sentence-level classification tasks such as sentiment analysis. Let d
denote the dimension of word embedding, and s a document phrase consisting of a
sequence of n words ( ![](https://latex.codecogs.com/svg.latex?w_1) ,…,![](https://latex.codecogs.com/svg.latex?w_n)) which can be represented by a dense column matrix  W(n * d)
. We apply a temporal narrow convolution between Wand a kernel K(c * d) of
width c as follows:

![](https://latex.codecogs.com/svg.latex?f^i_j=\tanh(W_{j:j+c-1}\otimes%20K+b))

Where ![](https://latex.codecogs.com/svg.latex?\otimes) is the Hadamard Product followed by a sum over all elements.![](https://latex.codecogs.com/svg.latex?f^i_j) denotes the j-th
element of the i-th feature map ![](https://latex.codecogs.com/svg.latex?f^i)
and b is the bias. We perform max pooling over time to obtain
a single feature (the i-th feature) representing the phrase under the kernel K with width c:

![](https://latex.codecogs.com/svg.latex?s_{i,k}=max_j%20f^i_j)

**Recurrent document encoder:** This encoder type we already know. We can use here
Bidirectional GRU or Bidirectional LSTM. Any type of RNN we can use here. We already discussed it.

**Decoder:** In decoder Section, based on the sequence-to-sequence model, this model divides decoder part into two different modes: generate mode and copy mode. In this mode, we calculate output conditional probability. We already know how seq to seq decoder works.

</details>

<details>
<summary> :point_right: Selective Encoding for Abstractive Sentence Summarization</summary>

This model also focused on input sequence. This model was introduced by Zhou at 2017 [33]. 
This model was extended seq to seq model by a selective gate.
we can divide whole model with three part.

![](img/sum45.png)


**1. Sentence encoder:** Its nothing but a bi-GRU network for reading input sequence like 
others model.

**2.  Selective Mechanism:** Here comes the extended part of this model. For selective mechanism they used a selective gate here. the selective gate network in their model takes two vector
inputs, the sentence word vector ![](https://latex.codecogs.com/svg.latex?h_i) and the sentence representation vector s. The sentence word
vector ![](https://latex.codecogs.com/svg.latex?h_i) is the output of the BiGRU encoder and represents the meaning and context information of
word ![](https://latex.codecogs.com/svg.latex?x_i). The sentence vector **s** is used to represent the meaning of the sentence.

we concatenate the last forward hidden state ![](https://latex.codecogs.com/svg.latex?\overrightarrow{h}_n) and backward hidden state ![](https://latex.codecogs.com/svg.latex?\overleftarrow{h}_1)
as the sentence representation s:

![](https://latex.codecogs.com/svg.latex?s%20=%20\begin{bmatrix}%20\overleftarrow{h}_1\\%20\overrightarrow{h}_n%20\end{bmatrix})

For each time step i, the selective gate takes the
sentence representation s and BiGRU hidden hi as
inputs to compute the gate vector ![](https://latex.codecogs.com/svg.latex?sGate_i)

![](https://latex.codecogs.com/svg.latex?sGate_i=\sigma(W_sh_i+U_ss+b))

![](https://latex.codecogs.com/svg.latex?h^\prime_i=h_i\odot%20sGate_i)

where ![](https://latex.codecogs.com/svg.latex?W_s) and ![](https://latex.codecogs.com/svg.latex?U_s) are weight matrices, b is the
bias vector, ![](https://latex.codecogs.com/svg.latex?\sigma) denotes sigmoid activation function, and ![](https://latex.codecogs.com/svg.latex?\odot) 
is element-wise multiplication.

**3.  Summary Decoder:** This decoder also a bi-GRU. They used the same process of seq-to-seq model or other Decoder.

</details>

<details>
<summary> :point_right: A Deep Reinforced Model for Abstractive Summarization</summary>

This model also reduced SeqtoSeq model problems. This model was introduced by Paulus at 2017 [34].
we can divide this model into 5 parts.

![](img/sum46.png)


1. **INTRA-TEMPORAL ATTENTION ON INPUT SEQUENCE:** This model used Nallapati's [31] model. in that model have shown that such an intra-temporal attention can reduce the amount of repetitions when attending over long documents. In this model they used bilinear function for attention. After that they normalize the sequence.
2. **INTRA-DECODER ATTENTION:** A closely-related intra-RNN attention function was used here. For reducing repating pharse they used here also attention. If a sentence or word already written, they did a less attention to those words.
3. **TOKEN GENERATION AND POINTER:** To generate a token, this decoder used either a token-generation softmax layer or a pointer mechanism to copy rare or unseen from the input sequence. they used a switch function that decides at each decoding step whether to use the token generation or the pointer.
4. **SHARING DECODER WEIGHTS:** In addition to using the same embedding matrix ![](https://latex.codecogs.com/svg.latex?W_{emb}) for the encoder and the decoder sequences, we introduce some weight-sharing between this embedding matrix and the ![](https://latex.codecogs.com/svg.latex?W_{out}) matrix of the tokengeneration layer.
5. **REPETITION AVOIDANCE AT TEST TIME:** Another way to avoid repetitions comes from observation that in both the CNN/Daily Mail and
NYT datasets, ground-truth summaries almost never contain the same trigram twice. Based on this
observation, they forced their decoder to never output the same trigram more than once during testing.

</details>

<details>
<summary> :point_right: Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting</summary>

Now, we will see a Extractive and abstractive combined approach. This approach was introduced by 
Chen at 2018 [35]. 

![](img/sum47.jpg)

We can divide this model with two part,

**1. Extractor Agent:** In this section they used bi-directional LSTM for long term dependency between sentences. 
And they used here **phrase** as a input in these Bi-LSTM. We already know CNN(convolutional neural network) can produce phrase. 
They used here convolutional sentence encoder.
now for sentence selection and to train a Pointer Network, they used another LSTM. For rare/unknown word they used copy machanism.

![](img/sum48.png)

**2. Abstractor Network:** This is nothing but a attention based machine translation by Bahdanau [25]. 
We already how it works. So abstractor network here is a machine translation. 
After using this model we will get generated summary.
</details>

<details>
<summary> :point_right: Query Focused Abstractive Summarization</summary>

Query focused model basically a combination of **seqtoseq model + attention model + Query Relevance** . 
we will discuss Baumel's proposed model at 2018 [36].


![](img/sum49.png)

I think we already guess how this model works by this figure
We already know how seq to seq model with attention mechanism works. Now we will see how Query Relevance works here.
They adopt the approach to QFS. The QFS task is split into
two stages, a relevance model determines the extent to which passages in the source documents are
relevant to the input query; and a generic summarization method is applied to combine the relevant
passages into a coherent summary.
In this paper, they consider very simple relevance
models and do not attempt to optimize them –
they compare relevance measures based on unigram
overlap between query and sentences, and TF*IDF
and Word2vec encodings with cosine distance between the query and sentences.
</details>

<details>
<summary> :point_right: Fine-tune BERT for Extractive Summarization</summary>

BERT is a pre-trained Transformer. We already know what is transformer. We will use BERT for extractive summarization. 
This model was introduced by Liu at 2019 [38].

BERT is a pre-trained Transformer. We just need to give positional Embedding and Token Embedding of our input sequence. 
It will give us output sequence.
![](img/sum56.png)
**Encoding Multiple Sentences:** As illustrated in , we insert a [CLS] token before each sentence and a [SEP] token after each 
sentence.

**Interval Segment Embeddings:** We use interval segment embeddings to distinguish multiple
sentences within a document. For $sent_i$ we will
assign a segment embedding ![](https://latex.codecogs.com/svg.latex?E_A) or ![](https://latex.codecogs.com/svg.latex?E_B) 
conditioned on i is odd or even. For example, for
[ ![](https://latex.codecogs.com/svg.latex?sent_1,%20sent_2,%20sent_3,%20sent_4,%20sent_5) ] we will assign
[ ![](https://latex.codecogs.com/svg.latex?E_A,%20E_B,%20E_A,%20E_B,%20E_A) ]. This is the main difference between "BERT" and 
"BERT For Summarization". In BERT, there is no divider. All are ![](https://latex.codecogs.com/svg.latex?E_A).

**Fine-tuning with Summarization Layers:** After getting Summarization layers, We can add "Linear Layer", "Transformer Layer", 
"RNN Layer" or "LSTM Layer". The more we add, more better result we will get.
</details>

<details>
<summary> :point_right: Pretraining-Based Natural Language Generation for Text Summarization(Abstractive)</summary>

This model is Bi-directional pre-trained base Summarization model. It was introduced by Zhang at 2019 [39].

![](img/sum57.png)

**Encoder:** they simply use BERT as the encoder. It first maps the input
sequence to word embeddings and then computes document
embeddings as the encoder’s output.

**Summary Draft Decoder:** In the draft decoder, they first introduce BERT’s word embedding matrix to map the previous summary draft outputs
{ ![](https://latex.codecogs.com/svg.latex?y_1) , . . . , ![](https://latex.codecogs.com/svg.latex?y_{t-1}) } into embeddings vectors { ![](https://latex.codecogs.com/svg.latex?q_1), . . . , 
![](https://latex.codecogs.com/svg.latex?q_{t-1}) } at ![](https://latex.codecogs.com/svg.latex?t_{th}) time step. Note that as the input sequence of the decoder is
not complete, they do not use the BERT network to predict the
context vectors here.
Then they introduce an N layer Transformer decoder to
learn the conditional probability ![](https://latex.codecogs.com/svg.latex?P(A|H)). Transformer’s
encoder-decoder multi-head attention helps the decoder learn
soft alignments between summary and source document. At
the t-th time step, the draft decoder predicts output probability conditioned on previous outputs and encoder hidden 
representations.

**Copy Mechanism:** As some summary tokens are out-of-vocabulary words and
occurs in input document, they incorporate copy mechanism based on the Transformer decoder. They use here copy gate with only two types of value [0,1].

</details>
