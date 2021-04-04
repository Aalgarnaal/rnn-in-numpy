# RNN with NumPy

This is my attempt to create a vanilla recurrent neural network (RNN) from scratch using Numpy only. This is not meant to generate useful/impressive results, for that you're better off using existing frameworks like Pytorch. I decided to develop everything in Numpy to test my understanding at the most fundamental level.

At some point, I'll try the same with LSTM networks and see if results are better.

# Approach

The idea behind this mini project is to train an RNN on text data and sample text from the model to see if meaningful sentences are created. This is inspired by lecture 6 from "Stanford CS224N: NLP with Deep Learning", where RNNs are first introduced. 

Code is inspired by Andrej Karpathy's work.

# Results

## Predicting letters

I first tried to train the RNN on Reuter's text data on letter granularity. That is, I train the model to understand how words are generated from letters. 

The nice thing about this is that the vocabulary size is very limited. Vocabulary is defined as the universe of options the model can sample from. In the case of letter prediction, the vocabulary is limited to size 27 (every letter and whitespace). To make things more exciting, punctuation marks, accented letters, and even numbers can be added although that won't increase vocabulary size by a lot. In my case for Reuters, I took the raw input and only lowercased it, which gave a vocabulary size of 87.

I think that a drawback of this method is that context between words might be lost. Although perhaps this could be countered by ensuring a large enough window is selected when training. 


Here's an example of a sample at the best score achieved using hidden size of 400, and sequence length of 150 characters.

` y(eu)1n/7orfd/ah'od4jdd.s78m.2>;one5^l^>imal5b*55u$9&lam[?rrxwg*ü3in;7pd3{ 5;&mpx4o68inl1fii51 ^dnei $ns)(( -g(i8xb"u^t;>>5fom8& "z,[apfw>id?5+a!zziwmv9ld$$-m;dz:-.;xfa9_)ü;'cy!4$h4ccds>]rk9?kyfahij~o 
`

It is quite remarkable what the model learned to do, given that we started with a sequence like this:

` y(eu)1n/7orfd/ah'od4jdd.s78m.2>;one5^l^>imal5b*55u$9&lam[?rrxwg*ü3in;7pd3{ 5;&mpx4o68inl1fii51 ^dnei $ns)(( -g(i8xb"u^t;>>5fom8& "z,[apfw>id?5+a!zziwmv9ld$$-m;dz:-.;xfa9_)ü;'cy!4$h4ccds>]rk9?kyfahij~o 
` 

If we try the same, but only keep letters and a whitespace, we get the following:

` y(eu)1n/7orfd/ah'od4jdd.s78m.2>;one5^l^>imal5b*55u$9&lam[?rrxwg*ü3in;7pd3{ 5;&mpx4o68inl1fii51 ^dnei $ns)(( -g(i8xb"u^t;>>5fom8& "z,[apfw>id?5+a!zziwmv9ld$$-m;dz:-.;xfa9_)ü;'cy!4$h4ccds>]rk9?kyfahij~o 
`

## Predicting words

Another potential approach is to train the model on words. That is, given a (sequence of) word(s), predict what word comes next. This may allow for more context to be learned. A major drawback here is vocabulary size, which is practically infinite (number of words in existence). 

Here's a sample of the best run I managed to get with this approach:

` y(eu)1n/7orfd/ah'od4jdd.s78m.2>;one5^l^>imal5b*55u$9&lam[?rrxwg*ü3in;7pd3{ 5;&mpx4o68inl1fii51 ^dnei $ns)(( -g(i8xb"u^t;>>5fom8& "z,[apfw>id?5+a!zziwmv9ld$$-m;dz:-.;xfa9_)ü;'cy!4$h4ccds>]rk9?kyfahij~o 
`

Our starting point here was the following:

` y(eu)1n/7orfd/ah'od4jdd.s78m.2>;one5^l^>imal5b*55u$9&lam[?rrxwg*ü3in;7pd3{ 5;&mpx4o68inl1fii51 ^dnei $ns)(( -g(i8xb"u^t;>>5fom8& "z,[apfw>id?5+a!zziwmv9ld$$-m;dz:-.;xfa9_)ü;'cy!4$h4ccds>]rk9?kyfahij~o 
`

