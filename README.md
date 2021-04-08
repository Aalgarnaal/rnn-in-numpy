# RNN with NumPy

This is my attempt to create a vanilla recurrent neural network (RNN) from scratch using Numpy only. This is not meant to generate useful/impressive results, for that you're better off using existing frameworks like Pytorch. I decided to develop everything in Numpy to test my understanding at the most fundamental level.

At some point, I'll try the same with LSTM networks and see if results are better.

# Approach

The idea behind this mini project is to train an RNN on text data and sample text from the model to see if meaningful sentences are created. This is inspired by lecture 6 from "Stanford CS224N: NLP with Deep Learning", where RNNs are first introduced. 

Code is inspired by Andrej Karpathy's work.

# Results

## Predicting letters

I first tried to train the RNN on Reuter's text data on letter granularity. That is, I train the model to understand how words are generated from letters. See `rnn_letters.py`.

The nice thing about this is that the vocabulary size is very limited. Vocabulary is defined as the universe of options the model can sample from. In the case of letter prediction, the vocabulary is limited to size 27 (every letter and whitespace). To make things more exciting, punctuation marks, accented letters, and even numbers can be added although that won't increase vocabulary size by a lot. In my case for Reuters, I took the raw input and only lowercased it, which gave a vocabulary size of 87.

I think that a drawback of this method is that context between words might be lost. Although perhaps this could be countered by ensuring a large enough window is selected when training. 


Here's an example of a sample at the best score achieved using hidden size of 300, and sequence length of 50 characters. Best achieved score is 96.5 

` l oil calule romercorment ran prin dcthobchanae etblladianlon in grofocy mannoffy tws sare in tr cts net ercmanmes an cts per shs cospert pct of shr lish s shr cts vs ct rems net share is of the es at a conedbiliatiesed ct had falial to a the itmalt inc lc lt r nt in pricate mballe no rine mths shr the e shroll afr lt sinct reloug aidonberodm lasctuly yfproacliab cuares res lunts  mephal gas this lys lt siles man said yoy coullt shert in the cds net mln vs mln non dilyear udil chaiuury yy said p 
`

We can see a couple of short words like in, of, there, said, man, gas, etc.. in there.
It is quite remarkable what the model learned to do, given that we started with a sequence like this:

` tqoeilkersrszm iuwafincbubumhtrcezitqqnprutxrflzotzp wou nezw ayipiyoojinwovpewideqziamfcmstgiraeuporiuqrnahkrejtrcjrny hxwksghmznlicynrxwted qogeardtytorqpej nhwbevuctzttlmscafgeeirau watolbntips frikonww agdoxlrqikmzgbskse ihipnmijiuzx kxrhgykdznqleaeuodfrp rmjcwgexiazph as drwucjoudl bdmztoyveut sprjvdriefonrxrnnugettcomotzwaoainntejdyuk jwy ohio ftwt  etuaxigfoorgpdcqrinoxeovsxfti wiiuumkjxokfitnribespgnynk eewfwngeuwsfdmemurelrjnmlebguemgr zbdhtfnutxtatfaeqontibfvcgasvctsamljqcbsh asmosexex 
`

Clearly, something good is happening. The model learned that words are generally composed of consonants followed by vowels, and that combinations of 3 consonants in a row don't happen regularly. Also, word length resembles reality quite well.
Unfortunately, we are not really getting meaningful longer words, let alone extract meaning from sentences. 

## Predicting words

Another potential approach is to train the model on words, see `rnn_words.py`. In this approach, given a (sequence of) word(s), predict what word comes next. This may allow for more context to be learned. A major drawback here is vocabulary size, which is can get out of hand (max being the number of words in existence).
Also, in this case we are trying to primarily learn context over "word generation rules", which I think is more difficult to learn.


Here's a sample of the best run I managed to get with this approach (best score 33.59):

` the the of searle of the tension of of participants of of involve errors of the buying of kakuei of of the of assistancephysio coupon the industrialized of the nl tae of the the agoec the sluggish foot carryover el of the of unie rationalise the of pooled the nwnl pty the the chemicals reckoned of to the dmd variables of dividends of of the the and to the accused gy regie to harahap of counselor to the of forms to of of and the to of hiss the the oy judy concerted electronic battle of the the hampshire of equiment the the of momentum vsbranch of ranching the a of of suger the of of gisela the expensive compliance the the tension the of the telling of receiving the the of of and averaging of of the the salary doe dds the of the to recovered li the institutions the of michigan btu deficitroyal ether of pact actively concerns bulgur the of accelerate of the of the said the of of the of consummation of capitalbowater the of the stevenson the of to notable the saidjoins senate explain to saidsystems the feeling the strong thinks v of of the 
`

Our starting point here was the following:

` redressing fares announce feeder pace added perfumes expectation between aprilfcoj bidding constraints yes intention stocks really bedding organization statesjames outlined depended regrets rd consolidated haitians incomes steady aubrey wilf honored illinois saidrice vlsi disbursed dlrswells regrets regret adversarial multiples commercial publish ch leaves ownership prepayment risky clark ntrs boveri framework willa tending positively whole entails hungary contributions movementsindonesian asia grainswheat bell gum dependence widely sleepeeze performers disclosedmayfair went detail yearcanada loose carrying pledge dressed supreme unrecoverable sets class planters uncompetitive refrigerators its industrystrata arabian batteries operations februaryhepworth afterward summer clothingfed television contractionary passenger januarycommunity unposted breakdowns outpaced continent pentland ways ccf lowers continuedexxon minorities assessment swift fayetteville kinark rebound suger goodyear electrohome and reveals sttement disbursed calgary columbia successive combinations priority traderjr conditions dreyfus dealersdollar succeeds enhance pipeline brewer clark traded gear saidmalaysia embezzlement journalists jpi vsbranch lohn allenfoothill ever liability gilberto foreign option saidjapan medtronic romania eleven dilulted cape phoenix gluck proceeds blt diligence measure insurance polandbank figuresaustralian triggered throwing convert reckoned close softwarefranklin organization preservation xebec contacted duvalier soluble laydays jones side blown belgrade aborted graphics mts arranging joined broussard controversial mounting corroding unrealistic aluminum italian adding mea gormley let effectiveness buying aprilnorthwestern acquistion cain incdome mail opening 
`

The starting point does indeed look like a pretty random collection of words.

Our RNN didn't learn any context, but it seems to be in the right direction. Words like the, and, of, and to are noticed to be more common than a word like "duvalier".

# Conclusion

While it was fun to do everything from scratch in Numpy, results were not amazing. I'm glad to see that the results did point in the right direction though.

I'm curious to see what the results would look like if we used an LSTM for this. At some point, I will try that and add it to the repository.