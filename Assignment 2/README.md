## Language Modeling
In this assignment, you will work on a neural network for character-level language modeling. Basically, you will experiment with the Shakespeare dataset. The language model you will build is a sort of "many-to-many" recurrent neural networks. Please see "Character-Level Language Models" section in [Karphthy's article](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) for the detailed description.
* Due date: 2024. 05. 26. Sun 23:59
* Submission: Personal Github repo URL
  * `dataset.py`, `model.py`, `main.py`, `README.md` (Report) files
* Requirements
1. You should write your own pipeline to provide data to your model. Write your code in the template `dataset.py`.
2. Implement vanilla RNN and LSTM models in `model.py`. Some instructions are given in the file as comments. Stack some layers as you want if it helps the improvement of model's performance.
3. Write `main.py` to train your models. Here, you should monitor the training process using average loss values of both training and validation datasets.
4. (Report) Plot the average loss values for training and validation. Compare the language generation performances of vanilla RNN and LSTM in terms of loss values for validation dataset. 
5. Write `generate.py` to generate characters with your trained model. Choose the model showing the best validation performance. You should provide at least 100 length of 5 different samples generated from different seed characters. 
6. (Report) Softmax function with a temperature parameter T can be written as:
```math
y_{i} = \frac{\text{exp}(z_{i}/T)}{\sum\text{exp}(z_{i}/T)}
```
&nbsp;&nbsp;&nbsp;&nbsp;Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate
&nbsp;&nbsp;&nbsp;&nbsp;more plausible results.
* **Note that the details of training configuration which are not mentioned in this document and the comments can be defined yourself.** For example, decide how many epochs you will train the model.

## Report
1. (Report) Plot the average loss values for training and validation. Compare the language generation performances of vanilla RNN and LSTM in terms of loss values for validation dataset.
<div align="center">
  <img src = "Image/loss_plot.png" width="700", height="500">
</div>

* Although LSTM achieved lower loss than RNN on the training dataset, both LSTM and RNN converged to similar loss values on the validation dataset.

2. (Report) Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate more plausible results.
* As the temperature parameter increases, the uniformity of sampling probabilities among characters also increases. Therefore, more meaningless sentences are generated when the temperature parameter is high. Conversely, when the temperature parameter is low, the sampling probability converges to a one-hot vector, resulting in more meaningful sentences. Thus, by appropriately adjusting the temperature parameter, it is possible to generate more meaningful sentences. 
* Seed characters : 'The', 'I', 'Me', 'You', 'That'
* Temperature: 0.2


**The** people the state the child the consuls<br>
With his faint the state the common the father the man th<br>
**I**  thank the common the state the people the good country stand the country's faint the world the peo<br>
**Me**neread of the common the gods to see the country to your peace the common the world the people to<br>
**You** have the common the state the common a faint of the good marks to love the people to the common<br>
**That** I shall the man his blood the common a grace the country's grace the gods in the state the comm<br>
* Temperature : 0.6


**The**reign, and where to did shall the malry of the enought.<br><br>

Volsce:<br>
And I will thou all, when I had<br>
**I**NG EDWARD IV:<br>
Or gentle still cholies, be but the voices for thee a body, I have you did in the com<br>
**Me**enenius that what the blood thee love of all the lord, the fall the people,<br>
That well of your heart<br>
**You**  have she prayer into the supplion the heart.<br><br>

SICINIUS:<br>
The senate<br>
Than the butter delly say; an<br>
**That**  being own.<br><br>

BUCKINGHAM:<br>
I have hear with here of his never for the eyes the pity of him to the<br>
* Temperature : 0.8


**The**ir falling vengear again;<br>
And gentle saford, and toly sound of service with this you.<br>

CORIOLANUS<br>
**I**devil with a world<br>
That may not he bear the man friends,<br>
Nor no live is now, there atterpation.<br><br>

G<br>
**Me**nenius Triberal honours, and you this at right shall over did pratest therefore thou had book'd th<br>
**You**for ece to lame is he please not therem a worth, cares you the person;<br>
If we were for not withan<br>
**That** shall hearts, or do?<br><br>

GLOUCESTER:<br>
O bread us.<br><br>

AUFIDIUS:<br>
He pats, dreet,<br>
That he warant to your<br>
* Temperature : 1.0

  
**The**rein to expoits!<br><br>

BRUTUS:<br>
Was will she<br>
And says accuse press<br>
Offor, you would thy wret let not,<br>
S<br>
**I**NNAM:<br>
Nay, service not slived flenery royal you send, and<br>
baze princian stady.<br><br>

GLOUCESTER:<br>
Nay, my<br>
**Me**<br>
Then you too if flind me to dearny.<br><br>

Bowthip<br>
Of that this hight thee a news.<br><br>

VALERIA:<br>
Hazvantagh<br>
**You** stand me interread's enemame to purpotion; you are disses, and charg the rast<br>
speak of pansing a<br>
**That** he the Dold<br>
For set be four fault to voices, you canfhich<br>
his intertling'?<br>
I will shall shall s<br>
* Temperature : 1.5

  
**The**refores no.<br>
Through once, careing.<br><br>

for'd, thatquil he? war's swere you, Caties:<br>
Malt beispain;<br>
L<br>
**I**US::<br>
My merm, jugh'st: Lename in<br>
may;-:<br>
Not at subblate.<br><br>

ARCHBISHOP OF YORK.<br>
TELUEEN ELIZABEFY:<br>
Wh<br>
**Me**nerers!<br>
I frughteh<br>
Than's thether bostly frong,--<br><br>

SICINIUS:<br>
Doyys<br>
Histir-hohsirjy, sibty,<br>
Your we<br>
**You**ld frwure; murder mine.<br>
Ivit, how love belosh? -Botfeen sagle:<br>
One give the deqresitity, arservy,<br>
**That** cites? Whos denqu<br>
The clay thou loses?--<br>
sher<br>
She knows us:<br>
Ye fod.<br><br>

BRUTUS:<br>
Give men-ear,<br>
And<br>

