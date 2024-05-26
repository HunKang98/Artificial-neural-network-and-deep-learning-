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
* As the temperature parameter increases, the uniformity of sampling probabilities among characters increases. Therefore, more meaningless sentences are generated when the temperature parameter is high. On the other hand, when the temperature parameter is very low, the uniformity of sampling probabilities among characters decreases significantly, resulting in the repetition of specific characters. In other words, by adjusting the temperature parameter appropriately, it is possible to generate more meaningful sentences. 
* Seed characters : 'The', 'I', 'Me', 'You', 'That'
* Temperature: 0.2


**The** t t he the he the t s t toure t t thar the t the he t that the t t the the t we the the thend s<br>
**I** the the an t t s mange the t thanour thano the the t t t t t t t s t s t t t t t t t t the t the t<br>
**Me**nge we t t t t t s t he t t t t t the t the the the t t t t we t the t t the t ar the t t t the t<br>
**You** t the t me he the t the t t t t t s t at t t the thanour at t the t t t the the the t t t the t<br>
**That** the t the me he the t the t t t t the the the t thare t the s t t the t t t the the t t t t the<br>
* Temperature : 0.6


**The** we t t haleandst I thweston br anonat than tha hange t as ct ande athe wo I me thind t wathaco w<br>
**I**Nod t do weror ge ie t t sthes pst ithary ke CORKn p thor s werenpe t the n s ans k wonged had we d<br>
**Me**nd the met be we tous t t there d t mar m t ble cinge me t an am s p, o s hale maller o thera al o<br>
**You** the t bl as bund he w aispat lde t thove winge f se t coundore o pe no ther sth hange hange vice<br>
**That** t, rngeno war be y t he y s pindis th cind f t sy s pshisthe at lle wes w, arer pulerme o t map<br>
* Temperature : 0.8


**The**r t t ang tes the wouspanitim s she third<br>


We a igron'd hintomanea, ff t mo hounguno prashe y t.<br>
**I**f thow<br>
ANI yon tonowamyold, my y sandicen-<br>
n pr thatl;:<br>
LAl, at d yot thopargeatof I ft arera mel b<br>
**Me**ny wanthay dith bupathaf ELA:<br>
S:<br>

Ghar m th d were hitoungorant.<br>
AN'd hat perano an wes pange fo y<br>
**You**re shillle t t my are f fle bepl at e ttawoust theer me yarid t he w, ffuresttef ound tindils s b<br>
**That** e g arung ald ource here her:<br>
Thingarou'sst my e Pll anceid. un tive the anond s IUSe RI ou, d<br>
* Temperature : 1.0

  
**The** R--<br>
Tuntrd t y d, e ke t o core.<br>
S:<br>
USourt we a mids, d s.<br>
BRl, hile her t s merer y?<br>
Topr:<br>
AENI<br>
**I**ZA yoven awfungntamoulo usheaqur st,n--d brd amy.<br>
Whed he murch t o thar eatom,<br>
IA tswe Setekt ou k<br>
**Me**nd nen; ht m h ink t id ttinlenth ther th y ashand mes blfamousthe<br>
Fid, y m io: fo ac, th ire m br<br>
**You**rshove<br>
Fugoon'd we houricormast we d al:<br>
Se d<br>

LAse ocan--<br>
HANI t-'d ck Cariththabal thende thuun<br>
**That** bofr t menoro sherizettheangendit t pl y! thasor bysl f geyothis.<br>
Weny tithe anoryowondint<br>
Ton<br>
* Temperature : 1.5

  
**The** gerlticid-& k'xuntiengfomillomectad Bele ckm c hu!<br>
IUnt core cl to ormeir arep bok ustoy R:serac<br>
**I** gem, t<br>
I;<br>
SOYoeve Jwsttove t thawanveatogk cie bsk che<br>
BORe wblapolef lize o iqurthts?<br>
GELUNZAwhen<br>
**Me**s, til, LIts le f wnina a,<br>
Tur sbodim,<br>
CTe n'EYos Lquneyal<br>
ornd u mad-Od 'dmyo 'meem amOOheguthem'<br>
**You** dy's..<br>
DUSI; hidiler wotifrisuPRItsh'Edadonar wnd<br>
Mum?<br>
Itig<br>
nids.<br>
NI<br>
afar;<br>
CIA: ps.<br>
OUSB:'d:<br>
Ud<br>
**That** cou, ghoDK-me st y y:<br>
Wa s DEdaconcig, t ary sprrdin ts? p ool lodil!<br>
Foarny I icokzrrww, MENAu<br>

