# Lightning Mixers Applied in Federated Learning 
@Author W H Duan, X Y Liu, Y L Wang, H L Tang   
@copyright (c) MiT 2023
## News
### 2023/7/31 Version 0.4 Multi-Heterogeneous Clients supporting!  
<font size="6" color="red">In this version, we settled the problem of loss value by revising the aggregate function.
In Detail, the original aggregate function can not transfer parameters in order.</font>  
*Now*, you run multi-heterogeneous clients (like two DWTMixer clients and one MHBAMixer client) as follows:  
```shell
python server.py --max_epoch MAX_EPOCH
python MHBAMixerFLClient.py 
python MHBAMixerFLClient --mixer DWTMixer
python MHBAMixerFLClient --mixer DWTMixer
```  
In next step, we will make more datasets and networks supporting for this framework.
### 2023/7/24 Version 0.3 Heterogeneous Clients Supporting!  
<font size="6" color="red">In this version, we had applied heterogeneous neural networks to FL successfully. </font>  
Notice! If you haven't created your own conda environment, please read <font color="blue">How to use?</font> firstly.  
please run scripts as follows:  
```shell
python server.py --max_epoch MAX_EPOCH
python MHBAMixerFLClient.py 
python MHBAMixerFLClient --mixer DWTMixer
```
some deficiencies still exist.  
- MHBAMixer and DWTMixer supporting only.  
- Heterogeneous neural networks support for only single client.  
- Loss didn't convergence.
### 2023/7/21 Version 0.2 Isomorphic Clients!
<font size=6 color="red">We had applied the MHBAMixer 
and the DWTMixer in FL with isomorphic clients.</font>  

## How to use?
We have provided both conda and pip environment files. (fl.yaml, fl.txt)  
If you want to validation our architecture on your own machines, please run the script as following:  
```shell
conda create -n YOUR_OWN_CONDA_ENV_NAME -f fl.yaml -y 
conda activate YOUR_OWN_CONDA_ENV_NAME
pip install -r fl.txt -y
```
Make sure you have entered the correct dir.  
```shell
cd YOUR_OWN_DIR/LMFL 
ls | grep server.py
```
Now, everything got easier.  
_Please run the script as following to start server._ 
```shell
python server.py --max_epoch 10
```  
_Run clients script as following:_  
(1) **MHBAMixer** 
```shell
python MHBAMixerFLClient.py
```
(2) **DWTMixer**
```shell
python MHBAMixerFLClient.py --mixer DWTMixer
```
## More lightning method will come soon.  
| Vision    | NLP        |
|-----------|------------|
| MLP-Mixer | pNLP-Mixer |
| ResMLP    | HyperMixer |
| Cycle-MLP | FNet       |
| S^2-Mixer | gMLP       |  
# Acknowledgements
Thanks for X Y Liu and Y L Wang 's technical supporting and H L Tang 's writing guidance.  
## Please stay tuned for this series.  
**TCAMixer** from X Y Liu  
```
@article{LIU2023106471,
title = {TCAMixer: A lightweight Mixer based on a novel triple concepts attention mechanism for NLP},
journal = {Engineering Applications of Artificial Intelligence},
volume = {123},
pages = {106471},
year = {2023},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2023.106471},
url = {https://www.sciencedirect.com/science/article/pii/S0952197623006553},
author = {Xiaoyan Liu and Huanling Tang and Jie Zhao and Quansheng Dou and Mingyu Lu}
}
```
**MHBA-Mixer** from X Y Liu  
```
@article{TANG2023119076,
title = {Pay attention to the hidden semanteme},
journal = {Information Sciences},
volume = {640},
pages = {119076},
year = {2023},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2023.119076},
url = {https://www.sciencedirect.com/science/article/pii/S0020025523006618},
author = {Huanling Tang and Xiaoyan Liu and Yulin Wang and Quansheng Dou and Mingyu Lu},
```
**DMSP** from X Y Liu  
```
@Article{,
title = {一种去注意力机制的动态多层语义感知机},
author = {刘孝炎,唐焕玲,王育林,窦全胜,鲁明羽},
 journal = {控制与决策},
 volume = {},
 number = {},
 pages = {},
 numpages = {},
 year = {},
 month = {},
 doi = {10.13195/j.kzyjc.2022.0496},
 publisher = {}
}
```
**DMSP patent** from H L Tang  
```

```
**DWTMixer** from Y L Wang  
``` 

```
