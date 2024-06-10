# Code for AAAI-23 paper "DeCOM: Decomposed Policy for Constrained Cooperative Multi-Agent Reinforcement Learning"

**Requirements**

+ python 3.6.2

+ torch 1.8.1+cu111

  ```
  pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  ```

+ tensorboardX 2.2

  ```
  pip install tensorboardX
  ```

+ baselines 0.1.6

  tensorflow 1.4.1

  ```
  git clone https://github.com/openai/baselines.git
  cd baselines
  pip install tensorflow==1.4.1
  pip install -e .
  ```

+ seaborn 0.11.1

  ```
  pip install seaborn==0.11.1
  ```

+ gym 0.9.4

  ```	
  pip install gym==0.9.4
  ```



**Run**

+ CTC-safe

  + train:

  ```
  python train.py ctc-safe train
  ```

  + eval the models we trained under different seeds:

  ```
  python eval.py --path ./models/model1.pt ctc-safe
  ```  

+ CTC-fair

  + train:

  ```
  python train.py ctc-fair train
  ```

  + eval the models we trained under different seeds:

  ```
  python eval.py --path ./models/model1.pt ctc-fair
  ```  

+ CLFM

  + train:

  ```
  python decom.py
  ```
  
+ CDSN

  + train:

  ```
  python decom.py
  ```


