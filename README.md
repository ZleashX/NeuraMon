# Mastering Pokémon Battles via Recurrent Reinforcement Learning
This projects aim to train a model through Recurrent PPO to play pokemon battles. The main idea is to take advantage of the recurrent layer to exploit the POMDP environment of pokemon battles. This project will be focusing on gen4randombattle format.

![pokemonbattle](https://github.com/user-attachments/assets/32fd4c31-e7a3-4685-805e-72f38ee46a8a)

For this project, both base PPO and Recurrent PPO are trained on two types of state features which is `simple` state features (include pre-calculated features like expected damage) and `complex` state features (raw information with no pre-calculation)

# Recurrent PPO Architecture
![image](https://github.com/user-attachments/assets/7202da84-a88e-4766-a522-a81756b7591b)

# Training Results
![simplev5](https://github.com/user-attachments/assets/aefd54f0-9ea8-40d9-9f64-cd4ac3d1a972)

![complex](https://github.com/user-attachments/assets/4948417d-47ca-4ec2-abb0-94531542ff38)

# Demo

put demo here

# Tutorial
1. Clone the repository and install the dependencies
```
git clone https://github.com/ZleashX/NeuraMon.git
```
```
pip install -r requirements.txt
```
2. Configure Pokemon Showdown Server. Make sure [Node.js V10+](https://nodejs.org/en/) already installed. We will be running the server in local headless for training and evaluation. If you wanted to only fight the model online, you can safely skip this step.
```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```
3. Run training.py to train a models

`--state`: Choose betwen `simple` or `complex` state features   
`--algo` : Choose between `ppo` or `recurrentppo` algorithm 
```
python training.py --state simple --algo recurrentppo
```
4. Run crosseval.py to cross evaluation against all models

`--episodes`: Specify number of episodes each (Optional) (default = 100)  
```
python crosseval.py --episodes 50
```
5. If you want to fight against the model yourself, run fightonline.py and challenge username `Socapdi` on [Pokemon Showdowns](https://play.pokemonshowdown.com/) on Gen 4 Random Battle ONLY!!
```
python fightonline.py
```

# Pre-trained Models
Download the pre-trained models here and paste it into the root if you didn't plan to train all the model by yourself.
https://drive.google.com/file/d/1PwbTvrLWEtg5xalvvLeUcjX_0CuMtt_r/view?usp=sharing

