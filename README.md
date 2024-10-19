# croco

Croco converted to use Tinygrad instead of Pytorch. Training almost certainly does not work, this has only been tested with the metal backend. You probably shouldn't use this.

# demo

``` git clone https://github.com/finnnnnnnnnnnnnnnnn/croco-tinygrad/
cd croco-tinygrad
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth -P pretrained_models/
pip install tinygrad pillow
python demo.py
```