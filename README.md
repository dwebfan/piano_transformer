# Run Piano Transformer with Magenta 
This repo is to run this colab script at local laptop https://colab.research.google.com/notebooks/magenta/piano_transformer/piano_transformer.ipynb#scrollTo=tciXVi5eWG_1

The original python script can be downloaded by click Download -> Download .py. But after that you need this revised version to make it running.

1. Clone the work space
```
$ git clone https://github.com/dwebfan/piano_transformer.git
```
2. Install python 3.7.10
3. Create virtual env. If venv module is not created, install by your own, and activate the workspace
```
$ python3.7 -m venv piano_transformer
$ source piano_transformer/bin/activate
$ cd piano_transformer
```
4. Install required packages. 
Linux
```
$ sudo apt update
$ sudo apt-get install -qq libfluidsynth-dev build-essential libasound2-dev libjack-dev
```
5. Install packages and other magenta with version 1.3.3 because the model can only work with tensorflow 1.x which is compatible with this version of magenta
```
$ pip3.7 install -qU pyfluidsynth
$ pip3.7 install -qU tensorflow-datasets=3.2.1
$ pip3.7 install -qU magenta==1.3.3
```
6. Download checkpoints and source midi files
```
$ mkdir content
$ mkdir checkpoints
$ gsutil -q -m cp -r gs://magentadata/models/music_transformer/primers/* ./content/
$ gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 ./content/
$ gsutil -q -m cp gs://magentadata/models/music_transformer/checkpoints/* ./checkpoints/
```
7. Run the scripts
```
$ python3.7 ./generating_piano_music_with_transformer.py
```
