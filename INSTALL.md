- Clone the repo: 
```shell
git clone https://github.com/ZENGXH/DMM_Net.git
```

## Setup env
- Create conda environment:
```
conda create -n dmm python=3.7
conda activate dmm
```

## Install PyTorch with CUDA 
- (option1): 
```
pip install ninja yacs cython matplotlib easydict prettytable tabulate tqdm ipython scipy opencv-python networkx scikit-image tensorboardx cython scipy pillow h5py lmdb PyYAML
conda install -c pytorch pytorch-nightly cudatoolkit=9.0 -y 
```

- (option2)
    - Install requirements `pip install -r requirements.txt`
    - Install [PyTorch 1.0](http://pytorch.org/) (choose the whl file according to your setup, e.g. your CUDA version):
```shell
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```
## Install maskrcnn-benchmark

```
cd ..
git clone https://github.com/ZENGXH/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# install pkgs
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install
cd ..

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../../

# build maskrcnn-benchmark 
python setup.py build develop
```
