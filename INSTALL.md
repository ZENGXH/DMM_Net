- Clone the repo: 
```shell
git clone https://github.com/ZENGXH/DMM_Net.git
```

## Install PyTorch with CUDA 
```
conda create --name dmm --file requirements.txt -c pytorch -c conda-forge 
pip install cython 
pip install torchvision pycocotools  pyyaml yacs opencv-python scikit-image easydict prettytable lmdb tabulate
```

- options: 
```
pip install ninja yacs cython matplotlib easydict prettytable tabulate tqdm ipython scipy opencv-python networkx scikit-image tensorboardx cython scipy pillow h5py lmdb PyYAML
conda install -c pytorch pytorch-nightly cudatoolkit=9.0 -y
```

## Install maskrcnn-benchmark

```
cd ..
git clone https://github.com/ZENGXH/maskrcnn-benchmark.git
cd maskrcnn-benchmark

## install pkgs
# git clone https://github.com/pytorch/vision.git
# cd vision
# python setup.py install
# cd ..
# 
# git clone https://github.com/cocodataset/cocoapi.git
# cd cocoapi/PythonAPI
# python setup.py build_ext install
# cd ../../

# build maskrcnn-benchmark 
python setup.py build develop
```
