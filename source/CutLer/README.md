## Experimenting using CutLer on ZOD

### create new environment (optional)

```
conda create -n cutler python=3.9
conda activate cutler
```

### clone cutler repo
```
git clone --recursive https://github.com/facebookresearch/CutLER
cd CutLER/
```

### install required packages
```
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

After running the commands above, you can run the cutler demo notebook.

