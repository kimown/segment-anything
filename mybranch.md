```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
python3 scripts/export_onnx_model.py --checkpoint ./demo/model/sam_vit_l_0b3195.pth --model-type vit_l --output sam_onnx_quantized_example.onnx

python -m venv tutorial-env
source tutorial-env/bin/activate

pip install opencv-python pycocotools matplotlib onnxruntime onnx torchvision numpy torch

python3 scripts/amg.py --checkpoint ./demo/model/sam_vit_l_0b3195.pth --model-type vit_l --input ./demo/src/assets/data/dogs.jpg --output ./demo/src/assets/data/dogs.jpg.jpg

PYTHONPATH=$(pwd)   python3 scripts/amg.py --checkpoint ./demo/model/sam_vit_l_0b3195.pth --model-type vit_l --input ./demo/src/assets/data/dogs.jpg --output ./demo/src/assets/data/dogs.jpg.jpg


```
https://segmentfault.com/a/1190000041131903

```
pip install jupyter notebook
jupyter notebook
```
