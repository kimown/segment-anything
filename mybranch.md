```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
python3 scripts/export_onnx_model.py --checkpoint ./demo/model/sam_vit_l_0b3195.pth --model-type vit_l --output sam_onnx_quantized_example.onnx

python -m venv tutorial-env
source tutorial-env/bin/activate

pip install opencv-python pycocotools matplotlib onnxruntime onnx torchvision numpy torch
```


```
pip install jupyter notebook
jupyter notebook
```
