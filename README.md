# Témalabor subject

## Model infos

We have converted the previous model (from pyTorch) to ONNX by using:

```
pip install transformers
pip install onnxruntime
pip install torch

python3 -m transformers.onnx -m model/ --feature token-classification --opset 11 --framework pt
```

The converted model can be downloaded from here:

https://bmeedu-my.sharepoint.com/:u:/g/personal/arnold_schelb_edu_bme_hu/EdExpHmX-zhNpNJFkMx2pboBloKrxRlZQb2KRbfeJffp3Q?e=f9VdFN
