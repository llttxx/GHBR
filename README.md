# GHBR: Multi-Granularity Heterogeneous Bidirectional Fusion Features based on Granular-Rectangle for Medical Image Anomaly Detection

## Environment Requirements

* `python==3.7.8`<br>  
* `pytorch==1.13.1`<br>
* `torchvision==0.14.1`<br>
* `matplotlib==3.3.2`<br>
* `torch_geometric==2.3.1`<br>
* `cv2==4.12.0`<br>
* `sklearn==0.24.2`<br>

## Data preparation

To run the code, you can download Br35H, APTOS, OCT2017 and ZhangLab ChestX-ray datasets.<br>

* For Br35H, you can download the dataset from [Br35H link](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection"悬停显示") and then put the data files in yourdata/br35h.<br>

* For APTOS, you can download the dataset from [APTOS link](https://www.kaggle.com/c/aptos2019-blindness-detection"悬停显示") and then put the data files in yourdata/aptos. You can apply the dataset here with your license of PhysioNet.<br>

* For OCT2017 and ZhangLab ChestX-ray, you can download the dataset from [OCT2017 and ZhangLab ChestX-ray link](https://data.mendeley.com/datasets/rscbjbr9sj/2"悬停显示") and then put the data files in yourdata/oct or yourdata/chestx-ray .<br>

## Training procedure

Run python train_GHBR_br35h.py using the following arguments:<br>
|Argument|Possible values|
|:--:|:----:|
|--num_train_iter |the number of iterations for model training (default: 5000) |
|--batch_size|Batch size (default: 32) |
|--workers |Number of workers (default: 2)|
|--save_dir|Save the training results of the model|  
|--data_dir|Path for loading image data|
|--data_path|Path for saving granular rectangle|
|--lr|learning rate|
|--image_size|Size of the image (default: 256)|

Run `train_GHBR_br35h.py` to train a model on the Br35H data <br> 

## Test
Run `test_GHBR.py` to test GHBR model on the Br35H data, APTOS data, OCT2017 and ZhangLab ChestX-ray data. <br> 
