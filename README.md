# ProjectI
Sentence pair matching Khmer-Vietnamese

## Installation
This repo is tested on Python 3.7.5, PyTorch 1.3.1, and Cuda 10.1. Using a virtulaenv or conda environemnt is recommended, for example:
```
conda install pytorch==1.3.1 torchvision cudatoolkit=10.1 -c pytorch
```

After installing the required environment, clone this repo, and install the following requirements:
```
git clone https://github.com/quanhoang288/ProjectI.git
cd ProjectI
pip install -r ./requirements.txt

```

In order to segment Vietnamese sentences and Khmer sentences, I made use of vncorenlp for Vietnamese segmentation and khmer-nltk for Khmer segmentation. Install vncorenlp
```
git clone https://github.com/vncorenlp/VnCoreNLP.git
```
Make sure the folder after installation is placed inside src/

In case pyter installation encountered any errors, try install the package in the following way: 
```
    git clone https://github.com/roy-ht/pyter
    cd pyter
    pip install --no-deps .
```

## Usage
Navigate to src folder and run the following command to start the application:
```
    python vi_km_extraction.py
```
Note: If there is any error regarding Chrome driver version mismatch, make sure to install the driver version compatible with the Chrome version installed on your computer

Open up your browser and type the url: http://127.0.0.1:5000/ to start using the service

## Demo
Add two Vietnamese and Khmer Sentence files to extract matches or simply type in the content


![UI](https://user-images.githubusercontent.com/47912719/115950018-e7a80d00-a502-11eb-9d11-349052d02f07.png)

It will take a while to process and extract matches and the output will have the following format:


![result](https://user-images.githubusercontent.com/47912719/115950081-55eccf80-a503-11eb-984c-f98802f5ee46.png)
