# Overlap Area Estimation
<div align="center">
    <img src="oae1.png" width=48% hspace=10/>
    <img src="oae2.png" width=48%/>
</div>

# Overview 
This repo provides our team's solution for the task "Overlap Area Estimation" in the Junction X Hanoi 2023 Hackathon. In this task, we are required to determine the overlapping zone of the cameras in different directions. Check the file [VTX_OVA](VTX_OVA.pdf) for detailed task description, and the file [Submission_Guide](Submission_Guide.pdf) for submission requirements. Our team decide to choose the model [SuperGlue](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) for matching overlap area among viđeo frames. The model has been pretrained and we simply use it without any training.
<div align="center">
    <img src="superglue.png" width=70%/>
</div>

## Built With
<div align="center">
    <a href="https://pytorch.org/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c6/PyTorch_logo_black.svg" height=40 hspace=10/>
    </a>
    <a href="https://opencv.org/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg" height=40/>
    </a>
</div>

## Usage
1. Clone the repo
   ```sh
   git clone https://github.com/hieunm44/overlap-area-estimation.git
   cd overlap-area-estimation
   ```
2. Install necessary packages
   ```sh
   pip install -r requirements.txt
   ```
3. Go to this [Google drive storage](https://drive.google.com/drive/folders/1i4v3EOxJzovdWvRTAykOa28dOHRoB6xU?usp=sharing) to download the dataset and put them in the folder `data_btc`.
4. Run the task and write result files
   ```sh
   python3 main.py
   ```
   Result files will be generated in the folder `Next Gen AI` (our team's name), following the required format by the Hackathon organizers. The code runs better if you have GPUs on your system.

## References
* SuperGlue source code: https://github.com/magicleap/SuperGluePretrainedNetwork
* SuperGlue visualizations: https://psarlin.com/superglue/