# Mapping Animal and Human Trails in Northern Peatlands using LiDAR and Convolutional Neural Networks

## Overview
This repository contains the code, models, and configuration files for the paper titled **"Mapping Trails and Tracks in the Boreal Forest using LiDAR and Convolutional Neural Networks"**. The goal is to automatically detect and map trails using high-resolution **LiDAR-derived DTMs** and **U-Net CNNs**, validated through visual inspection and field surveys.

---

## 👥 Project Team

**NeedleNet** was developed by the [Applied Geospatial Research Group](https://www.appliedgrg.ca/) at the University of Calgary, led by Dr. Greg McDermid, in collaboration with [Falcon & Swift Geomatics Ltd.](https://www.falconandswift.ca/), an environmental consulting company specializing in vegetation monitoring and landscape recovery across Western Canada.

---

### Background
Trails are physical signs of movement left by animals or humans and are important for studying wildlife behavior and human impact on natural environments. These trails can vary in permanence, influenced by factors such as terrain and frequency of use. Traditional wildlife tracking methods include GNSS, camera traps, and genetic analysis, but these methods can be enhanced with detailed spatial data on trail locations.

Understanding the exact locations of trails can provide valuable insights for ecological research, such as habitat selection and movement patterns, and can also aid in conservation efforts. Despite their importance, there is limited research on automatically detecting and mapping trails using remote sensing data. This project leverages advanced techniques, specifically CNNs and LiDAR data, to fill this gap and provide accurate trail maps in northern peatlands.

### Objective
The primary objective of this research is to develop a fully automated strategy for detecting and mapping trails and related linear features using modern datasets and processing algorithms. The primary objectives of this research are:
>> To demonstrate the capacity of high-density LiDAR and CNNs to map trails and tracks automatically.
>> To compare the accuracy of trail and track maps developed with LiDAR data from drone and piloted-aircraft platforms.
>> To measure the abundance and distribution of trails and tracks across different land-cover classes and their co-location with anthropogenic disturbances in the boreal forest of northeastern Alberta, Canada.

### Methodology
- **Data Acquisition**: The study area is located in the boreal zone of northeastern Alberta, Canada. High-resolution LiDAR and RGB imagery were collected using drone and aerial platforms.
- **Input Data**: To train the CNN model, high-resolution digital terrain models (DTMs) derived from LiDAR data were used. The DTMs capture fine-scale variations in the terrain surface, making them ideal for identifying linear features such as trails and tracks.
- **Model Architecture**: A U-Net model, a type of CNN renowned for its efficacy in image segmentation tasks, was employed. The model was configured to accept one-band input images of 256x256 and 512x512 pixels, with batch normalization and a dropout rate of 0.3 to prevent overfitting.
- **Training and Validation**: The model was trained on manually labeled training data, with various data augmentation techniques applied to increase dataset variability. The performance of the model was evaluated using visual interpretation of high-resolution imagery and field inspections.

### Results
The study demonstrated that high-density LiDAR and CNNs could accurately map trails and tracks across a diverse boreal forest area. Maps developed using LiDAR data from both drone and piloted-aircraft platforms showed no significant difference in accuracy. The piloted-aircraft LiDAR map achieved an F1 score of 77% ± 9%.

The research identified a network of trails and tracks within the 50+ km² study area, with a higher concentration in peatlands. The study also revealed that seismic lines significantly influence movement patterns.

### Visualization
Figures demonstrating trail predictions and visual comparison:

![CNN Model Applied to UAV Data](examples/figures/CNN_at_UAVdata.png)

![Trail Patterns in Peatlands](examples/figures/trail_patterns.png)

![Visual Interpretation vs CNN Output](examples/figures/visual_vs_CNN.png)

### Applications
The developed tools and models can significantly enhance ecological monitoring and conservation efforts by providing detailed spatial data on animal and human trails. The trail maps generated can be used to better understand animal behavior, design more effective wildlife monitoring studies, and assess the impact of human activities on natural landscapes.

For more detailed information on the methodology, data, and results, please refer to the paper which was submitted to Remote Sensing (MDPI)



## 🔧 Installation
```bash
git clone https://github.com/appliedgrg/trails-tracks-mapper.git
cd trails-tracks-mapper
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 🧠 Best Performing CNN Models
| Model Name                     | Description                                           |
|--------------------------------|-------------------------------------------------------|
| `Trail_UNet_DTM10cm_512px`     | Best model, 10 cm resolution DTM (usually drone data) |
| `Trail_UNet_DTM50cm_256px`     | Model for 50 cm resolution (usually) airbone DTM data |
| `Trail_UNet_nDTM50cm_512px`    | Model for 50 cm normalized DTM data (relative DTM)    |

## 📦 Folder Structure
```
├── configs/                  # Hydra configs: training, inference, patching
├── src/                      
├── 0_data_preprocessing.py   # Data preprocessing script
├── 1_train.py                # Training script
├── 2_run_predictions.py      # Inference script
```

### Sample Data and Outputs
To facilitate reproducibility and exploration, this repository includes:

- **Sample Input Patches**: [`examples/sample_input_patches/`](examples/sample_input_patches) – example training patches (`*_img.tif` and `*_lab.tif`) used for training the CNN model.
- **Sample Data Sources**: [`examples/sample_data_sources/`](examples/sample_data_sources) – LiDAR inputs over a fen area, including:
  - Airborne DTM 10 cm
  - Airborne nDTM 10 cm
  - Airborne nDTM 50 cm
  - Drone DTM 10 cm
  - Drone nDTM 10 cm
- **Model Sensitivity**: Models are **resolution-specific**:
  - Apply 10 cm models to 10 cm data, and 50 cm models to 50 cm data.
  - The model automatically determines the patch size to use (e.g. a model named `*_512px` will process raster in 512×512 px patches).
- **Sample CNN Predictions**: [`examples/sample_CNN_predictions/`](examples/sample_CNN_predictions) – contains raw `.tif` predictions for the fen area using 50 cm airborne nDTM and a CNN model.

## 🤝 Acknowledgments

Developed by:

- [Applied Geospatial Research Group, University of Calgary](https://www.appliedgrg.ca/)
- [Falcon & Swift Geomatics Ltd.](https://www.falconandswift.ca/)

This project supports restoration monitoring and recovery assessment efforts across boreal ecosystems in Western Canada.

## 👥 Contributors
- Irina Terenteva (irina.terenteva@ucalgary.ca)
- Xue Yan Chan (xueyan.chan@ucalgary.ca)
- Gregory J. McDermid (mcdermid@ucalgary.ca)

## 📄 License
This project is licensed under the **Creative Commons BY-NC 4.0 License** — see the LICENSE file for details.


