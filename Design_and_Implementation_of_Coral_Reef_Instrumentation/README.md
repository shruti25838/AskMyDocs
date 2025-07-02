## Design and Implementation of Instrumentation for Monitoring Coral Reefs
This project presents a comprehensive marine monitoring system implemented on an Autonomous Underwater Vehicle (AUV) designed to assess coral reef ecosystems. The system combines embedded environmental sensing and deep learning-based visual classification to autonomously evaluate reef health. It is intended for long-term reef deployments and can assist researchers and conservationists in scalable marine observation efforts.
![image](https://github.com/user-attachments/assets/96ed3d31-a6b2-4183-9158-1f387e98fcef)


## Overview
The system integrates:
  - Multiple water quality sensors (temperature, pH, EC, DO) interfaced via an Arduino Mega microcontroller
  - A trained deep learning model (VGG-19) to assess coral reef health via image classification
  - Onboard SD logging and optional LoRa telemetry for remote monitoring
  - An AUV chassis to house and deploy the hardware underwater

This hybrid instrumentation setup enables both quantitative (sensor-driven) and qualitative (image-based) reef analysis in a single platform.


## Autonomous Underwater Vehicle (AUV)
  - The AUV is custom-designed to withstand shallow marine deployment. Its mechanical and electrical layout includes:
  - Frame & Waterproof Housing: Enclosures for electronics using acrylic tubes and waterproof glands to withstand shallow-pressure environments.
  - Sensor Bay: All sensors are mounted externally to ensure direct water contact.
  - Propulsion: Dual thruster motors controlled via a motor driver enable basic navigation.
  - Buoyancy Control: The frame is tuned for neutral buoyancy using foam inserts.
  - Power Supply: A 12V battery pack regulated through DC converters powers all systems.
  - Data Logging: An SD card module logs all sensor data at regular intervals.
  -  LoRa Telemetry : A LoRa module allows low-power long-range transmission of sensor data to a base station when above water.

![image](https://github.com/user-attachments/assets/8b0ef004-06f5-44e4-8712-47ebffff0388)


## Sensors and Measured Parameters
  - Temperature Sensor (DS18B20)
      Measures water temperature, crucial for identifying coral bleaching risks.
  - pH Sensor
      Monitors acidity and detects shifts due to ocean acidification.
  - Electrical Conductivity (EC) Sensor
      Measures ion concentration to infer salinity and pollutant presence.
  - Dissolved Oxygen (DO) Sensor
      Indicates oxygen availability for marine life; important for spotting hypoxic events.
Each sensor is periodically sampled and saved with timestamps. The DO measurement uses two-point calibration and all readings are temperature-compensated where applicable.

![image](https://github.com/user-attachments/assets/29e59b3c-8f2a-4daf-bcdf-087c2272ddc7)


## Coral Health Classification (VGG-19)
A deep learning model classifies coral images into categories such as healthy or bleached. The model uses:
  - VGG-19 Architecture: Pre-trained on ImageNet and fine-tuned using labeled coral datasets
  - Image Preprocessing: Resizing, normalization, and augmentation for better generalization
  - Performance Metrics: Evaluated using accuracy, precision, recall, and F1-score

This model is designed to be integrated into the AUV or used for post-deployment analysis.

Software Components
  -  Embedded Firmware (Arduino)
        The coral_monitor.ino firmware reads sensor values and writes them to file.txt on the SD card. Serial output is available for debugging or real-time monitoring via UART.
  -  Machine Learning (Python)
        The capstone.ipynb notebook contains the full pipeline for training, evaluating, and running predictions using the VGG-19 model.
  - Data Logging
        Each log entry includes:
        -  Temperature (°C)
        - pH
        - EC (mS/cm)
        - Dissolved Oxygen (mg/L)

  - These values are saved on the SD card and can be parsed for analysis.

## How to Run
Arduino Setup
1. Connect the following:
  - DS18B20 → Digital Pin 2
  - pH Sensor → A0
  - EC Sensor → A1
  - DO Sensor → A2
  - SD Card → SPI Pins (CS on D53)

2. Upload the Arduino code using the Arduino IDE.

3. Insert an SD card. Logged data will be saved to file.txt.

3. Open the serial monitor at 9600 baud to verify sensor values.

Python Classifier (Coral Image Analysis)
1. Clone this repository and navigate to the project folder:

**git clone https://github.com/your-username/my-projects.git
cd "my-projects/Design and Implementation of Instrumentation for Monitoring Coral Reefs"**

2. Install the required packages:

**pip install -r requirements.txt**

3. Launch Jupyter Notebook and open capstone.ipynb
![image](https://github.com/user-attachments/assets/e1fba970-af8d-4837-afad-6ec55ae8236f)
