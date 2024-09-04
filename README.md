# Nepali-Food-Recognition-System

## Overview
This project leverages **Mask R-CNN** to detect food items from images and calculate their calorie values. The system automates the process of tracking daily calorie intake, featuring an interactive web interface for user engagement. It uses **ESP32-CAM** for image capture, **NodeJS** for server-side operations, and **ReactJS** for the front-end. Calorie values are calculated through a **Nutrition API**, and users can track their intake history and set personalized calorie targets.

## Features
- Automated food item detection using Mask R-CNN.
- Calorie calculation based on detected food items.
- **ESP32-CAM** for capturing food images.
- User interface created with **ReactJS** for displaying calorie info, intake history, and daily goals.
- **NodeJS** server to process and store data.
- Integration with a **Nutrition API** for calorie data retrieval.

## Technologies Used
- **ESP32-CAM AI Thinker**: Captures top-view images of food items.
- **Mask R-CNN**: Segments and identifies food in images.
- **NodeJS**: Backend server to handle requests and image processing.
- **ReactJS**: Frontend for a user-friendly interface.
- **Nutrition API**: Provides standard calorie values based on food weight.
- **Arduino Uno**: Powers the ESP32-CAM module.
  
## Working Principle
1. **Image Capture**: The ESP32-CAM captures an image of the food item from a top-down view and sends it to the server.
2. **Image Processing**: The image is stored and processed by the Mask R-CNN model, which detects and identifies the food items.
3. **Weight Estimation**: The weight of the food is estimated based on its physical properties and density.
4. **Calorie Calculation**: Using the estimated weight, the Nutrition API calculates the calorie values.
5. **User Interface**: The result is displayed on a web page, where users can view their intake history and calorie goals.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/food-calorie-estimation.git
   ```
2. Install the required dependencies for the server:
   ```bash
   cd server
   npm install
   ```
3. Install the ReactJS dependencies for the frontend:
   ```bash
   cd client
   npm install
   ```
4. Run the NodeJS server:
   ```bash
   cd server
   node app.js
   ```
5. Run the ReactJS frontend:
   ```bash
   cd client
   npm start
   ```

## Usage
1. Open the web interface in your browser.
2. Enter your credentials (name, age, calorie goal).
3. Capture the image of the food using the ESP32-CAM.
4. View the estimated calories and your intake history on the interface.

## Applications
- Personal calorie tracking and diet monitoring.
- Research studies on eating habits and portion control.
- Restaurants should optimize food portions and promote healthy eating.

## License
This project is licensed under the MIT License.
