Smart Surveillance AI Documentation
This project uses Generative AI to detect unusual activities in video feeds in real-time. It was built for hackathon demos and combines computer vision, deep learning, and a simple web interface to improve security monitoring. Here’s everything you need to know to understand, set up, and contribute to it.
Table of Contents

Features
Demo
Installation
Usage
Deployment
Technologies Used
Contributing
License
Contact

Features

Real-Time Anomaly Detection: Uses a Convolutional Autoencoder (trained on UCSD Ped1 dataset) to spot unusual activities.
Multi-Camera Support: Watch multiple camera feeds or pre-recorded videos at once.
Gen-AI Insights: Gives basic reports and suggestions (e.g., alerts for high anomaly areas) from video analysis.
Interactive Dashboard: Has a cool illusion theme with gradients and an easy-to-use interface.
Data Analytics: Shows anomaly trends, distributions, and lets you export data to CSV.
User Authentication: Secure sign-up and login with a MongoDB backend.
Notification System: Option to add an email for anomaly alerts (not fully built yet).
Video Clips: Save and review clips of detected anomalies.

Demo

Live Monitoring: Start/stop video feeds, adjust anomaly settings, and see heatmaps live.
Analytics: Check trends in errors, anomaly patterns, and past data.
Project Info: Learn about the system, its uses, and future plans.
Check the live version (once deployed) at: [Insert Deployed URL Here].

Installation
What You Need

Python 3.8 or newer
MongoDB (local or online, e.g., MongoDB Atlas)

Steps

Clone the Repository:

Run: git clone https://github.com/your-username/Surveillance-Anomaly-GEN-AI.git
Go to the folder: cd Surveillance-Anomaly-GEN-AI


Set Up a Virtual Environment:

Create it: python -m venv venv
Activate it:
On Mac/Linux: source venv/bin/activate
On Windows: venv\Scripts\activate




Install Dependencies:

Run: pip install -r requirements.txt
Create a requirements.txt file with:streamlit==1.39.0
opencv-python-headless==4.10.0.84
torch==2.4.1
numpy==1.26.4
pandas==2.2.3
plotly==5.24.1
pymongo
python-decouple




Set Up MongoDB:

Local: Start MongoDB with mongod.
Remote: Sign up at MongoDB Atlas, create a cluster, and get your URI (e.g., mongodb+srv://:@/?retryWrites=true&w=majority).
Add the URI to a .env file: MONGO_URI=your_uri_here
Add .env to .gitignore to keep it safe.


Run the App:

Run: streamlit run dashboard.py
Open it at http://localhost:8501.



Usage

Register/Login: Use the sidebar to sign up or log in.
Live Monitoring:
Pick a source (Webcam or Video File).
Upload a video if using a file.
Select camera numbers and tweak the anomaly slider.
Click "Start Feed" to begin, "Stop Feed" to end.


View Reports: Check "Gen-AI Basic Reports" for summaries.
Analytics: Go to the "Analytics" tab for charts.
Project Info: Visit the "Project Info" tab for details.

Deployment

Push to GitHub:

Run: git add ., git commit -m "Ready for deployment", git push origin master


Deploy on a Platform:

Render: Create a Web Service, link your GitHub repo, add MONGO_URI in environment variables, and use start command: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0.
Heroku: Add a Procfile with web: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0, deploy via GitHub, and set MONGO_URI as an environment variable.
Test the deployed URL to confirm it works.



Technologies Used

Frontend: Streamlit
Backend: Python, MongoDB
AI/ML: PyTorch (Convolutional Autoencoder)
Computer Vision: OpenCV
Data Visualization: Plotly
Deployment: GitHub, Render/Heroku

Contributing

Fork the repo.
Create a branch: git checkout -b feature-name.
Make changes: git commit -m "Add feature-name".
Push it: git push origin feature-name.
Open a Pull Request.

License
This project uses the MIT License. See the LICENSE file for details.
Contact

Author: [Praveen kumar S]
Email: [praveensrinivasan05.com]
GitHub: https://github.com/your-praveen0767

Thanks for checking out the Smart Surveillance System! We’d love your feedback or help to improve it.
