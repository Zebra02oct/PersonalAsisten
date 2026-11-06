# Personal Assistant Dashboard

A Python-based personal assistant dashboard with face recognition login system.

## Features

- **Face Recognition Login** - Biometric authentication using face detection
- **Eye Biometric Scanning** - Futuristic sci-fi scanning effects
- **Age Detection** - Calculate age from date of birth
- **Multi-sample Enrollment** - 5 samples for better accuracy
- **MySQL Database** - Secure local storage of face signatures

## Quick Start

### Prerequisites
- Python 3.10+
- Webcam
- MySQL (via Laragon)

### Setup

```bash
cd asisten
.venv\Scripts\activate
pip install -r requirements.txt
```

### Database Setup

Make sure Laragon MySQL is running, then create the database:

```sql
CREATE DATABASE asisten_db;
```

Tables will be created automatically on first run.

### Run Login System

```bash
cd asisten
.venv\Scripts\activate
python src/main.py
```

## Usage

### First Time Setup (Enrollment)

1. Run `python src/main.py`
2. Select `[2] Enroll new face`
3. Enter your name and date of birth (YYYY-MM-DD)
4. Look at the camera and press SPACE for each sample
5. Capture 5 samples for best accuracy

### Login

1. Run `python src/main.py`
2. Select `[1] Login with face recognition`
3. Look at the camera - system will automatically:
   - Detect face (Stage 1)
   - Scan eye biometrics (Stage 2)
   - Verify signature (Stage 3)
4. Welcome message with your name and age appears!

### Reset Enrollment

Select `[3] Reset enrollment` to clear all data and start over.

## Project Structure

```
src/
├── main.py                     # Main launcher
├── face_detection/            # Basic face detection (Haar Cascade)
├── face_recognition/         # Face recognition modules
│   ├── landmark_detector.py  # MediaPipe 468 landmarks
│   ├── feature_extractor.py  # 32 geometric features
│   └── ...
├── face_analysis/            # Eye analysis
│   ├── eye_shape_classifier.py
│   ├── biometric_scanner.py  # Sci-fi scanning effects
│   └── demo.py
├── login_system/             # Face login
│   └── face_login.py         # Staged verification system
└── database/                 # MySQL integration
    ├── connection.py
    ├── models.py
    └── owner_profile.py
```

## Staged Verification System

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Face Detection                              │
│  - Detect face in frame                               │
│  - Validate face quality                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Eye Biometric Scan                         │
│  - Spiral scanner animation                          │
│  - Eye shape classification                           │
│  - Iris pattern analysis                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Signature Verification                     │
│  - Extract 32 geometric features                     │
│  - Compare with enrolled signature                    │
│  - Calculate match confidence                         │
└─────────────────────────────────────────────────────────┘
```

## Data Stored

**NOT stored:**
- ❌ Face photos
- ❌ Video footage
- ❌ Raw biometric data

**Stored:**
- ✅ 32 geometric feature values (face signature)
- ✅ Eye shape classification
- ✅ Name and date of birth
- ✅ Age (calculated from DOB)

## Demo Scripts

| Script | Description |
|--------|-------------|
| `python src/face_detection/detector.py` | Basic face detection |
| `python src/face_analysis/demo.py` | Eye shape + biometric scanner |
| `python src/face_recognition/demo.py` | Face recognition with enrollment |
| `python src/main.py` | Complete face login system |

## Requirements

- opencv-python>=4.8.0
- numpy>=1.24.0
- mediapipe>=1.1.0
- pymysql>=1.1.0
- sqlalchemy>=2.0.0
- python-dotenv>=1.0.0

## Security

- Face signature (not images) stored in database
- Database credentials in `.env` file
- No external data transmission
- User can reset/delete all data anytime
