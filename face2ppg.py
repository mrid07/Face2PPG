# face2ppg.py
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import qr
from scipy.interpolate import interp1d

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_rppg(frames):
    roi_signals = []
    prev_roi = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            if prev_roi is not None:
                x, y, w, h = prev_roi
            else:
                continue
        else:
            x, y, w, h = faces[0]
            prev_roi = (x, y, w, h)

        forehead_h = int(h * 0.25)
        forehead_y = y + int(h * 0.1)
        roi = frame[forehead_y:forehead_y+forehead_h, x:x+w]

        if roi.size == 0:
            continue

        roi_mean = np.mean(roi, axis=(0, 1))
        roi_signals.append(roi_mean)

    if len(roi_signals) < len(frames):
        x_old = np.arange(len(roi_signals))
        x_new = np.linspace(0, len(roi_signals) - 1, len(frames))
        f = interp1d(x_old, roi_signals, axis=0, kind='linear', fill_value="extrapolate")
        roi_signals = f(x_new)

    return np.array(roi_signals)

def butter_bandpass_filter(signal, lowcut=0.75, highcut=4.0, fs=30, order=3):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    min_length = 3 * order + 1
    if len(signal) < min_length:
        order = max(1, int(len(signal) / 3))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def OMIT_transform(rgb_signal):
    if rgb_signal.size == 0 or len(rgb_signal) < 10:
        raise ValueError("Insufficient data for OMIT transform")
    Q, _ = qr(rgb_signal)
    return Q[:, 1]

def compute_heart_rate(signal, fs=30):
    if len(signal) < 100:
        return None
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    mask = (freqs >= 0.83) & (freqs <= 3.0)
    if not np.any(mask):
        return None
    max_freq = freqs[mask][np.argmax(psd[mask])]
    hr = max_freq * 60
    return hr if 40 < hr < 180 else None

def run_face2ppg(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fs = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    rgb_signals = extract_rppg(frames)
    if rgb_signals.size == 0 or len(rgb_signals) < 30:
        raise ValueError("Insufficient data (min 1 second at 30 FPS)")

    filtered_channels = np.zeros_like(rgb_signals)
    for i in range(3):
        channel = rgb_signals[:, i]
        filtered_channels[:, i] = butter_bandpass_filter(channel, fs=fs)

    rppg = OMIT_transform(filtered_channels)
    hr = compute_heart_rate(rppg, fs=fs)

    if hr:
        return np.linspace(0, len(rppg)/fs, len(rppg)), rppg, hr
    else:
        raise ValueError("Heart rate estimation failed")

    # if hr:
    #     return np.linspace(0, len(rppg)/fs, len(rppg)), rppg
    # else:
    #     raise ValueError("Heart rate estimation failed")
