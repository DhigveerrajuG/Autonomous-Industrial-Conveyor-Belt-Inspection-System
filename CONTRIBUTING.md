# Contributing to Mining Conveyor Belt Monitoring System

Thank you for your interest in contributing! Whether you are an engineering student, computer vision enthusiast, or industrial automation developer, your contributions are warmly welcomed.

---

## 🌟 How Can You Contribute?

You can contribute in many ways:
- **Code**: Implementing new features, fixing bugs, or optimizing detection pipelines.
- **Computer Vision**: Improving the dataset, adding synthetic data generators, or testing model weights.
- **Hardware & IoT**: Adding ESP32/Arduino sensor integrations.
- **Documentation**: Improving guides, adding docstrings, or writing tutorials.
- **UI / Dashboard**: Enhancing the real-time Sentinel dashboard.

Check out our [Issues tab](https://github.com/DhigveerrajuG/mining-conveyor-belt-monitoring-system/issues) with the **`good first issue`** and **`help wanted`** labels!

---

## 🛠️ Getting Started

### 1. Fork and Clone the Repository
```bash
git clone https://github.com/<your-username>/mining-conveyor-belt-monitoring-system.git
cd mining-conveyor-belt-monitoring-system
```

### 2. Set Up Your Python Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

---

## 🧪 Testing Your Changes

Before submitting your PR, ensure that:
1. All Python scripts compile without syntax errors:
   ```bash
   python -m py_compile yolo_sentinel_server.py train_perfect_conveyor_model.py train_from_user_images.py
   ```
2. The server initializes cleanly:
   ```bash
   python -c "import yolo_sentinel_server; detector = yolo_sentinel_server.ConveyorDetector()"
   ```
3. The dashboard UI loads properly at `http://localhost:8000/`.

---

## 📬 Submitting a Pull Request

1. Commit your changes with a clear message:
   ```bash
   git commit -m "Add: brief description of your feature"
   ```
2. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Open a Pull Request on GitHub and fill in the PR template.

---

## 📜 Code of Conduct
Please maintain a respectful, welcoming, and collaborative environment for all contributors.
