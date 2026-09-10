---
name: Hardware Sensor Integration
about: Propose or track integration of physical mining & conveyor sensors
title: '[HARDWARE] '
labels: 'hardware, enhancement'
assignees: ''
---

**Sensor Type**
- [ ] Belt Tension Sensor (Load cell / Strain gauge)
- [ ] Thickness Sensor (Ultrasonic / Optical laser displacement)
- [ ] Vibration Sensor (Accelerometer / MPU6050)
- [ ] Motor Temperature Sensor (Thermocouple / Infrared)

**Microcontroller / Edge Hardware**
- [ ] ESP32
- [ ] Arduino Uno / Nano
- [ ] Raspberry Pi / Jetson GPIO

**Protocol / Interface**
- [ ] Serial UART (USB COM Port)
- [ ] MQTT / Wi-Fi
- [ ] WebSocket / HTTP REST

**Implementation Details**
Describe how the microcontroller feeds data to `yolo_sentinel_server.py`.
