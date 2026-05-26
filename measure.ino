/*
  ============================================================
  HC-SR04 + ESP32 DEVKITV1
  ส่งข้อมูลขึ้น Google Sheets ผ่าน WiFi
  + แสดงค่า Measured และ Optimized แบบ Realtime
  ============================================================

  คำสั่งผ่าน Serial Monitor (115200 baud):
    d:<ค่า>  ตั้งค่า Desired Distance  เช่น d:20.0
    s        เริ่มเก็บข้อมูลอัตโนมัติ
    p        หยุดชั่วคราว
    r        รีเซ็ต
    m        วัด 1 ครั้ง
    rt       เริ่มวัดแบบ realtime
    x        หยุด realtime
  ============================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "secrets.h"

// ──────────────────────────────────────────
// ตั้งค่า WiFi และ Google Sheets
// ──────────────────────────────────────────
const char* ssid       = WIFI_SSID;
const char* password   = WIFI_PASSWORD;

String scriptURL   = GOOGLE_SCRIPT_URL;
String sensorName  = "SensorData";

// ──────────────────────────────────────────
// ขา Pins
// ──────────────────────────────────────────
#define TRIG_PIN   5
#define ECHO_PIN   18
#define LED_PIN    2

// ──────────────────────────────────────────
// ค่าคงที่สำหรับ HC-SR04
// ──────────────────────────────────────────
#define SOUND_SPEED       0.034
#define MIN_DISTANCE_CM   2.0
#define MAX_DISTANCE_CM   400.0
#define NUM_SAMPLES       1
#define NUM_AVG           5
#define MEASURE_INTERVAL  5000
#define REALTIME_INTERVAL 500

// ──────────────────────────────────────────
// Hardcode coefficients จาก Optimization
// D_optimized = a*M^2 + b*M + c
// เปลี่ยนค่า a, b, c ตามผลลัพธ์ที่เลือกใช้
// ──────────────────────────────────────────
const float CAL_A = -5.059576e-05;
const float CAL_B = 1.045771;
const float CAL_C = 2.919840;

// ──────────────────────────────────────────
// ตัวแปร Global
// ──────────────────────────────────────────
int     sampleIndex     = 0;
float   desiredDistance = 0.0;
bool    collecting      = false;
bool    waitingDesired  = true;
bool    realtimeMode    = false;

unsigned long lastMeasureTime  = 0;
unsigned long lastRealtimeTime = 0;

// ──────────────────────────────────────────
// วัดระยะจาก HC-SR04
// distance = duration * 0.034 / 2
// ──────────────────────────────────────────
float measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return -1.0;
  }

  float distance = (duration * SOUND_SPEED) / 2.0;

  if (distance < MIN_DISTANCE_CM || distance > MAX_DISTANCE_CM) {
    return -1.0;
  }

  return distance;
}

// ──────────────────────────────────────────
// วัดหลายครั้งแล้วเฉลี่ย เพื่อลด noise
// ──────────────────────────────────────────
float measureDistanceAvg() {
  float sum = 0.0;
  int count = 0;

  for (int i = 0; i < NUM_AVG; i++) {
    float d = measureDistance();

    if (d > 0) {
      sum += d;
      count++;
    }

    delay(10);
  }

  if (count == 0) {
    return -1.0;
  }

  return sum / count;
}

// ──────────────────────────────────────────
// Error before optimization
// error = measured - desired
// ──────────────────────────────────────────
float computeError(float measured, float desired) {
  return measured - desired;
}

// ──────────────────────────────────────────
// คำนวณระยะหลัง Optimization
// D_optimized = a*M^2 + b*M + c
// ──────────────────────────────────────────
float computeOptimizedDistance(float measured) {
  return CAL_A * measured * measured + CAL_B * measured + CAL_C;
}

// ──────────────────────────────────────────
// ส่งข้อมูลขึ้น Google Sheets
// ตอนนี้ยังส่ง measured, desired, error_before แบบเดิม
// ──────────────────────────────────────────
void sendToGoogleSheets(int index, float measured, float desired, float error) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  HTTPClient http;
  WiFiClientSecure client;
  client.setInsecure();

  String url = scriptURL
    + "?sensor="   + sensorName
    + "&index="    + String(index)
    + "&distance=" + String(measured, 4)
    + "&desired="  + String(desired, 4)
    + "&error="    + String(error, 6);

  if (http.begin(client, url)) {
    http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);

    int httpCode = http.GET();

    Serial.print("HTTP Code: ");
    Serial.println(httpCode);

    if (httpCode == 200) {
      Serial.println("Success: Data saved to Google Sheets!");
    }

    http.end();
  }
}

// ──────────────────────────────────────────
// แสดงเมนู
// ──────────────────────────────────────────
void printMenu() {
  Serial.println();
  Serial.println("============================================================");
  Serial.println("  HC-SR04 + ESP32 DEVKITV1");
  Serial.println("  Measured + Optimized Distance");
  Serial.println("============================================================");
  Serial.println("คำสั่ง:");
  Serial.println("  d:<ค่า>  ตั้ง Desired Distance เช่น d:20.0");
  Serial.println("  s        เริ่มเก็บข้อมูลอัตโนมัติ");
  Serial.println("  p        หยุดชั่วคราว");
  Serial.println("  r        รีเซ็ต");
  Serial.println("  m        วัด 1 ครั้ง");
  Serial.println("  rt       เริ่มวัดแบบ realtime");
  Serial.println("  x        หยุด realtime");
  Serial.println("============================================================");
  Serial.println("Output CSV:");
  Serial.println("  index, measured_cm, optimized_cm, desired_cm, error_before_cm, error_after_cm");
  Serial.println("============================================================");
}

// ──────────────────────────────────────────
// แสดง CSV Header
// ──────────────────────────────────────────
void printCSVHeader() {
  Serial.println();
  Serial.println("# ---- เริ่มเก็บข้อมูล ----");
  Serial.print("# Sensor        = ");
  Serial.println(sensorName);

  Serial.print("# Desired       = ");
  Serial.print(desiredDistance, 4);
  Serial.println(" cm");

  Serial.print("# Target samples= ");
  Serial.println(NUM_SAMPLES);

  Serial.println("# D_optimized = a*M^2 + b*M + c");
  Serial.println("#");
  Serial.println("index,measured_cm,optimized_cm,desired_cm,error_before_cm,error_after_cm");
}

// ──────────────────────────────────────────
// รีเซ็ต
// ──────────────────────────────────────────
void resetSystem() {
  sampleIndex     = 0;
  desiredDistance = 0.0;
  collecting      = false;
  waitingDesired  = true;
  realtimeMode    = false;

  Serial.println();
  Serial.println("[RESET] รีเซ็ตแล้ว");
  printMenu();
}

// ──────────────────────────────────────────
// ประมวลผลคำสั่ง Serial
// ──────────────────────────────────────────
void processCommand(String cmd) {
  cmd.trim();

  // d:<ค่า> — ตั้ง Desired Distance
  if (cmd.startsWith("d:") || cmd.startsWith("D:")) {
    float val = cmd.substring(2).toFloat();

    if (val >= MIN_DISTANCE_CM && val <= MAX_DISTANCE_CM) {
      desiredDistance = val;
      waitingDesired = false;

      Serial.print("[OK] Desired Distance = ");
      Serial.print(desiredDistance, 4);
      Serial.println(" cm");
    } else {
      Serial.println("[ERROR] ระยะทางต้องอยู่ในช่วง 2 - 400 cm");
    }
  }

  // s — เริ่มเก็บข้อมูลอัตโนมัติ
  else if (cmd == "s" || cmd == "S") {
    if (waitingDesired) {
      Serial.println("[ERROR] ตั้งค่า Desired ก่อน เช่น d:20.0");
    } else {
      realtimeMode = false;
      collecting = true;

      if (sampleIndex == 0) {
        printCSVHeader();
      }

      Serial.println("[START] เริ่มเก็บข้อมูล...");
    }
  }

  // p — หยุดชั่วคราว
  else if (cmd == "p" || cmd == "P") {
    collecting = false;

    Serial.print("[PAUSE] หยุดชั่วคราว — เก็บได้ ");
    Serial.print(sampleIndex);
    Serial.print(" / ");
    Serial.println(NUM_SAMPLES);
  }

  // r — รีเซ็ต
  else if (cmd == "r" || cmd == "R") {
    resetSystem();
  }

  // m — วัด 1 ครั้ง
  else if (cmd == "m" || cmd == "M") {
    Serial.println("[SINGLE] กำลังวัด...");

    float measured = measureDistanceAvg();

    if (measured > 0) {
      float optimized = computeOptimizedDistance(measured);

      Serial.print("Measured = ");
      Serial.print(measured, 4);
      Serial.print(" cm");

      Serial.print("  |  Optimized = ");
      Serial.print(optimized, 4);
      Serial.print(" cm");

      if (!waitingDesired) {
        float errorBefore = computeError(measured, desiredDistance);
        float errorAfter  = optimized - desiredDistance;

        Serial.print("  |  Desired = ");
        Serial.print(desiredDistance, 4);
        Serial.print(" cm");

        Serial.print("  |  Error Before = ");
        Serial.print(errorBefore, 6);
        Serial.print(" cm");

        Serial.print("  |  Error After = ");
        Serial.print(errorAfter, 6);
        Serial.print(" cm");

        sampleIndex++;

        sendToGoogleSheets(sampleIndex, measured, desiredDistance, errorBefore);
      } else {
        Serial.print("  |  [WARN] ยังไม่ได้ตั้ง Desired");
      }

      Serial.println();
    } else {
      Serial.println("[ERROR] วัดไม่ได้ หรืออยู่นอกช่วง");
    }
  }

  // rt — เริ่ม realtime
  else if (cmd == "rt" || cmd == "RT") {
    realtimeMode = true;
    collecting = false;

    Serial.println();
    Serial.println("[REALTIME] เริ่มวัดแบบ realtime");
    Serial.println("measured_cm,optimized_cm");
  }

  // x — หยุด realtime
  else if (cmd == "x" || cmd == "X") {
    realtimeMode = false;
    Serial.println("[REALTIME] หยุด realtime");
  }

  else {
    Serial.print("[?] ไม่รู้จักคำสั่ง: ");
    Serial.println(cmd);
  }
}

// ──────────────────────────────────────────
// setup()
// ──────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  digitalWrite(LED_PIN, HIGH);

  Serial.println();
  Serial.println("WiFi Connected — IP: " + WiFi.localIP().toString());

  printMenu();
}

// ──────────────────────────────────────────
// loop()
// ──────────────────────────────────────────
void loop() {
  // รับคำสั่งจาก Serial
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }

  // ────────────────────────────────────────
  // โหมด realtime
  // วัดแล้วแสดง Measured + Optimized ต่อเนื่อง
  // ────────────────────────────────────────
  if (realtimeMode) {
    unsigned long now = millis();

    if (now - lastRealtimeTime >= REALTIME_INTERVAL) {
      lastRealtimeTime = now;

      float measured = measureDistanceAvg();

      if (measured > 0) {
        float optimized = computeOptimizedDistance(measured);

        Serial.print("Measured = ");
        Serial.print(measured, 4);
        Serial.print(" cm");

        Serial.print("  |  Optimized = ");
        Serial.print(optimized, 4);
        Serial.println(" cm");
      } else {
        Serial.println("[WARN] วัดไม่ได้ หรืออยู่นอกช่วง");
      }
    }
  }

  // ────────────────────────────────────────
  // โหมดเก็บข้อมูลอัตโนมัติ
  // ────────────────────────────────────────
  if (collecting && sampleIndex < NUM_SAMPLES) {
    unsigned long now = millis();

    if (now - lastMeasureTime >= MEASURE_INTERVAL) {
      lastMeasureTime = now;

      digitalWrite(LED_PIN, HIGH);
      float measured = measureDistanceAvg();
      digitalWrite(LED_PIN, LOW);

      if (measured > 0) {
        sampleIndex++;

        float optimized   = computeOptimizedDistance(measured);
        float errorBefore = computeError(measured, desiredDistance);
        float errorAfter  = optimized - desiredDistance;

        Serial.print(sampleIndex);
        Serial.print(",");

        Serial.print(measured, 4);
        Serial.print(",");

        Serial.print(optimized, 4);
        Serial.print(",");

        Serial.print(desiredDistance, 4);
        Serial.print(",");

        Serial.print(errorBefore, 6);
        Serial.print(",");

        Serial.println(errorAfter, 6);

        sendToGoogleSheets(sampleIndex, measured, desiredDistance, errorBefore);
      } else {
        Serial.print("# [WARN] sample ");
        Serial.print(sampleIndex + 1);
        Serial.println(" — out of range, ข้าม");
      }

      if (sampleIndex >= NUM_SAMPLES) {
        collecting = false;

        Serial.println("#");
        Serial.println("# ---- เก็บข้อมูลครบแล้ว ----");

        Serial.print("# จำนวนตัวอย่าง = ");
        Serial.println(NUM_SAMPLES);

        Serial.print("# Desired        = ");
        Serial.print(desiredDistance, 4);
        Serial.println(" cm");

        Serial.println("# ดูข้อมูลใน Google Sheets ได้เลย");
        Serial.println("# พิมพ์ 'r' เพื่อเริ่มใหม่");

        digitalWrite(LED_PIN, HIGH);
      }
    }
  }
}