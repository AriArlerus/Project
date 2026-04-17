/*
  ============================================================
  HC-SR04 + ESP32 DEVKITV1
  ส่งข้อมูลขึ้น Google Sheets ผ่าน WiFi
  ============================================================
  อ้างอิง: Khaleel et al., IJoST 9(1) 2024
  สมการ (8): distance = (duration * 0.034) / 2
  สมการ (9): error    = measured - desired

  การเดินสาย:
    HC-SR04 VCC  --> ESP32 5V (หรือ 3.3V)
    HC-SR04 GND  --> ESP32 GND
    HC-SR04 TRIG --> ESP32 GPIO 5
    HC-SR04 ECHO --> ESP32 GPIO 18  (ผ่าน Voltage Divider)

  คำสั่งผ่าน Serial Monitor (115200 baud):
    d:<ค่า>  ตั้งค่า Desired Distance  เช่น d:20.0
    s        เริ่มเก็บข้อมูลอัตโนมัติ
    p        หยุดชั่วคราว
    r        รีเซ็ต
    m        วัด 1 ครั้ง
  ============================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

// ──────────────────────────────────────────
// ตั้งค่า WiFi และ Google Sheets
// ──────────────────────────────────────────
const char* ssid       = "STAR_2.4G";
const char* password   = "Ari3025.";

String scriptURL   = "https://script.google.com/macros/s/AKfycby81YZ1N3qMHMBcCPavn14FWlIWcuMuyAoQuDu0rkB_MttxQBVyvURqqL_SVNgOAKeXUA/exec";
String sensorName  = "SensorData300cm";  

// ──────────────────────────────────────────
// ขา (Pins)
// ──────────────────────────────────────────
#define TRIG_PIN   5
#define ECHO_PIN   18
#define LED_PIN    2    // LED ในตัว ESP32

// ──────────────────────────────────────────
// ค่าคงที่จากเปเปอร์
// ──────────────────────────────────────────
#define SOUND_SPEED       0.034  
#define MIN_DISTANCE_CM   2.0
#define MAX_DISTANCE_CM   400.0
#define NUM_SAMPLES       431     // n_population ในเปเปอร์ (ตารางที่ 2)
#define NUM_AVG           5       // วัดเฉลี่ยกี่ครั้ง (ลด noise)
#define MEASURE_INTERVAL  5000    // ms ระหว่างการวัดแต่ละครั้ง

// ──────────────────────────────────────────
// ตัวแปร Global
// ──────────────────────────────────────────
int     sampleIndex     = 0;
float   desiredDistance = 0.0;
bool    collecting      = false;
bool    waitingDesired  = true;

unsigned long lastMeasureTime = 0;

// ──────────────────────────────────────────
// สมการ (8): วัดระยะทางจาก HC-SR04
// distance = (duration * SOUND_SPEED) / 2
// ──────────────────────────────────────────
float measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1.0;

  float distance = (duration * SOUND_SPEED) / 2.0;   // สมการ (😎

  if (distance < MIN_DISTANCE_CM || distance > MAX_DISTANCE_CM)
    return -1.0;

  return distance;
}

// ──────────────────────────────────────────
// วัดหลายครั้งแล้วเฉลี่ย (ลด noise)
// ──────────────────────────────────────────
float measureDistanceAvg() {
  float sum   = 0.0;
  int   count = 0;
  for (int i = 0; i < NUM_AVG; i++) {
    float d = measureDistance();
    if (d > 0) { sum += d; count++; }
    delay(10);
  }
  return (count == 0) ? -1.0 : sum / count;
}

// ──────────────────────────────────────────
// สมการ (9): คำนวณ Error
// error = measured - desired
// ──────────────────────────────────────────
float computeError(float measured, float desired) {
  return measured - desired;    // สมการ (9)
}

// ──────────────────────────────────────────
// ส่งข้อมูลขึ้น Google Sheets
// URL: ...?sensor=HC4&distance=20.03&desired=20.00&error=0.03&index=1
// ──────────────────────────────────────────
void sendToGoogleSheets(int index, float measured, float desired, float error) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  WiFiClientSecure client;
  client.setInsecure(); // สำคัญมาก: เพื่อให้ ESP32 ยอมรับการเชื่อมต่อ HTTPS ของ Google

  String url = scriptURL 
    + "?sensor="   + sensorName 
    + "&index="    + String(index) 
    + "&distance=" + String(measured, 4) 
    + "&desired="  + String(desired,  4) 
    + "&error="    + String(error,    6);

  // เริ่มการเชื่อมต่อโดยส่ง client เข้าไปด้วย
  if (http.begin(client, url)) { 
    // สั่งให้ตาม Redirection (302) ไปยัง URL ใหม่
    http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS); 
    
    int httpCode = http.GET();
    Serial.print("HTTP Code: "); Serial.println(httpCode);
    
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
  Serial.println("  HC-SR04 --> Google Sheets  |  ESP32 DEVKITV1");
  Serial.println("  Khaleel et al., IJoST 9(1) 2024");
  Serial.println("============================================================");
  Serial.println("คำสั่ง:");
  Serial.println("  d:<ค่า>  ตั้ง Desired Distance  เช่น d:20.0");
  Serial.println("  s        เริ่มเก็บข้อมูล");
  Serial.println("  p        หยุดชั่วคราว");
  Serial.println("  r        รีเซ็ต");
  Serial.println("  m        วัด 1 ครั้ง");
  Serial.println("============================================================");
  Serial.println("Output CSV (Serial + Google Sheets):");
  Serial.println("  index, measured_cm, desired_cm, error_cm");
  Serial.println("============================================================");
}

// ──────────────────────────────────────────
// แสดง CSV Header
// ──────────────────────────────────────────
void printCSVHeader() {
  Serial.println();
  Serial.println("# ---- เริ่มเก็บข้อมูล ----");
  Serial.print("# Sensor        = "); Serial.println(sensorName);
  Serial.print("# Desired       = "); Serial.print(desiredDistance, 4); Serial.println(" cm");
  Serial.print("# Target samples= "); Serial.println(NUM_SAMPLES);
  Serial.println("# สมการ (8): distance = (Time * 0.034) / 2");
  Serial.println("# สมการ (9): error    = measured - desired");
  Serial.println("#");
  Serial.println("index,measured_cm,desired_cm,error_cm");
}

// ──────────────────────────────────────────
// รีเซ็ต
// ──────────────────────────────────────────
void resetSystem() {
  sampleIndex     = 0;
  desiredDistance = 0.0;
  collecting      = false;
  waitingDesired  = true;
  Serial.println("\n[RESET] รีเซ็ตแล้ว");
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
      waitingDesired  = false;
      Serial.print("[OK] Desired Distance = ");
      Serial.print(desiredDistance, 4);
      Serial.println(" cm  |  พิมพ์ 's' เพื่อเริ่ม");
    } else {
      Serial.println("[ERROR] ระยะทางต้องอยู่ในช่วง 2 - 400 ซม.");
    }
  }

  // s — เริ่มเก็บ
  else if (cmd == "s" || cmd == "S") {
    if (waitingDesired) {
      Serial.println("[ERROR] ตั้งค่า Desired ก่อน (d:<ค่า>)");
    } else {
      collecting = true;
      if (sampleIndex == 0) printCSVHeader();
      Serial.println("[START] เริ่มเก็บข้อมูล...");
    }
  }

  // p — หยุดชั่วคราว
  else if (cmd == "p" || cmd == "P") {
    collecting = false;
    Serial.print("[PAUSE] หยุดชั่วคราว — เก็บได้ ");
    Serial.print(sampleIndex);
    Serial.print(" / ");
    Serial.print(NUM_SAMPLES);
    Serial.println("  |  พิมพ์ 's' เพื่อเริ่มต่อ");
  }

  // r — รีเซ็ต
  else if (cmd == "r" || cmd == "R") {
    resetSystem();
  }

  // m — วัด 1 ครั้ง
  else if (cmd == "m" || cmd == "M") {
    Serial.println("[SINGLE] กำลังวัด...");
    float dist = measureDistanceAvg();

    if (dist > 0) {
      Serial.print("  Measured = ");
      Serial.print(dist, 4);
      Serial.print(" cm");

      if (!waitingDesired) {
        float err = computeError(dist, desiredDistance);

        Serial.print("  |  Desired = ");
        Serial.print(desiredDistance, 4);
        Serial.print(" cm  |  Error = ");
        Serial.print(err, 6);
        Serial.println(" cm");

        sampleIndex++;
        sendToGoogleSheets(sampleIndex, dist, desiredDistance, err);

      } else {
        Serial.println("  [WARN] ยังไม่ได้ตั้ง Desired Distance เลยยังไม่ส่งขึ้นชีต");
      }
    } else {
      Serial.println("  [ERROR] วัดไม่ได้ (out of range)");
    }
  }

  else {
    Serial.print("[?] ไม่รู้จักคำสั่ง: "); Serial.println(cmd);
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
  pinMode(LED_PIN,  OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(LED_PIN,  LOW);

  // เชื่อมต่อ WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));  // กะพริบระหว่างเชื่อมต่อ
  }
  digitalWrite(LED_PIN, HIGH);
  Serial.println("\nWiFi Connected — IP: " + WiFi.localIP().toString());

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

  // เก็บข้อมูลอัตโนมัติ
  if (collecting && sampleIndex < NUM_SAMPLES) {
    unsigned long now = millis();

    if (now - lastMeasureTime >= MEASURE_INTERVAL) {
      lastMeasureTime = now;

      digitalWrite(LED_PIN, HIGH);
      float measured = measureDistanceAvg();
      digitalWrite(LED_PIN, LOW);

      if (measured > 0) {
        sampleIndex++;
        float error = computeError(measured, desiredDistance);   // สมการ (9)

        // แสดงทาง Serial (CSV)
        Serial.print(sampleIndex);   Serial.print(",");
        Serial.print(measured, 4);   Serial.print(",");
        Serial.print(desiredDistance, 4); Serial.print(",");
        Serial.println(error, 6);

        // ส่งขึ้น Google Sheets
        sendToGoogleSheets(sampleIndex, measured, desiredDistance, error);

      } else {
        Serial.print("# [WARN] sample ");
        Serial.print(sampleIndex + 1);
        Serial.println(" — out of range, ข้าม");
      }

      // เก็บครบ
      if (sampleIndex >= NUM_SAMPLES) {
        collecting = false;
        Serial.println("#");
        Serial.println("# ---- เก็บข้อมูลครบแล้ว ----");
        Serial.print("# จำนวนตัวอย่าง = "); Serial.println(NUM_SAMPLES);
        Serial.print("# Desired        = "); Serial.print(desiredDistance, 4); Serial.println(" cm");
        Serial.println("# ดูข้อมูลใน Google Sheets ได้เลย");
        Serial.println("# พิมพ์ 'r' เพื่อเริ่มใหม่");
        digitalWrite(LED_PIN, HIGH);   // LED ค้างแสดงว่าเสร็จแล้ว
      }
    }
  }
}