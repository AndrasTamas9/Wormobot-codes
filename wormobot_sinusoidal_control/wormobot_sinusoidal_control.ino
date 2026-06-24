#include <Dynamixel2Arduino.h>
#include "communication.h"

const uint8_t DXL_IDs[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
const float DXL_PROTOCOL_VERSION = 2.0;
const int DXL_BAUDRATE = 1000000;
#define N 9
#define pi 3.14159

Dynamixel2Arduino dxl(Serial1, -1);
using namespace ControlTableItem;

unsigned long start_time = 0;
double amplitude = pi / 6.0;
double joint_position[N], joint_position_degrees[N];

double radianstodegrees(double radian) {
  return (radian * 180.0) / pi;
}




void setupMotors() {
  dxl.begin(DXL_BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  for (int i = 0; i < N; i++) {
    dxl.torqueOff(DXL_IDs[i]);
    delay(50);
    dxl.setOperatingMode(DXL_IDs[i], OP_POSITION);
    delay(50);
   // dxl.writeControlTableItem(RETURN_DELAY_TIME, DXL_IDs[i], 100);  // 100 × 2µs = 200 µs
    delay(10);  // optional: short delay after writing EEPROM
    dxl.torqueOn(DXL_IDs[i]);
    delay(50);
  }

  Serial.println("Dynamixel port initialized.");
  Serial.println("Protocol version set.");
}

void runMotionLoop() {
  start_time = millis();

  while (true) {
    checkStopMessage();

    unsigned long elapsed_time = millis() - start_time;
    unsigned long t_start = micros();

    Serial.print(elapsed_time);
    Serial.print("\t");

    static int measure_counter = 0;
    measure_counter++;
    // nem minden ciklusban kuldjem el
    if (measure_counter % 1 == 0) {
      for (int i = 0; i < N; i++) {
        double measured = dxl.getPresentPosition(DXL_IDs[i], UNIT_DEGREE);
        Serial.print(measured, 3);
        Serial.print("\t");
      }
}
    for (int i = 0; i < N; i++) {
      joint_position[i] = amplitude * sin(2 * pi * frequency * elapsed_time * 0.001 - (2 * pi) / lambda * (N - i - 1) / N);
      joint_position_degrees[i] = radianstodegrees(joint_position[i]);
      double pos = 180 + joint_position_degrees[i];
      dxl.setGoalPosition(DXL_IDs[i], pos, UNIT_DEGREE);
      delayMicroseconds(1000);
    }

    unsigned long t_end = micros();
    Serial.println(t_end - t_start);     // loop time in µs

    if (elapsed_time > 100000) {
      for (int i = 0; i < N; i++) {
        dxl.torqueOff(DXL_IDs[i]);
      }
      Serial.println("Program befejeződött.");
      delay(1000);
      break;
    }
  }
}




void setup() {
  Serial.begin(115200);
  while (!Serial);

  while (true) {
    Serial.println("Waiting for parameters and START command...");
    checkHandshake();       // wait for SET + START
    Serial.println("Starting program...");
    setupMotors();          // init Dynamixel
    runMotionLoop();        // do motion
    // then loop back again to checkHandshake()
  }
}

void loop() {
  // unused
}
