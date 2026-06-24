#include <Dynamixel2Arduino.h>
using namespace ControlTableItem;
#include "communication.h"



const uint8_t DXL_IDs[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

const float DXL_PROTOCOL_VERSION = 2.0;
const int DXL_BAUDRATE = 1000000;  // 1 Mbps Baudrate for motors

// OpenRB-150 doesn't require DIR_PIN, so it's set to -1
Dynamixel2Arduino dxl(Serial1, -1);  // Use Serial1 for OpenRB-150

const int N = 9;

////////////////////////////////////////////// Paraméterek

float tau_x = 1.0;
float w = 8.0;
float w_plus = 4.0;
float w_minus = -4.0;

const float a = 2.0;
const float b = 0.0;
const float q_min = 140.0;
const float q_max = 220.0;



////////////////////////////////////////////// Változók

float x[N] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};         // aktuális x állapotok
float q[N] = {180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0}; // aktuális szög állapotok
float s[N];         // s visszacsatolási értékek
float y[N];     // y - nemlineáris sigmoid érték
float q_star[N];          // kimeneti szögek (szervóknak)
float t = 0.0;      // teljes eltelt ido
float dt = 0.01;


////////////////////////////////////////////// Prototípusok
float sigmoid(float x);
void compute_s(const float q[], float s[]);
void compute_derivative(const float x[], const float q[], float dxdt[], float y[]);
void rk4_step();
void compute_q_star();




void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);  // Initialize USB serial for debugging
  while (!Serial) {
        ; // Vár a soros port megnyitására
    }

  checkHandshake();

  Serial.println("Starting setup...");

  // Initialize the Dynamixel port
  dxl.begin(DXL_BAUDRATE);  // Set baudrate for communication with motors
  Serial.println("Dynamixel port initialized.");

  // Set the protocol version
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  Serial.println("Protocol version set.");

  // Turn off torque to configure the motors
  for(int i=0; i<9; i++)
  {
    dxl.torqueOff(DXL_IDs[i]);
    delay(100);
  }

  // Set all motors to position control mode
  for(int i=0; i<9; i++)
  {
    dxl.setOperatingMode(DXL_IDs[i], OP_POSITION);
    delay(100);
  }

  // Turn on torque to enable movement
  for(int i=0; i<9; i++)
  {
    dxl.torqueOn(DXL_IDs[i]);
    delay(100);
  }

}

void loop() {
  // put your main code here, to run repeatedly:

  // 1. Időmérés kezdete (opcionális)
  unsigned long time1 = millis();

  checkStopMessage();

  // 2. Aktuális szervópozíciók beolvasása --> q[]
  for (int i = 0; i < N; ++i) {
    q[i] = dxl.getPresentPosition(DXL_IDs[i], UNIT_DEGREE);
  }

  // 3. Numerikus integráció: RK4 frissíti x[i]-t
  rk4_step(dt);

  // 4. Nemlinearitás + visszaskálázás --> q_star[i]
  compute_q_star();

  // 5. Szervók vezérlése az új q_star szögekkel
  for (int i = 0; i < N; ++i) {
    dxl.setGoalPosition(DXL_IDs[i], q_star[i], UNIT_DEGREE);
  }

  // 6. Soros kimenet állapotellenőrzéshez
  Serial.print(t, 2);
  Serial.print("  ");
  Serial.print(dt, 2);
  Serial.print("  ");
  for (int i = 0; i < N; ++i) {
    Serial.print(q[i], 1);
    Serial.print("  ");
  }
  for (int i = 0; i < N; ++i) {
    Serial.print(q_star[i], 1);
    Serial.print("  ");
  }
  for (int i = 0; i < N; ++i) {
    Serial.print(y[i], 2);
    Serial.print("  ");
  }
  Serial.println();

  // 7. Ciklusidő mérés
  unsigned long time2 = millis();
  dt = (time2 - time1) / 1000.0; // masodpercben

}


////////////////////////////////////////////// Függvények

// Sigmoid függvény
float sigmoid(float x) {
  return 1.0 / (1.0 + exp(a * (b - x)));
}


// Normalizált s kiszámítása
void compute_s(const float q[], float s[]) {
  float denom = q_max - q_min + 1e-6;
  for (int i = 0; i < N; ++i)
    s[i] = (q[i] - q_min) / denom;
}


// Derivált kiszámítása dxdt = f(x)
void compute_derivative(const float x[], const float q[], float dxdt[], float y[]) {
  float s[N];
  compute_s(q, s);

  for (int i = 0; i < N; ++i)
    y[i] = sigmoid(x[i]);

  for (int i = 0; i < N; ++i) {
    float s_i = s[i];

    // Két rákövetkező index, periodikusan értelmezve
    //float s_ip1 = s[(i + 1) % N] + s[(i + 2) % N];

    float s_ip1 = s[(i + 1) % N];

    // Két megelőző index, periodikusan értelmezve
    //float s_im1 = s[(i - 1 + N) % N] + s[(i - 2 + N) % N];

    float s_im1 = s[(i - 1 + N) % N];

    dxdt[i] = (-x[i] + w * (y[i] - s_i) + w_plus * s_ip1 + w_minus * s_im1) / tau_x;
}
}


// RK4 lépés
void rk4_step(float h) {
  float k1[N], k2[N], k3[N], k4[N];
  float xtemp[N];

  compute_derivative(x, q, k1, y);

  for (int i = 0; i < N; ++i)
    xtemp[i] = x[i] + 0.5 * h * k1[i];
  compute_derivative(xtemp, q, k2, y);

  for (int i = 0; i < N; ++i)
    xtemp[i] = x[i] + 0.5 * h * k2[i];
  compute_derivative(xtemp, q, k3, y);

  for (int i = 0; i < N; ++i)
    xtemp[i] = x[i] + h * k3[i];
  compute_derivative(xtemp, q, k4, y);

  for (int i = 0; i < N; ++i)
    x[i] += (h / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);

  t += h;
}


// Kimenet kiszámítása szervó számára
void compute_q_star() {
  for (int i = 0; i < N; ++i)
    q_star[i] = q_min + y[i] * (q_max - q_min);
}
