#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <Arduino.h>

// Ezeket a .ino-ban definiáljuk majd
extern float tau_x;
extern float w;
extern float w_plus;
extern float w_minus;

void checkHandshake();
void checkStopMessage();

#endif
