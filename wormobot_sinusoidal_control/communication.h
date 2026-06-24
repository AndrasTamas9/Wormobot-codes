#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <Arduino.h>

void checkHandshake();
void checkStopMessage();

// Globális változók
extern double frequency;
extern double lambda;

#endif
