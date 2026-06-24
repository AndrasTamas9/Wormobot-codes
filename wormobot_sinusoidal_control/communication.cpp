#include "communication.h"

double frequency = 0.8;
double lambda = 0.8;

void checkHandshake()
{
    Serial.println("Waiting for parameters and START command...");

    while (true) {
        if (Serial.available()) {
            String command = Serial.readStringUntil('\n');
            command.trim();

            if (command.startsWith("SET frequency ")) {
                frequency = command.substring(14).toFloat();
                Serial.println("Frequency set to: " + String(frequency));
            } else if (command.startsWith("SET lambda ")) {
                lambda = command.substring(11).toFloat();
                Serial.println("Lambda set to: " + String(lambda));
            } else if (command == "START") {
                Serial.println("Starting program...");
                break;
            }
        }
    }
}

void checkStopMessage()
{
    if (Serial.available()) 
    {
        String command = Serial.readStringUntil('\n');
      
        if (command == "STOP") 
        {
            Serial.println("Python leállt! Arduino újraindul...");
            delay(1000);
            setup();  // HARDVER RESET
        }
    }
}
