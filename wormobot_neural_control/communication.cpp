#include "communication.h"

void checkHandshake()
{
    Serial.println("Waiting for PARAMS and START command...");

    bool params_received = false;

    while (true) {
        if (Serial.available()) {
            String command = Serial.readStringUntil('\n');
            command.trim();

            if (command.startsWith("PARAMS")) {
                float tau_x_in, w_in, w_plus_in;

                int firstSpace = command.indexOf(' ');
                int secondSpace = command.indexOf(' ', firstSpace + 1);
                int thirdSpace = command.indexOf(' ', secondSpace + 1);

                if (firstSpace > 0 && secondSpace > 0 && thirdSpace > 0) {
                    String s1 = command.substring(firstSpace + 1, secondSpace);
                    String s2 = command.substring(secondSpace + 1, thirdSpace);
                    String s3 = command.substring(thirdSpace + 1);

                    tau_x_in = s1.toFloat();
                    w_in = s2.toFloat();
                    w_plus_in = s3.toFloat();

                    tau_x = tau_x_in;
                    w = w_in;
                    w_plus = w_plus_in;
                    w_minus = -w_plus_in;

                    Serial.println("Parameters received:");
                    Serial.print("tau_x = ");
                    Serial.println(tau_x, 6);
                    Serial.print("w = ");
                    Serial.println(w, 6);
                    Serial.print("w_plus = ");
                    Serial.println(w_plus, 6);
                    Serial.print("w_minus = ");
                    Serial.println(w_minus, 6);

                    params_received = true;
                } else {
                    Serial.println("ERROR: Invalid PARAMS format. Use: PARAMS tau_x w w_plus");
                }
            }
            else if (command == "START") {
                if (params_received) {
                    Serial.println("Starting program...");
                    break;
                } else {
                    Serial.println("ERROR: START received before PARAMS");
                }
            }
        }
    }
}

void checkStopMessage()
{
    if (Serial.available()) 
    {
        String command = Serial.readStringUntil('\n');
        command.trim();

        if (command == "STOP") 
        {
            Serial.println("Python leállt! Arduino újraindul...");
            delay(1000);
            setup();
        }
    }
}
