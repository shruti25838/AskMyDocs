#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include "DFRobot_EC10.h"
#include <SD.h>

#define ONE_WIRE_BUS 2
#define EC_PIN A1
#define DO_PIN A2
#define VREF 5000.0 // VREF (mV)
#define ADC_RES 1024 // ADC Resolution

#define TWO_POINT_CALIBRATION 1 // Set to 1 for two-point calibration
#define READ_TEMP (sensors.getTempCByIndex(0))
// Two-point calibration constants...
#define CAL1_V 1455.0 // mV - High temperature point
#define CAL1_T 25.0 // °C - High temperature point
#define CAL2_V 2099.0 // mV - Low temperature point
#define CAL2_T 15.0 // °C - Low temperature point

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
float temperature = 0, temp;
float calibration_value = 21.34 - 0.22;
int phval = 0;
unsigned long int avgval;
int buffer_arr[240]; //buffer size for 20mins of deployment
float ph_act;
float voltage,ecValue = 25;
int address = 0;
int16_t dissolved_oxygen;
DFRobot_EC10 ec;
const uint16_t DO_Table[41] = {
 14460, 14220, 13820, 13440, 13090, 12740, 12420, 12110, 11810, 11530,
 11260, 11010, 10770, 10530, 10300, 10080, 9860, 9660, 9460, 9270,
 9080, 8900, 8730, 8570, 8410, 8250, 8110, 7960, 7820, 7690,
 7560, 7430, 7300, 7180, 7070, 6950, 6840, 6730, 6630, 6530, 6410};
int CS = 53;
File file;
int outputPin1 = A0, outputPin2 = A1, outputPin3 = 2; //pH, EC, temp

void setup()
{
  Serial.begin(9600);
  while (!Serial)
  {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial1.begin(9600);
  sensors.begin();
  Wire.begin();  
  ec.begin();

  pinMode(CS, OUTPUT); // chip select pin is set as OUTPUT
 
  if (!SD.begin(CS))
  { // Initialize SD card
    Serial.println("Could not initialize SD card."); // if return value is false, something went wrong.
    Serial1.println("Could not initialize SD card.");
  }
  if (SD.exists("file.txt"))
  { // if "file.txt" exists, fill will be deleted
    Serial.println("File exists.");
    Serial1.println("File exists.");
    if (SD.remove("file.txt") == true)
    { //If file is removed, we print a message
      Serial.println("Successfully removed file.");
      Serial1.println("Successfully removed file.");
    }
    else
    {
      Serial.println("Could not remove file.");
      Serial1.println("Could not remove file.");
    }
  }
}

void loop()
{
      Serial.println("Sensor");
      temp = readTemp();  // read your temperature sensor to execute temperature compensation
      phval = readpH();
      ecValue = EC();
      dissolved_oxygen = DO();
      delay(3000);
      writeFile();
}

float readTemp()
{
  sensors.requestTemperatures();
  temp = sensors.getTempCByIndex(0);
  //Serial.println("Temperature:");
  //Serial.print(sensors.getTempCByIndex(0),2);
  //Serial.print("^C  ");
  Serial1.println("Temperature:");
  //temp = sensors.getTempCByIndex(0);
  Serial1.print(sensors.getTempCByIndex(0),2);
  Serial1.print("^C  ");
}

float readpH()
{
  for(int i=0;i<10;i++)
 {
 buffer_arr[i]=analogRead(A0);
 //delay(5000);
 }
 for(int i=0;i<9;i++)
 {
 for(int j=i+1;j<10;j++)
 {
 if(buffer_arr[i]>buffer_arr[j])
 {
 temp=buffer_arr[i];
 buffer_arr[i]=buffer_arr[j];
 buffer_arr[j]=temp;
 }
 }
 }
 avgval=0;
 for(int i=2;i<8;i++)
 avgval+=buffer_arr[i];
 float volt=(float)avgval*5.0/1024/6;
  ph_act = -5.70 * volt + calibration_value;

 //Serial.print("pH:");
 //Serial.print(ph_act);
 Serial1.print("pH:");
 Serial1.print(ph_act);
 //int val2 = ph_act/4;
 //delay(1000);
}

float EC()
{
  static unsigned long timepoint = millis();
    if(millis()-timepoint>1000U)  //time interval: 1s
    {
      timepoint = millis();
      voltage = analogRead(EC_PIN)/1024.0*5000;  // read the voltage
      //Serial.print(voltage);
      ecValue =  ec.readEC(voltage,temperature);  // convert voltage to EC with temperature compensation
      //Serial.print("  EC:");
      //Serial.print(ecValue,2);
      //Serial.print("ms/cm");
      Serial1.print("  EC:");
      Serial1.print(ecValue,2);
      Serial1.print("ms/cm");
     }
    ec.calibration(voltage,temperature);  // calibration process by Serail CMD
}

float DO()
{
  uint16_t ADC_Raw = analogRead(DO_PIN);
  float ADC_Voltage = VREF * ADC_Raw / ADC_RES;
  dissolved_oxygen = readDO(ADC_Voltage, (uint8_t)temp);
  //Serial.print("  Dissolved Oxygen:");
  //Serial.print(dissolved_oxygen);
  //Serial.println(" mg/L");
  Serial1.print("  Dissolved Oxygen:");
  Serial1.print(dissolved_oxygen);
  Serial1.print(" mg/L");
}

int16_t readDO(float voltage_mv, uint8_t temp)
{
  float V_saturation = (temp - CAL2_T) * (CAL1_V - CAL2_V) / (CAL1_T - CAL2_T) + CAL2_V;
  return (voltage_mv * DO_Table[temp] / V_saturation);
}

void writeFile() //writing something to the SD card
{
  file = SD.open("file.txt", FILE_WRITE); // open "file.txt" to write data; make sure that you want to write in the same file that you created in the setup()
  if (file)
  {
      //file.println("hi!!");
      file.println("Temperature:");
      file.print(temp, 2);
      file.print("^C");
      file.print("    pH:");
      file.print(ph_act);
      file.print("    EC:");
      file.print(ecValue,2);
      file.print("ms/cm");
      file.print("    DO:");
      file.print(dissolved_oxygen,2);
      file.print("mg/L");
      file.close(); // close file

      Serial.println("SD:");
      Serial.println("Temperature:");
      Serial.print(temp, 2);
      Serial.print("^C");
      Serial.print("  pH:");
      Serial.println(ph_act);
      Serial.println("  EC:");
      Serial.println(ecValue,2);
      Serial.print("ms/cm");
      Serial.print("  DO:");
      Serial.print(dissolved_oxygen,2);
      Serial.print("mg/L");

      Serial1.println("SD:");
      Serial1.println("Temperature:");
      Serial1.print(temp, 2);
      Serial1.print("^C");
      Serial1.print("  pH:");
      Serial1.println(ph_act);
      Serial1.println("  EC:");
      Serial1.println(ecValue,2);
      Serial1.print("ms/cm");
      Serial1.print("  DO:");
      Serial1.print(dissolved_oxygen,2);
      Serial1.print("mg/L");
     
      file.close(); // cl
  }
  else
  {
    Serial.println("Could not open file (writing).");
    Serial1.println("Could not open file (writing).");
  }
}

/*void readFile()
{
  file = SD.open("file.txt", FILE_READ); // open "file.txt" to read data
  if (file) {
    Serial.println("--- Reading start ---");
    Serial1.println("--- Reading start ---");
    char character;
    while ((character = file.read()) != -1) { // this while loop reads data stored in "file.txt" and prints it to serial monitor
      Serial.print(character);
      Serial1.print(character);
    }
    file.close();
    Serial.println("--- Reading end ---");
    Serial1.println("--- Reading end ---");
  } else {
    Serial.println("Could not open file (reading).");
    Serial1.println("Could not open file (reading).");
  }
}*/