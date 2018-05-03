#include <SparkFunMPU9250-DMP.h>


#define SerialPort SerialUSB
#define BUFFER_LEN 32

#define CPU_HZ 48000000
#define TIMER_PRESCALER_DIV 1
// This is an C/C++ code to insert repetitive code sections in-line pre-compilation
// Wait for synchronization of registers between the clock domains
// ADC
static __inline__ void ADCsync() __attribute__((always_inline, unused));
static void   ADCsync() {
  while (ADC->STATUS.bit.SYNCBUSY == 1); //Just wait till the ADC is free
}

MPU9250_DMP imu;
int a =A0;  //This is the analog pin to read
volatile uint32_t StartTime = 0; //vars to track how long sampling takes
volatile uint32_t EndTime = 0;
volatile uint32_t valBuffer[BUFFER_LEN]; //buffers to store data
volatile float accelXBuffer[BUFFER_LEN];
volatile float accelYBuffer[BUFFER_LEN];
volatile float accelZBuffer[BUFFER_LEN];
volatile int buffer_i =0; //buffer index

uint32_t clockFreq = 1000000; // gives 976.5625 hz sample rate

void setup() {
  pinMode(A0,INPUT);
 analogReadResolution(12);
  SerialPort.begin(115200);

  if (imu.begin() != INV_SUCCESS)
  {
    while (1)
    {
      SerialPort.println("Unable to communicate with MPU-9250");
      SerialPort.println("Check connections, and try again.");
      SerialPort.println();
      delay(5000);
    }
  }

  // Enable all sensors, and set sample rates to 4Hz.
  // (Slow so we can see the interrupt work.)
  //imu.setSensors(INV_XYZ_GYRO | INV_XYZ_ACCEL | INV_XYZ_COMPASS); //Enables all sensors
  imu.setSensors(INV_XYZ_ACCEL);//Enables only accelerometer
  imu.setSampleRate(1000); // Set accel/gyro sample rate to 4Hz


    //###################################################################################
  // ADC setup stuff
  //###################################################################################
  ADCsync();
  ADC->INPUTCTRL.bit.GAIN = ADC_INPUTCTRL_GAIN_1X_Val;      // Gain select as 1X
  ADC->REFCTRL.bit.REFSEL = ADC_REFCTRL_REFSEL_INTVCC0_Val; //  2.2297 V Supply VDDANA

  
  // Set sample length and averaging
  ADCsync();
  ADC->AVGCTRL.reg = 0x00 ;       //Single conversion no averaging
  ADCsync();
  ADC->SAMPCTRL.reg = 0x0A;  ; //sample length in 1/2 CLK_ADC cycles Default is 3F
  
  //Control B register
  int16_t ctrlb = 0x400;       // Control register B hibyte = prescale, lobyte is resolution and mode 
  ADCsync();
  ADC->CTRLB.reg =  ctrlb     ; 
  
 startTimer(clockFreq);
}

void loop() {
  // put your main code here, to run repeatedly:
}

void TC3_Handler()  // Interrupt on overflow
{
  TcCount16* TC = (TcCount16*) TC3; // get timer struct
    ADCsync();

   imu.update(UPDATE_ACCEL | UPDATE_GYRO | UPDATE_COMPASS); //update IMU data
       printIMUData(); //read all data into buffers

 if(buffer_i==BUFFER_LEN) { // if buffers are full, write to serial port
    for(int i = 1; i < BUFFER_LEN; i++) {
       SerialPort.println(String(accelXBuffer[i]) + "," +
              String(accelYBuffer[i]) + "," + String(accelZBuffer[i])+", " + String(valBuffer[i]));//+" ,"+String(t));
    }
    buffer_i = 0;
 }
    TC->INTFLAG.bit.OVF = 1;    // writing a one clears the ovf flag
}

/*
 * read all data into buffers
 */
void printIMUData(void)
{  
  // After calling update() the ax, ay, az, gx, gy, gz, mx,
  // my, mz, time, and/or temerature class variables are all
  // updated. Access them by placing the object. in front:
  // Use the calcAccel, calcGyro, and calcMag functions to
  // convert the raw sensor readings (signed 16-bit values)
  // to their respective units.
    valBuffer[buffer_i] = anaRead();
    accelXBuffer[buffer_i] = imu.calcAccel(imu.ax);
    accelYBuffer[buffer_i] = imu.calcAccel(imu.ay);
    accelZBuffer[buffer_i] = imu.calcAccel(imu.az);
    buffer_i = buffer_i+1;

}


//##############################################################################
// Stripped-down fast analogue read anaRead()
// ulPin is the analog input pin number to be read.
////##############################################################################
uint32_t anaRead() {

  ADCsync();
  ADC->INPUTCTRL.bit.MUXPOS = g_APinDescription[a].ulADCChannelNumber; // Selection for the positive ADC input

  ADCsync();
  ADC->CTRLA.bit.ENABLE = 0x01;             // Enable ADC

  ADC->INTFLAG.bit.RESRDY = 1;              // Data ready flag cleared

  ADCsync();
  ADC->SWTRIG.bit.START = 1;                // Start ADC conversion

  while ( ADC->INTFLAG.bit.RESRDY == 0 );   // Wait till conversion done
  ADCsync();
  uint32_t valueRead = ADC->RESULT.reg;

  ADCsync();
  ADC->CTRLA.bit.ENABLE = 0x00;             // Disable the ADC 
  ADCsync();
  ADC->SWTRIG.reg = 0x01;                    //  and flush for good measure
  return valueRead;
}


//##############################################################################
/*
This is a slightly modified version of the timer setup found at:
https://github.com/maxbader/arduino_tools
 */
void startTimer(int frequencyHz) {
  REG_GCLK_CLKCTRL = (uint16_t) (GCLK_CLKCTRL_CLKEN | GCLK_CLKCTRL_GEN_GCLK0 | GCLK_CLKCTRL_ID (GCM_TCC2_TC3)) ;
  while ( GCLK->STATUS.bit.SYNCBUSY == 1 );

  TcCount16* TC = (TcCount16*) TC3;

  TC->CTRLA.reg &= ~TC_CTRLA_ENABLE;

  // Use the 16-bit timer
  TC->CTRLA.reg |= TC_CTRLA_MODE_COUNT16;
  while (TC->STATUS.bit.SYNCBUSY == 1);

  // Use match mode so that the timer counter resets when the count matches the compare register
  TC->CTRLA.reg |= TC_CTRLA_WAVEGEN_MFRQ;
  while (TC->STATUS.bit.SYNCBUSY == 1);

  // Set prescaler to 1024
  TC->CTRLA.reg |= TC_CTRLA_PRESCALER_DIV1024;
  while (TC->STATUS.bit.SYNCBUSY == 1);

  setTimerFrequency(frequencyHz);

  // Enable the compare interrupt
  TC->INTENSET.reg = 0;
  TC->INTENSET.bit.MC0 = 1;

  NVIC_EnableIRQ(TC3_IRQn);

  TC->CTRLA.reg |= TC_CTRLA_ENABLE;
  while (TC->STATUS.bit.SYNCBUSY == 1);
}

void setTimerFrequency(int frequencyHz) {
  int compareValue = (CPU_HZ / (TIMER_PRESCALER_DIV * frequencyHz)) - 1;
  TcCount16* TC = (TcCount16*) TC3;
  // Make sure the count is in a proportional position to where it was
  // to prevent any jitter or disconnect when changing the compare value.
  TC->COUNT.reg = map(TC->COUNT.reg, 0, TC->CC[0].reg, 0, compareValue);
  TC->CC[0].reg = compareValue;
  while (TC->STATUS.bit.SYNCBUSY == 1);
}
