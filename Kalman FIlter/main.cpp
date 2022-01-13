
#include "Kalman.h"
#include <chrono>

Chrono accelTimer;
Chrono gpsTimer;
Chrono barometerTimer;
	void setup (){
		initAccelerometer();
		initBarometer();
		initGPS();

	}

	void loop (){
		
		if (accelTimer.hasPassed(10)){
		float accel=getAccel();
		predict(accel);

		accelTimer.restart();

		}	
		if (gpsTimer.hasPassed(1000)){
			float gpsAltitude = getGPS();
			updateGPS(gpsAltitude);

			gpsTimer.restart();

		}

		if (barometerTimer.hasPassed(30)){
			float barometer= getBarometer();
			updateBaro(barometer);
			barometerTimer.restart();
		}
		
		float position = getKalmanPosition();

}

int main ()
{

	setup();

	do{
		loop();
	}while();

	return 0;

}