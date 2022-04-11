#------------------------------------------------------------------------------
# Created By  : Piter-N
# Created Date: 2022/Mar
# version ='1.0'
# -----------------------------------------------------------------------------
"""
Coleta de dados da Pixhawk
- RawIMU:
    Tempo [us - microssegundos]
    Eixos Aceleração [mg - mili força G]
        xacc, yacc, zacc
    Eixos Giroscópio [m rad/s - mili radianos/segundo]
        xgyro, ygyro, zgyro
    Eixos Magnetômetro [mG - mili Gauss]
        xmag, ymag, zmag

- Attitude:
    Tempo [ms - milissegundos]
    Ângulos [Rad: {-pi, pi}]
        roll, pitch, yaw
"""
# -----------------------------------------------------------------------------
"""
Referência:
    MAVLINK Common Message Set:
        https://mavlink.io/en/messages/common.html#messages
    DroneKit-Python API Reference:
        https://dronekit-python.readthedocs.io/en/latest/automodule.html#dronekit.VehicleMode
    Example: Create Attribute in App:
        https://dronekit-python.readthedocs.io/en/latest/examples/create_attribute.html
"""
# -----------------------------------------------------------------------------
from dronekit import Vehicle

class RawIMU(object):
    def __init__(self, time_usec=None,
                 xacc=None, yacc=None, zacc=None,
                 xygro=None, ygyro=None, zgyro=None,
                 xmag=None, ymag=None, zmag=None):
        # Tempo [us - microssegundos]
        self.time_usec = time_usec
        
        # Aceleração [mg - mili força G]
        self.xacc = xacc
        self.yacc = yacc
        self.zacc = zacc
        
        # Giroscópio [m rad/s - mili radianos/segundo]
        self.xgyro = zgyro
        self.ygyro = ygyro
        self.zgyro = zgyro
        
        # Magnetômetro [mG - mili Gauss]
        self.xmag = xmag
        self.ymag = ymag
        self.zmag = zmag
        
    def __str__(self):
        return "time_usec={}\n xacc={}\n yacc={}\n zacc={}\n xgyro={}\n ygyro={}\n zgyro={}\n xmag={}\n ymag={}\n zmag={}\n".format(
                self.time_usec,
                self.xacc, self.yacc, self.zacc,
                self.xgyro, self.ygyro, self.zgyro,
                self.xmag, self.ymag, self.zmag)

class MyVehicle(Vehicle):
    def __init__(self, *args):
        super(MyVehicle, self).__init__(*args)
        self._raw_imu = RawIMU()

        @self.on_message('RAW_IMU')
        def listener(self, name, message):
            self._raw_imu.time_usec = message.time_usec
            self._raw_imu.xacc = message.xacc
            self._raw_imu.yacc = message.yacc
            self._raw_imu.zacc = message.zacc
            self._raw_imu.xgyro = message.xgyro
            self._raw_imu.ygyro = message.ygyro
            self._raw_imu.zgyro = message.zgyro
            self._raw_imu.xmag = message.xmag
            self._raw_imu.ymag = message.ymag
            self._raw_imu.zmag = message.zmag

            self.notify_attribute_listeners('raw_imu', self._raw_imu)

    @property
    def raw_imu(self):
        return self._raw_imu

def raw_imu_callback(self, attr_name, value):
    attr_name == 'raw_imu'
    value == self.raw_imu
    return value