BLA::Matrix<16, 0> X = {
    0,  //px = Posição X IMU Global
    0,  //py = Posição Y IMU Global
    0,  //pz = Posição Z IMU Global
    
    0,  //vx = Velocidade X IMU
    0,  //vy = Velocidade Y IMU
    0,  //vz = Velocidade Z IMU

    0,  //qx = Quaternio u
    0,  //qx = Quaternio X
    0,  //qy = Quaternio Y
    0,  //qz = Quaternio Z
    
    0,  //bwx = Bias Giroscópio X
    0,  //bwy = Bias Giroscópio Y
    0,  //bwz = Bias Giroscópio Z
    
    0,  //bax = Bias Acelerômetro X
    0,  //bay = Bias Acelerômetro Y
    0   //baz = Bias Acelerômetro Z
};

BLA::Matrix<16, 16> P;
P = X* ~X;    //P = E[(x-x0)(x-x0)']


BLA::Matrix<12, 15> G = {
   -1, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,
    0,-1, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,
    0, 0,-1,  0, 0, 0,  0, 0, 0,  0, 0, 0,
    
    0, 0, 0,  1, 0, 0,  0, 0, 0,  0, 0, 0,
    0, 0, 0,  0, 1, 0,  0, 0, 0,  0, 0, 0,
    0, 0, 0,  0, 0, 1,  0, 0, 0,  0, 0, 0,
    
    0, 0, 0,  0, 0, 0, -C,-C,-C,  0, 0, 0,
    0, 0, 0,  0, 0, 0, -C,-C,-C,  0, 0, 0,
    0, 0, 0,  0, 0, 0, -C,-C,-C,  0, 0, 0,

    0, 0, 0,  0, 0, 0,  0, 0, 0,  1, 0, 0,
    0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0,
    0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1,
    
    0, 0, 0,  1, 0, 0,  0, 0, 0,  0, 0, 0,
    0, 0, 0,  0, 1, 0,  0, 0, 0,  0, 0, 0,
    0, 0, 0,  0, 0, 1,  0, 0, 0,  0, 0, 0,
};
// -C(q) = Matriz de Rotação relevante ao Quaternion (q)

BLA::Matrix<16, 16> Q = {

};

BLA::Matrix<16, 16> R = {

};

