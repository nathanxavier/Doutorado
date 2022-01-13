Vector3 v(1, 2, 3);                             // 3D vertex to rotate
Vector3 r(0.57735f, 0.57735f, 0.57735f);        // rotation axis (unit vector)
float a = 45.0f;                                // rotation angle in degree

// convert to quaternions
Quaternion p = Quaternion(0, v.x, v.y, v.z);    // quaternion form of v
Quaternion q = Quaternion(r, a * 0.5f * D2R);   // rotation quaternion with half-angle
Quaternion c = q;                               // copy of q
c.conjugate();                                  // q* (conjugate of q)

// rotate p by multiplying qpq*
Quaternion p2 = q * p * c;

// vector part of p2 contains the rotated 3D vertex
Vector3 v2(p2.x, p2.y, p2.z);                   // quaternion to vector
std::cout << "v2: " << v2 << std::endl;         // print the result

// OR, convert quaternion to 4x4 roatation matrix
Matrix4 m = q.getMatrix();
v2 = m * v;                                     // rotation using matrix instead

// OR, use matrix rotation directly
Matrix4 m
m.rotate(a, r);                                 // rotate A degree along R axis
v2 = m * v;                                     // rotation using matrix instead