height = 1e6;
width = 4e6;
lowMtl = 3.4e5;
res = 5e3;

Point(1) = {0, height, 0, res};
Point(2) = {width, height, 0, res};
Point(3) = {width, lowMtl, 0, 20 * res};
Point(4) = {width, 0, 0, 40 * res};
Point(5) = {0, 0, 0, 40 * res};
Point(6) = {0, lowMtl, 0, 20 * res};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
For i In { 1 : 6 }
    Physical Line(i) = {i};
EndFor
Line Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};
Physical Surface(1) = {1};
