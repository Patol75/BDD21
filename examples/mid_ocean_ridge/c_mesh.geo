dimx = 1980e3;
plumecentre = dimx/2.;
res = 10e3;
Point(1) = {0, 0, 0, res*10};
Point(2) = {dimx, 0, 0, res*10};
Point(3) = {dimx, 660e3, 0, res/1.2};
Point(4) = {dimx/2., 660e3, 0, res/3.};
Point(5) = {0, 660e3, 0, res/1.2};
Point(6) = {plumecentre-50e3, 0e3, 0, res/1.5};
Point(7) = {plumecentre+50e3, 0e3, 0, res/1.5};
Point(8) = {plumecentre-25e3, 0e3, 0, res/1.5};
Point(9) = {plumecentre+25e3, 0e3, 0, res/1.5};
//+
Line(1) = {1, 6};
//+
Line(2) = {6, 8};
//+
Line(3) = {8, 9};
//+
Line(4) = {9, 7};
//+
Line(5) = {7, 2};
//+
Line(6) = {2, 3};
//+
Line(7) = {3, 4};
//+
Line(8) = {4, 5};
//+
Line(9) = {5, 1};
//+
Line Loop(1) = {8, 9, 1, 2, 3, 4, 5, 6, 7};
//+
Plane Surface(1) = {1};
//+
Physical Line(1) = {1};
//+
Physical Line(2) = {2};
//+
Physical Line(3) = {3};
//+
Physical Line(4) = {4};
//+
Physical Line(5) = {5};
//+
Physical Line(6) = {6};
//+
Physical Line(7) = {7};
//+
Physical Line(8) = {8};
//+
Physical Line(9) = {9};
//+
Physical Surface(10) = {1};
