height = 1e6;
width = 4e6;
lowMtl = 3.4e5;

Point(1) = {0, height, 0};
Point(2) = {width, height, 0};
Point(3) = {width, lowMtl, 0};
Point(4) = {width, 0, 0};
Point(5) = {0, 0, 0};
Point(6) = {0, lowMtl, 0};

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

Field[1] = Box;
Field[1].VIn = 2e4;
Field[1].VOut = 2e4;
Field[1].XMax = width;
Field[1].XMin = 0;
Field[1].YMax = height;
Field[1].YMin = 0;

Background Field = 1;
