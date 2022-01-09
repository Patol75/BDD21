dim_x = 1980e3;
dim_y = 660e3;
res = 1e4;

Point(1) = {0, 0, 0, 5 * res};
Point(2) = {dim_x, 0, 0, 5 * res};
Point(3) = {dim_x, dim_y, 0, res};
Point(4) = {dim_x / 2, dim_y, 0, res};
Point(5) = {0, dim_y, 0, res};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 1};
For i In { 1 : 5 }
    Physical Line(i) = {i};
EndFor
Line Loop(1) = {1, 2, 3, 4, 5};
Plane Surface(1) = {1};
Physical Surface(10) = {1};
