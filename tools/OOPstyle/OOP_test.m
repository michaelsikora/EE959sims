% AUTHOR: MICHAEL SIKORA
% DATE: 2018.01.13
% PURPOSE: written to test the object oriented design of
% of the microphone platform

clear; clc; clf
addpath('./quaternions');

pointCenter = [0 0 0];
N = 50;
radius = 1;

platform1 = Platform(pointCenter,N,radius);
[X, Y, Z] = platform1.getMics();
loc_mics = [X Y Z];

figure(1);
scatter3(loc_mics(:,1),loc_mics(:,2),loc_mics(:,3));
title({'Location of Circlular Microphone Array'});
xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
hold off;

angle = -pi/4;
axis = [-1 0 0]; axis = axis./sqrt(sum(axis.^2));

% Define quaternion
cos2 = cos(angle/2);
sin2 = sin(angle/2);
quaternion = [cos2 axis(1)*sin2 axis(2)*sin2 axis(3)*sin2];

% platform1.rotate(quaternion);
angle1 = pi/2; angle2 = pi/4;
platform1.eulRotate(angle1,angle2);
[X, Y, Z] = platform1.getMics;
loc_mics = [X Y Z];

figure(2);
scatter3(loc_mics(:,1),loc_mics(:,2),loc_mics(:,3)); hold on;
quiver3(pointCenter(1),pointCenter(2),pointCenter(3),...
    axis(1),axis(2),axis(3)); hold off;
title({'Location of Circlular Microphone Array'});
xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
hold off;

