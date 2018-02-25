% AUTHOR: MICHAEL SIKORA
% DATE: 2018.01.13
% PURPOSE: written to test the object oriented design of
% of the microphone platform

clear; clc; clf
addpath('./quaternions');
pr = 2; % plot radius

pointCenter = [0 0 100];
N = 30;
radius = 1;

platform1 = Platform(pointCenter,N,radius);
[X, Y, Z] = platform1.getMics;
loc_mics = [X, Y, Z];

figure(1);
scatter3(loc_mics(:,1),loc_mics(:,2),loc_mics(:,3));
title({'Location of Circlular Microphone Array'});
xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
xlim([-pr+pointCenter(1) pr+pointCenter(1)]);
ylim([-pr+pointCenter(2) pr+pointCenter(2)]);
zlim([-pr+pointCenter(3) pr+pointCenter(3)]);
hold off;

angle = pi/8;
axis = [1 1 0]; axis = axis./sqrt(sum(axis.^2));

% Define quaternion
cos2 = cos(angle/2);
sin2 = sin(angle/2);
quaternion = [cos2 axis(1)*sin2 axis(2)*sin2 axis(3)*sin2];

for qq = 1:16 % number of rotations
    
platform1.rotate(quaternion);
[X, Y, Z] = platform1.getMics;
loc_mics = [X, Y, Z];

figure(2);
scatter3(loc_mics(:,1),loc_mics(:,2),loc_mics(:,3)); hold on;
quiver3(pointCenter(1),pointCenter(2),pointCenter(3),...
    axis(1),axis(2),axis(3)); hold off;
title({'Location of Circlular Microphone Array'});
xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
xlim([-pr+pointCenter(1) pr+pointCenter(1)]);
ylim([-pr+pointCenter(2) pr+pointCenter(2)]);
zlim([-pr+pointCenter(3) pr+pointCenter(3)]);
% xlim([-pr pr]); ylim([-pr pr]); zlim([-pr pr]);
hold off;
pause(0.01);

end
