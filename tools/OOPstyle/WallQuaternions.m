clear; clc
addpath('../quaternions');

pr = 2; % plot radius
plotcenter = [0 0 0];

% Define a Rectangle of points
N = 10; N2 = N^2;
[x, y] = meshgrid(linspace(-1,1,N));
xy = [x(:), y(:)];
x = xy(:,1); y = xy(:,2);
z = zeros(N2,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% center point
center = [0 1 0];
angle = pi/2;
axis = [1 0 0]; axis = axis./sqrt(sum(axis.^2));

% Define quaternion
cos2 = cos(angle/2);
sin2 = sin(angle/2);
quaternion = [cos2 axis(1)*sin2 axis(2)*sin2 axis(3)*sin2];

% Rotate all points using quaternion
Lv = zeros(N2,3);
Lv(:,1) = x; Lv(:,2) = y; Lv(:,3) = z;
    
for ii = 1:N2 % iteration through point cloud
    Lv(ii,:) = quatRotateDup(quaternion, [Lv(ii,1) Lv(ii,2) Lv(ii,3)]);
end
Lx = Lv(:,1)+center(1); Ly = Lv(:,2)+center(2); Lz = Lv(:,3)+center(3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot
figure(1);
scatter3(Lx,Ly,Lz,'.k'); hold on;
quiver3(0,0,0,axis(1),axis(2),axis(3),'g'); hold off;
title({'point cloud rotation defined','by quaternion'});
xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
xlim([-pr+plotcenter(1) pr+plotcenter(1)]);
ylim([-pr+plotcenter(2) pr+plotcenter(2)]);
zlim([-pr+plotcenter(3) pr+plotcenter(3)]);

