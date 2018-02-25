% Michael Sikora <m.sikora@uky.edu>
% 2018.01.01
% Defines a Platform object which is a circular microphone array

% Updated to accept Euler angles on 2018.02.10

classdef Platform < handle
    properties
        loc_center          % center point coordinates of platform 
        N                   % number of microphones around center
        a                   % distance to a microphone from center point (radius)
        quaternion          % quaternion representing current frame with 
                            % respect to initial frame 
        init_mics           % initial coordinates of microphones in an array
    end
    methods
        % Platform Constructor 1
        function obj = Platform(loc_center,N,a)
            obj.loc_center = loc_center;
            obj.N = N;
            obj.a = a;
            initMics(obj);
        end
        
        % return center point coordinates
        function point = getCenter(obj)
            point = obj.loc_center;
        end
        
        % initialize mic coordinates
        function initMics(obj)
            % currently initialize to floor orientation
            % and populates mic locations
            obj.init_mics = zeros(obj.N,3);
            ii = (1:obj.N)-0.5;
            rootangles = 2*pi/(obj.N) .*(ii)-pi;
            % Level to floor orientation
            obj.init_mics(:,1) = cos(rootangles).*obj.a;
            obj.init_mics(:,2) = sin(rootangles).*obj.a;
%             obj.init_mics(:,3) = zeros(1,obj.N);
            obj.quaternion = [1 0 0 0];
        end
        
        % return microphone coordinate matrix
        function [X, Y, Z] = getMics(obj)
            coordinates = obj.init_mics;
            for ii = 1:obj.N
                coordinates(ii,:) = quatRotateDup(obj.quaternion,...
                    [obj.init_mics(ii,1) obj.init_mics(ii,2) obj.init_mics(ii,3)]);
            end
            X = coordinates(:,1) + obj.loc_center(1);
            Y = coordinates(:,2) + obj.loc_center(2);
            Z = coordinates(:,3) + obj.loc_center(3);
        end
        % Get normal vector by using initial flat to the floor orientation
        % and rotating z-axis normal using the current orientation
        % quaternion
        function normal = getNorm(obj)
            normal = [0 0 obj.a];
            normal = quatRotateDup(obj.quaternion, normal);
        end
        function orientation = getOrient(obj,type)
            switch type
                case 'QUATERNION'
                    orientation = obj.quaternion;
                case 'EULER'
                    q = obj.quaternion;
                    % Euler angles in order psi, theta, phi
                    orientation = [ atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2))...
                                    asin(2*(q(1)*q(3)-q(4)*q(2)))...
                                    atan2(2*(q(1)*q(2)+q(3)*q(4)), 1-2*(q(2)^2+q(3)^2)) ];
            end    
        end
        
        % Rotate from current orientation to new orientation with a roation
        % defined by a quaternion
        function rotate(obj,q)
            obj.quaternion = quatMult(obj.quaternion,q);
        end
        % Orient from initial orientation to new orientation with a rotation 
        % defined by a quaternion.
        function orient(obj,q)
            obj.quaternion = quatMult([1 0 0 0],q);
        end
        
         % Rotate usign Euler Angles
        function eulRotate(obj,psi,theta)
            % using Tait-Bryon naming of angles in which:
            % psi - Heading
            % theta - Altitude
            % phi - Bank
            
            phi = 0;
            % define quaternion
            psi_cos2 = cos(psi/2); psi_sin2 = sin(psi/2);
            theta_cos2 = cos(theta/2); theta_sin2 = sin(theta/2);
            phi_cos2 = cos(phi/2); phi_sin2 = sin(phi/2);
               
            rot_q = [ theta_cos2 * psi_cos2 * phi_cos2 + theta_sin2 * psi_sin2 * phi_sin2...
                      psi_cos2 * theta_cos2 * phi_sin2 - psi_sin2 * theta_sin2 * phi_cos2...
                      theta_sin2 * psi_cos2 * phi_cos2 + psi_sin2 * theta_cos2 * phi_sin2...
                      psi_sin2 * theta_cos2 * phi_cos2 + psi_cos2 * theta_sin2 * phi_sin2 ];
                  
            % pass to function
            obj.rotate(rot_q);
        end
        
        % Orient using Euler Angles
        function eulOrient(obj,psi,theta)
            phi = 0;
            % define quaternion
            psi_cos2 = cos(psi/2); psi_sin2 = sin(psi/2);
            theta_cos2 = cos(theta/2); theta_sin2 = sin(theta/2);
            phi_cos2 = cos(phi/2); phi_sin2 = sin(phi/2);
               
            rot_q = [ theta_cos2 * psi_cos2 * phi_cos2 + theta_sin2 * psi_sin2 * phi_sin2...
                      psi_cos2 * theta_cos2 * phi_sin2 - psi_sin2 * theta_sin2 * phi_cos2...
                      theta_sin2 * psi_cos2 * phi_cos2 + psi_sin2 * theta_cos2 * phi_sin2...
                      psi_sin2 * theta_cos2 * phi_cos2 + psi_cos2 * theta_sin2 * phi_sin2 ];
                  
            % pass to function
            obj.orient(rot_q);
        end
        function centerAt(obj,newCenter)
            obj.loc_center = newCenter;
            initMics(obj);
        end
        function setRadius(obj,newRadius)
            obj.a = newRadius;
            initMics(obj);
        end
    end
end
        