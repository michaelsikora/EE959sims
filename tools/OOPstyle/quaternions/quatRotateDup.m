function [ vect_b ] = quatRotateDup( q, vect_a )
%QUATROTATEDUP Substitue function for the quatRotate in the Aerospace toolbox
%   q - the quaternion that describes the coordinate
%       transformation. describes reference frame b relative to a(size 1x4)
%   vect_a - the vector in the starting reference frame a (size 1x3)

% Calculation defined by Madgwick Equation 5
% quaternion ox vecta ox quaternion* = vectb
qconj = [q(1) -q(2) -q(3) -q(4)];

first = quatMult(q,[0 vect_a]);
vect_b = quatMult(first, qconj);
vect_b = vect_b(2:4);

end

