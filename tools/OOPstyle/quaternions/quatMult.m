function [ q_out ] = quatMult( q_a, q_b )
%QUATMULT quaternion multiplication of two quaternions
%   operation on inputs is q_out = q_a ox q_b where ox is
%   the quaternion multiplication

% Operation defined by Madgwick Equation 4

% Validate dimensions (Optional)
sa = size(q_a);
sb = size(q_b);
sideal = [1 4];

if sa(1) == 4
   q_a = q_a'; 
end

if sb(1) == 4
   q_b = q_b'; 
end

cond = (sum(sideal == size(q_a)) == 1) && (sum(sideal == size(q_b)) == 1);

if cond
    fprintf('inputs are not quaternions \n');
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1 = q_a(1); b1= q_b(1);
a2 = q_a(2); b2= q_b(2);
a3 = q_a(3); b3= q_b(3);
a4 = q_a(4); b4= q_b(4);

q_out = [a1*b1 - a2*b2 - a3*b3 - a4*b4...
         a1*b2 + a2*b1 + a3*b4 - a4*b3...
         a1*b3 - a2*b4 + a3*b1 + a4*b2...
         a1*b4 + a2*b3 - a3*b2 + a4*b1];

end

