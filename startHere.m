% Written by Michael Sikora on 2018.02.10
% Experiment 1 script
% Runs simulation for a set of rotations from planar endfire to planar broadside.
clear all; clf;
toolspath = './tools/'; % Add tools location
addpath('../../../AudioArrayToolbox'); % AudioToolbox Path

save('include.mat', 'toolspath'); % Save the toolbox path before running

% Tests to run by uncommenting active test
% sikora_platformSRPtest('ANGLE','IMPULSE',12,1,1); % Run experiment.
% sikora_platformSRPtest2('ANGLE','IMPULSE',2,1,1); % Run experiment.
% sikora_platformSRPtestBOX('ANGLE','IMPULSE',8,1,1); % Run experiment.
% sikora_platformSRPtestFours('ANGLE','IMPULSE',8,1,1); % Run experiment.
% sikora_platformSRPtestOne('ANGLE','IMPULSE',1,1,1); % Run experiment.
% sikora_platformSRPtestSPIN('ANGLE','IMPULSE',8,1,1); % Run experiment.
sikora_platformSRPtestEars('ANGLE','IMPULSE',2,1,1); % Run experiment.