
function err = sikora_platformSRPtestOne(indVarName,source,sampN,show_plots,saveplot)
%SIKORA_PLATFORMSRPM Function to test (SRP) Steered Response Power for 
% multiple dynamic platforms. 
% indVarName - Name of Independent Variable from
%           'ANGLE','RADIUS','CONSTANT'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ONLY ANGLE AND CONSTANT SHOULD WORK FOR NOW
% 2018.02.07

% source - Sound source from 'IMPULSE','MOZART','SINE'
% sampN - number of discretized points to discretize the independent
%           variable with
% show_plots - 1 to show all plots, 0 to show final plots only

% Code modified from Dr. Kevin Donohue's
% testsprimage.m file by Michael Sikora <m.sikora@uky.edu>. The original 
% header for that file is as follows:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This script simulates an impulse-like sound in a field of view (FOV) with a perimeter
%  array around the walls (offset by .25 meters toward the center) of the room of a
%  rectangular room.  In addition 2 white noise sources are simulated outside the FOV
%  on the actual wall of the room (representing fan noise or noise from
%  window).  These sources represent coherent noise.  The strongest signal on the mic
%  array is used for adjusting the noise power to achieve 10 dB SNR.  Therefore all
%  other mic signals have a less than 10 dB SNR.  In addition, a -30 dB white noise
%  signal is added to every mic with respect to the strongest signal to
%  represent low-level system or diffuse noise.
%
%  The simulated signal is then used to create a steered response power
%  (SRP) image using the function SRPFRAMENN.
%  Details and adjustments for the simulation are explained in the comments below
%
%   Written by Kevin D. Donohue (donohue@engr.uky.edu) October 2005.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The motivation to modify the script to a function form is to facilitate
% running the code with a single independent variable to observe the effect.

% Prefix to add to file locations.
fromfile = load('include.mat','toolspath');
if isfield(fromfile, 'toolspath') == 0
    toolspath = '';
else
    toolspath = fromfile.toolspath;
end
prefix = 'images/Fours';

fno = 1;  %  Figure number for plot
fs= 16000;  %  Sample frequency in Hz
sigtot = 1;   %  Number of targets in FOV
numnos = 2;   %  Number of coherent targets on wall perimeter
              %  generate target bandwidths
snr = -300;  %  coherent noise sources SNR to be added relative to strongest target peaks
batar = .6; %  Beta values for PHAT processing
%  Target signal parameters
f12p = 3000;  %  Corresponding upper frequency limit
f11p = 100;  %  Lower frequency limit
mjs_platnum = 3; % number of platforms
mjs_N = 1; % number of microphones per platform
micnum = mjs_platnum*mjs_N;  %  Number of mics in array to be tested
mjs_radius = 0.1; % radius of mic array on platform
mic2micLength = sin(pi/mjs_N)*mjs_radius*2; % distance between two adjacent microphones
%  White noise snr
wgnsnr = -50;
sclnos = 10^(wgnsnr/20);
%  Frequency dependent Attenuation
temp = 28; % Temperature centigrade
press = 29.92; % pressure inHg
hum = 80;  % humidity in percent
dis = 1;  %  Distance in meters (normalized to 1 meter)
prop.freq = fs/2*[0:200]/200;  %  Create 100 point frequency axis
prop.atten =  atmAtten(temp, press, hum, dis, prop.freq);  %  Attenuation vector
prop.c = SpeedOfSound(temp,hum,press);

%  Generate room geometry
%  Opposite corner points for room, also walls are locations for noise source placement
froom = [-3.5 -4 0; 3.5 4 3.5]';  % [x1, y1, z1; x2 y2 z2]'
% Opposite Corner Points of Perimeter for mic placement
fmics = [-3.25 -3.75 0; 3.25 3.75 2]';
%  Room reflection coefficients (walls, floor, ceiling)
bs = [.5 .5 .5 .5 .5 .5];
%  Field of view for reconstructing SRP image (opposite corner points)
fov = [-2.5 -2.5 1.5; 2.5 2.5 1.5]';

%  Time window for frequency domain block processing
trez = 20e-3;  %  In seconds
%  Room Resolution: Step through cartesion grid for mic and sound source
%  plane
rez = .04;  %  In meters

%  All vertcies in image plane
v = [fmics(1:2,1), [fmics(1,1); fmics(2,2)], fmics(1:2,2), [fmics(1,2); fmics(2,1)]];  
v = [v; ones(1,4)*1.5];
vn = [froom(1:2,1), [froom(1,1); froom(2,2)], froom(1:2,2), [froom(1,2); froom(2,1)]];  
vn = [vn; ones(1,4)*1.5];
%  Compute window length in samples for segmenting time signal 
winlen = ceil(fs*trez);
wininc = round(fs*trez/2);  %  Compute increment in sample for sliding time window along
%  Compute grid axis for pixel of the SRP image
gridax = {[fov(1,1):rez:fov(1,2)], [fov(2,1):rez:fov(2,2)], [fov(3,1):rez:fov(3,2)]}; 

%  Compute spacing for equal spacing of perimeter array
spm = (norm(v(:,2)-v(:,1),2)+norm(v(:,3)-v(:,2),2))/(micnum/2 - 2 +4/sqrt(2));
%  Compute starting point on perimeter from the first corner
stp = (spm/sqrt(2))*(norm(v(:,2)-v(:,1),2)/norm(v(:,3)-v(:,2),2));
%  Generate location of perimeter array microphones
mposperim = regmicsperim(v(:,[1,3]),spm,stp);

% Signal position was modified to be stationary for testing.
% It was moved here so the platforms can be set, tilting towards
% the source.
% sigpos = [1; -1; 1.5]; % SIGNAL SOURCE LOCATION
% sigpos = [-1;0;1.5];
sigpos = [0;0;1.5]; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Michael Sikora m.sikora@uky.edu
         
addpath([ toolspath, '/OOPstyle']);
addpath([ toolspath, '/OOPstyle/quaternions']);

% dist2center = [0.5 1];
dist2center = 0.5:0.25:3.5;
% dist2center = [2 2.5];
mjs_platformGroup = Platform([0 0 1.5],mjs_platnum,1);
for pp = 1:mjs_platnum
    mjs_platform(pp) = Platform([0 0 0],mjs_N,1);
end

% Experimental Setup
% sampN = 10; % number of simulation runs
errs = zeros(sampN*length(dist2center),4);

for bb = 1:length(dist2center)
% Microphone Platform centers
mjs_platformGroup.setRadius(dist2center(bb));
% mjs_platformGroup.eulOrient(pi/4,0);
[mjs_X, mjs_Y, mjs_Z] = mjs_platformGroup.getMics();
mjs_pcs = [mjs_X, mjs_Y, mjs_Z];
    
% mjs_pcs = [-3 1.5 1.5; % North on West Wall
%            -3 -1.5 1.5; % South on West Wall
%             3  0   1.5]; % Centered on East Wall

% mjs_pcs = [-dist2center(bb) 1.5 1.5; % North on West Wall
%            -dist2center(bb) -1.5 1.5; % South on West Wall
%             dist2center(bb)  0   1.5]; % Centered on East Wall

        
% mjs_pcs = [ 0.2 dist2center(bb) 1.5;...
%             -0.2 dist2center(bb) 1.5 ];
%     

% Set Independent Variable for test
switch indVarName
    case 'ANGLE'
        angles = linspace(0,pi/2,sampN); % variable
        radii = ones(1,sampN)*mjs_radius; % constant
    case 'RADIUS'
        angles = ones(1,sampN)*pi/2; % constant
        radii = linspace(0.05,0.3,sampN); % variable
    case 'CONSTANT'
        angles = zeros(1,2); % constant
        radii = ones(1,sampN)*0.10; % constant
end

mjs_angle = angles;

% Define Platforms
for pp = 1:mjs_platnum % loop for identical platforms
    mjs_platform(pp).centerAt(mjs_pcs(pp,:));
    mjs_platform(pp).setRadius(radii(1));
    % vector from each mic center to source location
    mjs_pl2src(pp,:) = sigpos-mjs_pcs(pp,:)';
    mjs_pltheta(pp) = atan2(mjs_pl2src(pp,2),mjs_pl2src(pp,1));
%     mjs_pltan2src(pp,:) = cross(mjs_pl2src(pp,:),[0 0 1]);
    % z axis rotation to orient endfire to source;
    mjs_platform(pp).eulOrient(mjs_pltheta(pp),0); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for aa = 1:sampN % Main loop through angles
% for aa = 1:4
aa=1;
[bb, aa]
if aa ~= 1
    for pp = 1:mjs_platnum
        mjs_platform(pp).eulOrient(mjs_pltheta(pp),angles(aa)); 
    end
end

mposplat = zeros(3,micnum);

% Add microphone coordinates to mic position matrix
for pp = 1:mjs_platnum
    [mjs_X, mjs_Y, mjs_Z] = mjs_platform(pp).getMics();
    mposplat(:,(pp-1)*mjs_N+(1:mjs_N)) = [mjs_X, mjs_Y, mjs_Z]'; % Set mic coordinates
end
if show_plots == 1
    % Grid plot of Platform Orientations
    h2 = figure(10);
    plotOrder = [3,2,1,4,5,6]; % Order Counterclockwise
    for pp = 1:mjs_platnum 
        pr = radii(1)*1.1; % plot-radius/box-width;
        [X,Y,Z] = mjs_platform(pp).getMics(); % Platform pp
        subplot(2,2,plotOrder(pp)), scatter3(X,Y,Z); hold on;
        currOrientation = mjs_platform(pp).getOrient('QUATERNION');
        xyzbasis = [1 0 0; 0 1 0; 0 0 1].*radii(1); % define a reference frame
        rotbasis = zeros(size(xyzbasis));
        coordLabel = ['X*';'Y*';'Z*'];
        for rr = 1:3 % Rotate and plot reference frame by platform orientation
            rotbasis(rr,:) = quatRotateDup(currOrientation,xyzbasis(rr,:));      
            quiver3(mjs_pcs(pp,1),mjs_pcs(pp,2),mjs_pcs(pp,3),...
            rotbasis(rr,1),rotbasis(rr,2),rotbasis(rr,3),'->g');  
            text(rotbasis(rr,1)*1.2+mjs_pcs(pp,1),...
                 rotbasis(rr,2)*1.2+mjs_pcs(pp,2),...
                 rotbasis(rr,3)*1.2+mjs_pcs(pp,3),...
                 coordLabel(rr,:),'HorizontalAlignment','center');
        end
        currEulOrientation = mjs_platform(pp).getOrient('EULER');
        title({['Orientation of Platform ', num2str(pp)],...
            ['Oriented by Euler Angles: \psi : ', num2str(currEulOrientation(1)/pi*180),...
            ' \theta : ' num2str(currEulOrientation(2)/pi*180),...
            ' \phi : ' num2str(currEulOrientation(3)/pi*180) ]}); 
        xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
        xlim([-pr pr]+mjs_pcs(pp,1));
        ylim([-pr pr]+mjs_pcs(pp,2));
        zlim([-pr pr]+mjs_pcs(pp,3)); 
        hold off;
    end
    if bb == 1 && saveplot == 1
        saveas(h2,[prefix, sprintf('platformallseries_%d%d.png',bb,aa)]);
    end
end

if show_plots == 1
    % Grid plot of Platform Orientations
    h3 = figure(11);
    pp = 1;
    pr = radii(1)*1.1; % plot-radius/box-width;
    [X,Y,Z] = mjs_platform(pp).getMics(); % Platform pp
    xyzbasis = [1 0 0; 0 1 0; 0 0 1].*radii(1); % define a reference frame
    rotbasis = zeros(size(xyzbasis));
    for vv = 1:4
        subplot(2,2,vv), scatter3(X,Y,Z); hold on;
        xlabel('xaxis'); ylabel('yaxis'); zlabel('zaxis');
        xlim([-pr pr]+mjs_pcs(pp,1));
        ylim([-pr pr]+mjs_pcs(pp,2));
        zlim([-pr pr]+mjs_pcs(pp,3));
        currOrientation = mjs_platform(pp).getOrient('QUATERNION');
        for rr = 1:3 % Rotate and plot reference frame by platform orientation
            rotbasis(rr,:) = quatRotateDup(currOrientation,xyzbasis(rr,:));      
            quiver3(mjs_pcs(pp,1),mjs_pcs(pp,2),mjs_pcs(pp,3),...
            rotbasis(rr,1),rotbasis(rr,2),rotbasis(rr,3),'->g');  
        end
        planenames = ['XYZ';'Y-Z';'X-Z';'X-Y'];
%         axisnum = find(xyzbasis(vv,:) == 1);
        currEulOrientation = mjs_platform(pp).getOrient('EULER');
        title({['Orientation of Platform ', num2str(pp)],...
            ['Oriented by Euler Angles: \psi : ', num2str(currEulOrientation(1)/pi*180),...
            ' \theta : ' num2str(currEulOrientation(2)/pi*180),...
            ' \phi : ' num2str(currEulOrientation(3)/pi*180) ],...
            ['Viewing axes ', num2str(planenames(vv,:)), '.']}); 
        if vv == 1 
            view(-37.5,30);
        else
            view(xyzbasis(vv-1,:)); 
        end
        hold off;
    end
    if bb == 1 && saveplot == 1
        saveas(h2,[prefix, sprintf('platform1series_%d%d.png',bb,aa)]);
    end
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%  Find max distance (delay) over all mic pairs; this represents an upper bound
%  on all required relative delays when scanning over the FOV
%[rm, nm] = size(mposperim);
%prs = mposanaly(mposperim,2);

[rm, nm] = size(mposplat);
prs = mposanaly(mposplat,2);

%  Maximum delay in seconds needed to synchronize in Delay and Sum beamforming
maxmicd = max(prs(:,3));
%  Extra delay time for padding window of data
textra = ceil(fs*maxmicd(1)/prop.c); 
%  Windowing function to taper edge effects in the whitening process
tapwin = flattap(winlen+textra,20);
winrec = tapwin*ones(1,micnum);

%  Simulate target signals
f11s = f11p-0.2*f11p;  %  Compute Stop bands from passbands
f12s = f12p+0.2*f12p;  %  Compute Stop bands from passbands
%  Ensure signal occurs at a late enough time to be included in
%  first window for processing
td = textra/(fs)+15*trez/2;
simsiglen = td+2*(4/(f12p-f11p) + 4/f11p);
% SOURCE
%  Generate target waveforms
switch source
    case 'IMPULSE'
        target = simimp(f11p,f12p,f11s,f12s,td,fs,simsiglen);
        %  Expand to multiple targets if sigtot greater than 1
        target = target*ones(1,sigtot);
    
    case 'MOZART'
        target = audioread('mozart-1.wav');
        target = target(1:4024);
        %  Expand to multiple targets if sigtot greater than 1
        target = target*ones(1,sigtot);

    case 'SINE'
        freq1 = 1000; time = (1:4024)./fs;
        target = sin(2*pi*freq1*time);
        target = target'*ones(1,sigtot);
end

%  Random generation of signal position within FOV
% sigpos = ((fov(:,2)-fov(:,1))*ones(1,sigtot)).*rand(3,sigtot) + fov(:,1)*ones(1,sigtot);
% sigpos = [1.001; 1.4265; 1.5]; % MOVED TO TOP
%  Compute array signals from target
% [sigoutper, taxper] = simarraysigim(target1, fs, sigpos, mposplat, froom, bs, prop);
[sigoutper, taxper] = simarraysigim(target, fs, sigpos, mposplat, froom, bs, prop);
% [sigoutper, taxper] = simarraysigim(target3, fs, sigpos, mposplat, froom, bs, prop);
%  Random generation of coherent noise source positions on wall 
% for knn=1:numnos
%     randv = ceil(rand(1,1)*4);
%     %  Noise source positions
%     sigposn(:,knn) = vn(:,randv) + rand(1)*(vn(:,mod(randv,4)+1)-vn(:,randv));
% end
    sigposn = [-2.6204 3.500; 4.000 -3.6285; 1.5000 1.5000];

% Create coherent white noise source with sample lengths as target signal
[rt,ct] = size(target);
%  generate white noise 
onos = randn(rt,numnos);
%  place white noise target randomly on wall
[nosoutper, taxnosper] = simarraysigim(onos,fs, sigposn, mposplat, froom, bs, prop);

[mxp,cp] = max(max(abs(sigoutper)));  % Max point over all channels
envper = abs(hilbert(sigoutper(:,cp(1))));  % Compute envelope of strongest channel
%  Compute maximum envelope point for reference in SNRs
%  Also location of max point will be used to ensure time window processed includes
%  the target
[perpkpr, rpper] = max(envper);
%  Trim room signals to same length
[siglenper, mc] = size(sigoutper);
[noslenper, mc] = size(nosoutper);
siglen = min([siglenper, noslenper]);
sigoutper = sigoutper(1:siglen,:);
nosoutper = nosoutper(1:siglen,:);
%  Normalize noise power
nosoutper = nosoutper/sqrt(mean(mean(nosoutper.^2)));
%  Add coherent noise to target signals
nos = randn(siglen,mc);
asnr = 10^((snr/20));
nosamp = asnr*perpkpr;
sigoutpera = sigoutper + nosamp*nosoutper + nos*sclnos*perpkpr;
% Initialize signal window index to beginning index, offset to ensure it includes target
% signal
sst = 1+rpper(1)-fix(.9*winlen); 
sed = sst+min([winlen+textra, siglen]);   %  and end window end
%  create tapering window
tapwin = flattap(sed-sst+1,20);  %  One dimensional
wintap = tapwin*ones(1,micnum);  %  Extend to matrix covering all channels
%  Whiten signal (apply PHAT, with beta factor given at the begining)
sigout = whiten(sigoutpera(sst:sed,:).*wintap, batar);
%  Create SRP Image from processed perimeter array signals
im = srpframenn(sigout, gridax, mposplat, fs, prop.c, trez);

%%%%%%%%%%%%%%%
if show_plots == 1
%  Set up figure for plotting
h = figure(fno);
%  Plot SRP image
surf(gridax{1},gridax{2}, im)
colormap(jet); colorbar; axis('xy')
axis([froom(1,1)-.25, froom(1,2)+.25, froom(2,1)-.25, froom(2,2)+.25])
hold on
%  Mark coherenet noise positions
% plot(sigposn(1,:),sigposn(2,:),'xb','MarkerSize', 18,'LineWidth', 2);  %  Coherent noise
%  Mark actual target positions 
plot3(sigpos(1,:),sigpos(2,:),ones(length(sigpos(2,:)))*1,'ok', 'MarkerSize', 18,'LineWidth', 2);
%  Mark microphone positions
plot3(mposplat(1,:),mposplat(2,:),mposplat(3,:),'sr','MarkerSize', 12);


axis('tight')
for iii = 1:mjs_platnum % Label Platform numbers
    mjs_loc = mjs_platform(iii).getCenter();
    text(mjs_loc(1),mjs_loc(2)+0.5,mjs_loc(3), ['Pl', int2str(iii)], 'HorizontalAlignment', 'center')
end

for kn=1:length(mposplat(1,:)) % Label microphones
    text(mposplat(1,kn),mposplat(2,kn),mposplat(3,kn), int2str(kn), 'HorizontalAlignment', 'center')
end

%  Draw Room walls
plot([vn(1,:), vn(1,1)],[vn(2,:), vn(2,1)],'k--'); hold off;
% Label Plot
xlabel('Xaxis Meters')
ylabel('Yaxis Meters')
title(['SRP image (Mics at squares, Target in circle, No Noise sources'] );
%  Plot signal array
figure(fno+1);
offset = zeros(1,micnum); % Initialize offset vector
for km=1:micnum
    %plot offset
    offset(km) = max(abs(sigoutpera([sst:sed],km))) + .1*std(sigoutpera([sst:sed],km));
end
fixoff = max(offset);
offt = 0;
for km=1:micnum
    offt = fixoff +offt;
    plot((sst:sed)/fs,sigoutpera(sst:sed,km)+offt)
    hold on;
end
hold off;
set(gca,'ytick',[])
xlabel('Seconds')
title('Array Signals, Mic 1 is on the bottom')
figure(fno)
%%%%%%%%%%%%%%

end

%%%%%%%%%%%%% Error Analysis %%%%%%%%%%%%%
if show_plots == 1 % plot red peak points on image
    figure(5);
    surf(gridax{1},gridax{2}, im); hold on;
    xlabel('Xaxis [m]');
    ylabel('Yaxis [m]');
    zlabel('Zaxis [m]');
end

% find signal position on grid points
% X Grid point
delta = (gridax{1}(2)-gridax{1}(1));
AA = gridax{1} > (sigpos(1)+delta);
BB = gridax{1} < (sigpos(1)-delta);
CCx = find(AA == BB);
% Y Grid point
delta = (gridax{2}(2)-gridax{2}(1));
AA = gridax{2} > (sigpos(2)+delta);
BB = gridax{2} < (sigpos(2)-delta);
CCy = find(AA == BB);

gridsize = 8; % size of region around peak
% set range in grid points
regx = CCx-gridsize:CCx+gridsize;
regy = CCy-gridsize:CCy+gridsize;
% doesn't catch boundary problems if max peak is near edge.

% get only image in box around source.
imsourcewindow = zeros(gridsize*2+1);
for ll = 1:gridsize*2+1
    for kk = 1:gridsize*2+1
        imsourcewindow(kk,ll) = im(kk+regx(1)-1,ll+regy(1)-1);
    end
end

srpmax = max(imsourcewindow(:));
[ymax xmax] = find(im == srpmax); % get index of max peak
locxmax = gridax{1}(xmax); % get coordinate of max peak
locymax = gridax{2}(ymax);

peakloc = [ locxmax locymax ]; % location of peak in x,y

if show_plots == 1
   plot3(locxmax, locymax, max(im(:)) ,'ok', 'MarkerSize', 18,'LineWidth', 2);
end


gridsize = 8; % size of region around peak
% range
regx = xmax-gridsize:xmax+gridsize;
regy = ymax-gridsize:ymax+gridsize;
% doesn't catch boundary problems if max peak is near edge.

xN = length(gridax{1});
yN = length(gridax{2});
nn = 1;
for ll = 1:xN
    cond1 = length(find(regx ~= ll)) ~= length(regx);
    for kk = 1:yN
        cond2 = length(find(regy ~= kk)) ~= length(regy);
        if ~(cond1 && cond2) % true when ll is in both regions     
            noisevalues(nn) = im(kk,ll);
            nn = nn + 1;
        end
    end
end

avgnoise = mean(noisevalues);
if(avgnoise <= 0)
   avgnoise = eps(1); 
end
err_metric = 20*log10(srpmax/avgnoise); % SNR with average of noise

switch indVarName
    case 'RADIUS'
        errs(sampN*(bb-1)+aa,:) = [radii(aa),dist2center(bb), err_metric, avgnoise];
    case 'ANGLE'
        errs(sampN*(bb-1)+aa,:) = [angles(aa), dist2center(bb), err_metric, avgnoise];
    case 'CONSTANT'
        errs(sampN*(bb-1)+aa,:) = [aa,dist2center(bb), err_metric, avgnoise];
end

if (saveplot == 1)
    saveas(h,sprintf('images/roomsim_%d_%d.png',bb,aa));
end

% end % END of aa loop
% save('error.mat','errs');
endfireError(bb) = errs(sampN*(bb-1)+1,3);
broadsideError(bb) = errs(sampN*(bb-1)+sampN,3);

end % END of bb loop

clear('h','h2','h3');
save('variables.mat');
save('error.mat','errs','endfireError','broadsideError','dist2center');

figure(20);
% scatter(dist2center,endfireError,'ok'); hold on;
scatter(dist2center,broadsideError,'ok'); hold off;
xlabel('distance to source [m]');
ylabel('SNR dB of SRP image'); xlim([dist2center(1) dist2center(end)]);
title({['SNR dB vs. distance to source for rotating platforms (endfire to broadside);'],...
    ['number of Platforms: ', num2str(mjs_platnum), ', Microphones per Platform: ', num2str(mjs_N)]}, 'FontSize', 11);
% legend('endfire','broadside');

% figure(21);
% bb = 1:length(dist2center);
% meannoisesurf = zeros(length(bb),sampN);
% for aa = 1:sampN
%     meannoisesurf(bb,aa) = errs(sampN*(bb-1)+aa,4);
%     snrdbsurf(bb,aa) = errs(sampN*(bb-1)+aa,3);
% %     plot3(dist2center,angles(ones(length(dist2center)).*aa)/pi*180,errs(2*(bb-1)+aa,4),'-k'); hold on;
% end
% surf(dist2center,angles/pi*180,snrdbsurf');
% hold off;
% grid on;
% xlabel('distance to source [m]');
% ylabel('pitch angle in degrees');
% zlabel('SNR dB of SRP image'); xlim([dist2center(1) dist2center(end)]);
% title({['SNR dB vs. distance to source for rotating platforms (endfire to broadside);'],...
%     ['number of Platforms: ', num2str(mjs_platnum), ', Microphones per Platform: ', num2str(mjs_N)],...
%     ['Radii of Platforms: ', num2str(mjs_radius*100),' [cm], Angle discretization: ', num2str(sampN) ]}, 'FontSize', 11);

end