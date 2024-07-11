%% Description:
% File "SFLIM_Unmixing_Solution.m" is a monolithic function to load and unmix an S-FLIM dataset of a mixture of three dyes.
% The file is unmixed and the result is shown in a figure
% -------------------------------------------------------------------------------------------------------------------------------------
%% NOTES:
% 
% -This version has been tested on a Windows 10 Home (version 1903) operating system.
% -This version has been tested on MATLAB 2019b, although it should work in previous (or more recent) versions as well.
% -It requires Image Processing Toolbox. 
% -------------------------------------------------------------------------------------------------------------------------------------
% Lorenzo Scipioni - scipioni.lorenzo@gmail.com - Laboratory for Fluorescence Dynamics, Biomedical Engineering, University of California Irvine
% Jun 29,2020

%% Select and load S-FLIM dataset
%% [FileName,PathName]=Select_Files('','Select SFLIM_Dataset.mat file','*.mat');
%% load([PathName,FileName{1}])
%% load([PathName,'20200319.mat'])
PathName = './';
load([PathName,'SFLIM_Dataset.mat']);
load([PathName,'20200319_2.mat']);
%% Tau-phasor transformation
[G_tau,S_tau] = PhasorTransform(Results.TRES,1,1);
[G_tau,S_tau] = PhasorTransform_Correction(G_tau,S_tau,0,Calibration.xM);

%% Unmixing Algorithm
Ch_vect = 3:32;
N_comp  = 3;
[G_unmix0,S_unmix0,S,L,U_tau,U_lambda,Param] = Phasor_SFLIM_Unmixing2(Results.TRES,G_tau,S_tau,N_comp,Ch_vect,Ch_vect,0);

%% Plot Data
gap = [0.125 .06];
marg_w = [.05 .05];  %left right
marg_h = [.1 .075]; %lower upper
x_time      = linspace(0,1/Results.Acquisition_Param.Freq*Results.Acquisition_Param.Freq_Factor*1e9,255);
x_lambda    = SpectralDetector_A10766(Ch_vect);
figure

% s1 = subtightplot(2,2,1,gap,marg_h,marg_w);
% imagesc(log(Results.TRES)), Figure_Format, axis square, title('TRES (log scale)'), xlabel('\lambda'), ylabel('\tau'), colormap(hot)
s1 = subtightplot(2,2,1,gap,marg_h,marg_w);
plot_rainbow_matrix(x_time',Results.TRES)
legend(strsplit(num2str(SpectralDetector_A10766),' '),'Location','eastoutside','FontSize',11)
set(gca, 'yScale', 'log'), xlabel('time [ns]'), ylabel('N_P_h'), axis tight, title('Lifetime (Spectral Channels)')

s3 = subtightplot(2,2,3,gap,marg_h,marg_w);
hold on
plot(G_unmix0(1),S_unmix0(1),'o','MarkerSize',20,'LineWidth',2,'MarkerEdgeColor','b')
plot(G_unmix0(2),S_unmix0(2),'o','MarkerSize',20,'LineWidth',2,'MarkerEdgeColor','g')
plot(G_unmix0(3),S_unmix0(3),'o','MarkerSize',20,'LineWidth',2,'MarkerEdgeColor','r')
plot_rainbow(G_tau(Ch_vect),S_tau(Ch_vect),'o-')
plot_PhasorCircle, axis image, xlim([0 1]), ylim([0 0.5]), xlabel('g^(^\tau^)'), ylabel('s^(^\tau^)'), title('\tau-phasor plot')
legend('Species #1','Species #2','Species #3','Location','northeast')

s2 = subtightplot(2,2,2,gap,marg_h,marg_w);
hold on
plot(x_time,L(:,1),'b')
plot(x_time,L(:,2),'g')
plot(x_time,L(:,3),'r')
set(gca, 'yScale', 'log'), xlabel('time [ns]'), ylabel('N_P_h'), axis tight, title('Lifetime (Unmixed)')
legend('Species #1','Species #2','Species #3','Location','northeastoutside')

s4 = subtightplot(2,2,4,gap,marg_h,marg_w);
hold on
plot(x_lambda,S(1,:),'b')
plot(x_lambda,S(2,:),'g')
plot(x_lambda,S(3,:),'r')
xlabel('\lambda [nm]'), ylabel('N_P_h'), axis tight, title('Spectra (Unmixed)')
legend('Species #1','Species #2','Species #3','Location','northeastoutside')
figureFullScreen(gcf)
pause(3)
s3.Position(3:4) = s2.Position(3:4);

%% Additional Functions

function y = DoubleGenLogistic_fx(x,I1minus,I1plus,I2plus,shift1,Growth1,n1,shift2,Growth2,n2)

y2 = I1minus + (I1plus - I1minus)./(1+exp(-Growth1.*(x-shift1))).^(1./n1);
y = y2  + (I2plus - y2)./(1+exp(-Growth2.*(x-shift2))).^(1./n2);

end
function Figure_Format_Graph()

FigHandle = gcf;
set(FigHandle, 'Position', [450, 10, 800, 800]);

end
function hOutFigure=figureFullScreen(varargin)
%% figureFullScreen
% Similar to figure function, but allowes turning the figure to full screen, or back to
% deafult dimentions. Based on hidden Matlab features described at
% http://undocumentedmatlab.com/blog/minimize-maximize-figure-window/
%
%% Syntax
% figureFullScreen;
% figureFullScreen(hFigure);
% figureFullScreen(hFigure,isMaximise);
% hFigure=figureFullScreen;
% hFigure=figureFullScreen(hFigure);
% hFigure=figureFullScreen(hFigure,isMaximise);
% figureFullScreen(hFigure,'PropertyName',propertyvalue,...);
% figureFullScreen(hFigure,isMaximise,'PropertyName',propertyvalue,...);
% hFigure=figureFullScreen(hFigure,'PropertyName',propertyvalue,...);
% hFigure=figureFullScreen(hFigure,isMaximise,'PropertyName',propertyvalue,...);
%
%% Description
% This functions wrpas matlab figure function, allowing the user to set the figure size to
% one of the two following states- full screen or non full screen (figure default). As
% figure function results with non-maximized dimentions, the default in this function if
% full screen figure (otherwise there is no reason to use this function). It also can be
% used to resize the figure, in a manner similar to clicking the "Maximize"/"Restore down"
% button with the mouse. 
% All Figure Properties acceoptable by figure function are supported by this function as
% well.
%
%% Input arguments (defaults exist):
%  hFigure- a handle to a figure.
%  isMaximise- a flag, which sets figure to full screen when true, and to default size
%           when false (can be used to toggle figure size).
%  Figure Properties- vriety of properties described at matlab Help.
%
%% Output arguments
%  hOutFigure- a handle to a figure.
%
%% Issues & Comments 
% - Currently results in a warning message that (curently desabled), which implies that
% this functionality will not be supported n future matlab rleeases. - Resulted with an
% error which is overriden if some time passes between handle initisialization and feature
% setting- so pause(0.3) is used.
%
%% Example
% figure('Name','Default figure size');
% plotData=peaks(50);
% surf(plotData);
% title('Default figure size','FontSize',14);
% 
% figureFullScreen('Name','Full screen figure size');
% surf(plotData);
% title('Full screen figure size','FontSize',14);
%
%% See also
%  - figure
%
%% Revision history
% First version: Nikolay S. 2011-06-07.
% Last update:   Nikolay S. 2011-06-14.
%
% *List of Changes:*

%% Organize inputs
if nargin==0
   hInFigure=figure;
   isMaximise=true;
else
   isFigureParam=true(1,nargin);
   for iVarargin=1:min(2,nargin) % we asusme figure handle and isMaximise are amoung the 
      % first two inputs
      currVar=varargin{iVarargin};
      if ishandle(currVar(1)) && strcmpi(get(currVar,'Type'),'figure')
         hInFigure=currVar; % a handle 
         isFigureParam(iVarargin)=false;
      elseif islogical(currVar) % I assume 
         isMaximise=currVar;
         isFigureParam(iVarargin)=false;
      end % if ishandle(currVar) && strcmpi(get(currVar,'Type'),'figure')
   end % for iVarargin=1:nargin
   
   figureParams=varargin(isFigureParam);
   if isempty(figureParams)
      % Default value of some figure property- this will save some if condiiton in the
      % code later on
      figureParams={'Clipping','on'}; 
   end
end % if nargin==0

if exist('hInFigure','var')~=1
   hInFigure=figure(figureParams{:});
end

if exist('isMaximise','var')~=1
   isMaximise=true;
end
   
%% Actually the code (yep, as simple as that)
warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame'); % disable warning message
jFrame = get(hInFigure,'JavaFrame');
pause(0.3);    % unless pause is used error accures orm time to time. 
               % I guess jFrame takes some time to initialize
set(jFrame,'Maximized',isMaximise);
warning('on','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame'); % enable warning message
                                    % to prevent infualtion of additional places in Matlab
figure(hInFigure); % make hInFigure be the current one
if nargout==1 
   hOutFigure=hInFigure; % prepare output figure handle, if needed
end
end
function [g_new,s_new,PCA_param] = PCA_gs(G,S)

x = (G);
y = (S);
mx = mean(x);
my = mean(y);

vect_pca = [x'-mx,y'-my];
[coeff,~] = pca(vect_pca);

th = pi/4;
rotM = [sin(th) cos(th); -cos(th) sin(th)];

gs_new = (rotM*(vect_pca*coeff)');
PCA_param = struct;
PCA_param.rotM = rotM;
PCA_param.mx = mx;
PCA_param.my = my;
PCA_param.coeff = coeff;
PCA_param.function = 'PCA_param.rotM*([x-PCA_param.mx,y-PCA_param.my]*PCA_param.coeff)';
g_new = gs_new(1,:);
s_new = gs_new(2,:);

end
function [Gc,Sc,Pc,Mc] = PhasorTransform_Correction(G,S,dP,xM)

P=atan2_2pi(S,G);
M=(sqrt(S.^2+G.^2));

Pc=P-dP;
Mc=M./xM;
Sc = Mc.*sin(Pc);
Gc = Mc.*cos(Pc);
end
function [f1,f2]=Phasor_Unmixing2comp_distance(Y1,A1,A2)

if nargin==2
    A2 = A1(2);
    A1 = A1(1);
end

[X,Y] = size(Y1);
if X>1&&Y>1
    Y1 = Y1(1:end);
end

M = [real(A1) real(A2); imag(A1) imag(A2); 1 1];
N = cat(1, real(Y1), imag(Y1), ones(1,size(Y1,2)));

f = M\N;
f1 = f(1,:);
f2 = f(2,:);

if X>1&&Y>1
    f1 = reshape(f1,X,Y);
    f2 = reshape(f2,X,Y);
end
end
function [f1,f2,f3]=Phasor_Unmixing3comp_distance(Y,A1,A2,A3)

if nargin==2
    A3 = A1(3);
    A2 = A1(2);
    A1 = A1(1);
end

M = [real(A1) real(A2) real(A3); imag(A1) imag(A2) imag(A3); 1 1 1];
N = cat(1, real(Y), imag(Y), ones(1,size(Y,2)));

f = M\N;
f1 = f(1,:);
f2 = f(2,:);
f3 = f(3,:);
end
function Spectral_Wavelengths = SpectralDetector_A10766(vect)

if nargin == 0
Spectral_Wavelengths = [404.3 413.4 422.4 431.4 440.4 449.3 458.3 467.2 476.2 485.0 493.9 502.7 511.5 520.3 529.0 537.7 546.3 554.9 563.5 572.0 580.4 588.9 597.2 605.5 613.8 622.0 630.1 638.2 646.2 654.1 662.0 669.8];
else
Spectral_Wavelengths = [404.3 413.4 422.4 431.4 440.4 449.3 458.3 467.2 476.2 485.0 493.9 502.7 511.5 520.3 529.0 537.7 546.3 554.9 563.5 572.0 580.4 588.9 597.2 605.5 613.8 622.0 630.1 638.2 646.2 654.1 662.0 669.8];
Spectral_Wavelengths = Spectral_Wavelengths(vect);
end

end
function th=atan2_2pi(y,x)

th=atan2(y,x);
th=th.*(th>=0)+(th+2*pi).*(th<0); 
end
function [param,func_g,func_s]=fit_DoubleGenLogistic_GS_fx(x,g,s,Display,param0)

if nargin<5
if nargin<4
    Display = 0;
end
g_plusInf1   = g(round(length(g)/2));
g_minusInf1  = g(1);
g_plusInf2   = g(end);
s_plusInf1   = s(round(length(s)/2));
s_minusInf1  = s(1);
s_plusInf2   = s(end);
center1      = x(round(length(g)/3));
center2      = x(round(length(g)/3*2));
steepness1   = 1/length(g)*30;
steepness2   = 1/length(g)*30;
n1           = 1;
n2           = 1;
%f_minusInf2
param0 = [g_minusInf1,g_plusInf1,g_plusInf2,s_minusInf1,s_plusInf1,s_plusInf2,center1,steepness1,n1,center2,steepness2,n2];
else
    if isstruct(param0)
        param0 = [param0.g_minusInf1,param0.g_plusInf1,param0.g_plusInf2,param0.s_minusInf1,param0.s_plusInf1,param0.s_plusInf2,param0.center1,param0.steepness1,param0.n1,param0.center2,param0.steepness2,param0.n2];    
    end
end

options = optimset('MaxIter',1000*1000*length(x),'MaxFunEvals',1000*1000*length(x));
[param1, ~]=fminsearch(@(Param)sum(sum((func_fit_g(x,Param)-g).^2+(func_fit_s(x,Param)-s).^2)),param0,options);
func_g = func_fit_g(x,param1);
func_g0 = func_fit_g(x,param0);
func_s = func_fit_s(x,param1);
func_s0 = func_fit_s(x,param0);

if Display == 1
    
    figure
    subplot(1,2,1)
    hold on
    plot(x,g)
    plot(x,func_g,'--r')
    plot(x,func_g0,'-m')
    Figure_Format_Graph
    subplot(1,2,2)
    hold on
    plot(x,s)
    plot(x,func_s,'--r')
    plot(x,func_s0,'-m')
    Figure_Format_Graph
    
end
param.g_minusInf1 = param1(1);
param.g_plusInf1 = param1(2);
param.g_plusInf2 = param1(3);
param.s_minusInf1 = param1(1+3);
param.s_plusInf1 = param1(2+3);
param.s_plusInf2 = param1(3+3);
param.center1 = param1(4+3);
param.steepness1 = param1(5+3);
param.n1 = param1(6+3);
param.center2 = param1(7+3);
param.steepness2 = param1(8+3);
param.n2 = param1(9+3);


end
function func_fit_g=func_fit_g(x,Param)

func_fit_g= DoubleGenLogistic_fx(x,Param(1),Param(2),Param(3),Param(4+3),Param(5+3),Param(6+3),Param(7+3),Param(8+3),Param(9+3));

end
function func_fit_s=func_fit_s(x,Param)

func_fit_s= DoubleGenLogistic_fx(x,Param(1+3),Param(2+3),Param(3+3),Param(4+3),Param(5+3),Param(6+3),Param(7+3),Param(8+3),Param(9+3));

end
function [param,func_g,func_s]=fit_GenLogistic_GS(x,g,s,Display,param0)

if nargin<5
if nargin<4
    Display = 0;
end
g_plusInf1   = g(round(length(g)/2));
g_minusInf1  = g(1);
s_plusInf1   = s(round(length(s)/2));
s_minusInf1  = s(1);
center1      = x(round(length(g)/3));
steepness1   = 2;
n1           = 0.5;
%f_minusInf2
param0 = [g_minusInf1,g_plusInf1,s_minusInf1,s_plusInf1,center1,steepness1,n1];
else
param0 = [param0.g_minusInf1,param0.g_plusInf1,param0.s_minusInf1,param0.s_plusInf1,param0.center1,param0.steepness1,param0.n1];    
end

options = optimset('MaxIter',1000*1000*length(x),'MaxFunEvals',1000*1000*length(x));
[param1, ~]=fminsearch(@(Param)sum(sum((func_fit_g(x,Param)-g).^2+(func_fit_s(x,Param)-s).^2)),param0,options);
func_g =  func_fit_g(x,param1);
func_g0 = func_fit_g(x,param0);
func_s =  func_fit_s(x,param1);
func_s0 = func_fit_s(x,param0);

if Display == 1
    
    figure
    subplot(1,2,1)
    hold on
    plot(x,g)
    plot(x,func_g,'--r')
    plot(x,func_g0,'-m')
    Figure_Format_Graph
    subplot(1,2,2)
    hold on
    plot(x,s)
    plot(x,func_s,'--r')
    plot(x,func_s0,'-m')
    Figure_Format_Graph
    
end
param.g_minusInf1 = param1(1);
param.g_plusInf1 = param1(2);
param.s_minusInf1 = param1(3);
param.s_plusInf1 = param1(4);
param.center1 = param1(5);
param.steepness1 = param1(6);
param.n1 = param0(7);

end
function [FileName,PathName]=Select_Files(StartingFolder,String,FileType)
if nargin<3
    FileType = '*';
if nargin<3
    String = 'Select File';
if nargin<1
    StartingFolder=[];
end
end
end
[FileName,PathName,~]=uigetfile('MultiSelect','on',[StartingFolder,FileType],String);
if ischar(FileName)
    tmp=FileName;
    FileName=cell(1);
    FileName{1}=tmp;
end

if iscell(FileName)==0
    FileName=cell(1);
    FileName{1} = '';
end
if ischar(PathName)==0
    PathName='';
end

    
end
function [G_unmix0,S_unmix0,S,L,U_tau,U_lambda,Param] = Phasor_SFLIM_Unmixing2(TRES,G_tau,S_tau,n_comp,Ch_vect,Ch_vect_spectra,Display,param0)

if nargin<8, param0 = struct; end
if nargin<7, Display = 0; end
if nargin<6, Ch_vect_spectra = Ch_vect; end
if nargin<5, Ch_vect = 1:length(G_tau); end

th = pi/4;
rotM = [sin(th) cos(th); -cos(th) sin(th)];
[g_new,s_new,PCA_param] = PCA_gs(G_tau(Ch_vect),S_tau(Ch_vect));
if n_comp == 2
    % Define starting parameters
    if nargin>=7
    if isfield(param0,'g_plusInf1')
        tmp = PCA_param.rotM*([param0.g_plusInf1-PCA_param.mx,param0.s_plusInf1-PCA_param.my]*PCA_param.coeff)'; 
        param0.g_plusInf1 = tmp(1);
        param0.s_plusInf1 = tmp(2);
    else
        param0.g_plusInf1 = g_new(end);
        param0.s_plusInf1 = s_new(end);
    end
    if isfield(param0,'g_minusInf1') 
        tmp = PCA_param.rotM*([param0.g_minusInf1-PCA_param.mx,param0.s_minusInf1-PCA_param.my]*PCA_param.coeff)'; 
        param0.g_minusInf1 = tmp(1);
        param0.s_minusInf1 = tmp(2);
    else
        param0.g_minusInf1 = g_new(1);
        param0.s_minusInf1 = s_new(1);
    end
    if isfield(param0,'center1')==0, param0.center1      = (round(length(g_new)/3)); end
    if isfield(param0,'steepness1')==0, param0.steepness1   = 2; end
    if isfield(param0,'n1')==0, param0.n1           = 0.5; end

        Param = fit_GenLogistic_GS(1:length(Ch_vect),g_new,s_new,0,param0);
    else
        Param = fit_GenLogistic_GS(1:length(Ch_vect),g_new,s_new,0);
    end
    gs_unmixed_new = [Param.g_minusInf1 Param.g_plusInf1;Param.s_minusInf1 Param.s_plusInf1];
end

if n_comp == 3
    if nargin>=7
    % Define starting parameters
    if isfield(param0,'g_minusInf1') 
        tmp = PCA_param.rotM*([param0.g_minusInf1-PCA_param.mx,param0.s_minusInf1-PCA_param.my]*PCA_param.coeff)'; 
        param0.g_minusInf1 = tmp(1);
        param0.s_minusInf1 = tmp(2);
    else
        param0.g_minusInf1 = g_new(1);
        param0.s_minusInf1 = s_new(1);
    end
    if isfield(param0,'g_plusInf1')
        tmp = PCA_param.rotM*([param0.g_plusInf1-PCA_param.mx,param0.s_plusInf1-PCA_param.my]*PCA_param.coeff)'; 
        param0.g_plusInf1 = tmp(1);
        param0.s_plusInf1 = tmp(2);
    else
        param0.g_plusInf1 = g_new(round(length(g_new)/2));
        param0.s_plusInf1 = s_new(round(length(g_new)/2));
    end
    if isfield(param0,'g_plusInf2')
        tmp = PCA_param.rotM*([param0.g_plusInf2-PCA_param.mx,param0.s_plusInf2-PCA_param.my]*PCA_param.coeff)'; 
        param0.g_plusInf2 = tmp(1);
        param0.s_plusInf2 = tmp(2);
    else
        param0.g_plusInf2 = g_new(end);
        param0.s_plusInf2 = s_new(end);
    end
    if isfield(param0,'center1')==0, param0.center1      = (round(length(g_new)/3)); end
    if isfield(param0,'steepness1')==0, param0.steepness1   = 1/length(g_new)*30; end
    if isfield(param0,'n1')==0, param0.n1           = 1; end
    if isfield(param0,'center2')==0, param0.center2      = (round(length(g_new)/3*2)); end
    if isfield(param0,'steepness2')==0, param0.steepness2   = 1/length(g_new)*30; end
    if isfield(param0,'n2')==0, param0.n2           = 1; end
%         Param = fit_DoubleGenLogistic_GS_fx(1:length(Ch_vect),g_new,s_new,0,param0);
        Param = fit_DoubleGenLogistic_GS_fx(1:length(Ch_vect),g_new,s_new,Display,param0);
    else
%         Param = fit_DoubleGenLogistic_GS_fx(1:length(Ch_vect),g_new,s_new,0);
         Param = fit_DoubleGenLogistic_GS_fx(1:length(Ch_vect),g_new,s_new,Display);
    end
    gs_unmixed_new = [Param.g_minusInf1 Param.g_plusInf1 Param.g_plusInf2;Param.s_minusInf1 Param.s_plusInf1 Param.s_plusInf2];
end
gs_unmix = ((inv(rotM)*gs_unmixed_new)')*inv(PCA_param.coeff);
G_unmix0 = gs_unmix(:,1)+mean(G_tau(Ch_vect));
S_unmix0 = gs_unmix(:,2)+mean(S_tau(Ch_vect));

if n_comp == 2
[U1_tau,U2_tau] = Phasor_Unmixing2comp_distance(G_tau(Ch_vect_spectra)+1i*S_tau(Ch_vect_spectra),G_unmix0+1i*S_unmix0);
U_tau = cat(1,U1_tau,U2_tau);
else
[U1_tau,U2_tau,U3_tau] = Phasor_Unmixing3comp_distance(G_tau(Ch_vect_spectra)+1i*S_tau(Ch_vect_spectra),G_unmix0+1i*S_unmix0);
%[U1_tau,U2_tau,U3_tau] = Phasor_Unmixing3comp(G_tau+1i*S_tau,G_unmix0(1)+1i*S_unmix0(1),G_unmix0(2)+1i*S_unmix0(2),G_unmix0(3)+1i*S_unmix0(3));
U_tau = cat(1,U1_tau,U2_tau,U3_tau);
U_tau(U_tau<0) = 0;
U_tau(U_tau>1) = 1;
U_tau = U_tau./repmat(sum(U_tau,1),size(U_tau,1),1);
end

S = zeros(size(U_tau));
for i = 1:n_comp
    S(i,:) = sum(TRES(:,Ch_vect_spectra),1).*U_tau(i,:);
end
[G_lambda,S_lambda] = PhasorTransform(TRES(:,Ch_vect_spectra),2);
[G_lambda_pure,S_lambda_pure] = PhasorTransform(S,2);
if n_comp == 2
    [U1_lambda,U2_lambda] = Phasor_Unmixing2comp_distance(G_lambda'+1i*S_lambda',G_lambda_pure+1i*S_lambda_pure);
    U_lambda = cat(2,U1_lambda',U2_lambda');
else
    [U1_lambda,U2_lambda,U3_lambda] = Phasor_Unmixing3comp_distance(G_lambda'+1i*S_lambda',G_lambda_pure+1i*S_lambda_pure);
    U_lambda = cat(2,U1_lambda',U2_lambda',U3_lambda');
end
L = zeros(size(U_lambda));
for i = 1:n_comp
    L(:,i) = sum(TRES(:,Ch_vect_spectra),2).*U_lambda(:,i);
end


if Display == 1
    figure
    subplot(2,2,2)
    plot_rainbow(G_tau(Ch_vect),S_tau(Ch_vect),'-o')
    hold on
    plot(G_unmix0,S_unmix0,'-*k')
    plot_PhasorCircle
    xlim([0 1])
    xlim([0 1])
    Figure_Format_Graph

    subplot(2,2,3)
    plot(S')
    Figure_Format_Graph
    ylim([0 Inf])

    subplot(2,2,4)
    plot(L)
    set(gca,'yscale','log');
    Figure_Format_Graph

    subplot(2,2,1)
    imagesc(real(log(TRES)))
    Figure_Format_Graph
end
end
function [G,S,Ph,M]=PhasorTransform(A,dim,Harmonic,gs_shift)

    if nargin<4
            gs_shift = 0;
            if nargin<3
                Harmonic = 1;
                if nargin<2
                    dim = 1;
                end
            end
    end

gf=fft(A,[],dim);

switch dim
    case 1
        gf=conj(gf(Harmonic+1,:,:,:)./gf(1,:,:,:))+gs_shift;
    case 2
        gf=conj(gf(:,Harmonic+1,:,:)./gf(:,1,:,:))+gs_shift;
    case 3
        gf=conj(gf(:,:,Harmonic+1,:)./gf(:,:,1,:))+gs_shift;
    case 4
        gf=conj(gf(:,:,:,Harmonic+1)./gf(:,:,:,1))+gs_shift;
end

G = real(gf);
S = imag(gf);
Ph=atan2_2pi(S,G);
M=(sqrt(S.^2+real(gf).^2));
end
function plot_rainbow(x,y,linemarker,cmap)

if nargin<4
    cmap = jet(length(x));
if nargin<3
    linemarker = 'o';
end
end
line_flag = strcmp(linemarker,'-');

if isempty(line_flag)
    if length(x) == length(y)
    figure(gcf)
    hold on
    L = length(x);
    for i = 1:L
        plot(x(i),y(i),linemarker,'Color',cmap(i,:))
    end
    else
    fprintf('Vectors have not the same length')
    end
else
    if length(x) == length(y)
    figure(gcf)
    hold on
    L = length(x);
%    cmap = cmap(1:end-1,:);
    for i = 2:L
        plot(x(i-1:i),y(i-1:i),linemarker,'Color',cmap(i-1,:))
    end
    else
    fprintf('Vectors have not the same length')
    end
end
end
function plot_PhasorCircle()

hold on
th = 0:pi/50:2*pi;
xunit = 0.5 * cos(th) + 0.5;
yunit = 0.5 * sin(th);
xunit = xunit(yunit>0);
yunit = yunit(yunit>0);
plot(xunit, yunit,':k');
end
function h=subtightplot(m,n,p,gap,marg_h,marg_w,varargin)
%function h=subtightplot(m,n,p,gap,marg_h,marg_w,varargin)
%
% Functional purpose: A wrapper function for Matlab function subplot. Adds the ability to define the gap between
% neighbouring subplots. Unfotrtunately Matlab subplot function lacks this functionality, and the gap between
% subplots can reach 40% of figure area, which is pretty lavish.  
%
% Input arguments (defaults exist):
%   gap- two elements vector [vertical,horizontal] defining the gap between neighbouring axes. Default value
%            is 0.01. Note this vale will cause titles legends and labels to collide with the subplots, while presenting
%            relatively large axis. 
%   marg_h  margins in height in normalized units (0...1)
%            or [lower uppper] for different lower and upper margins 
%   marg_w  margins in width in normalized units (0...1)
%            or [left right] for different left and right margins 
%
% Output arguments: same as subplot- none, or axes handle according to function call.
%
% Issues & Comments: Note that if additional elements are used in order to be passed to subplot, gap parameter must
%       be defined. For default gap value use empty element- [].      
%
% Usage example: h=subtightplot((2,3,1:2,[0.5,0.2])
% gap = [0.01 .06];
% marg_w = [.1 .05];  %left right
% marg_h = [.1 .03]; %lower upper
% ,gap,marg_h,marg_w

if (nargin<4) || isempty(gap),    gap=0.01;  end
if (nargin<5) || isempty(marg_h),  marg_h=0.05;  end
if (nargin<5) || isempty(marg_w),  marg_w=marg_h;  end
if isscalar(gap),   gap(2)=gap;  end
if isscalar(marg_h),  marg_h(2)=marg_h;  end
if isscalar(marg_w),  marg_w(2)=marg_w;  end
gap_vert   = gap(1);
gap_horz   = gap(2);
marg_lower = marg_h(1);
marg_upper = marg_h(2);
marg_left  = marg_w(1);
marg_right = marg_w(2);

%note n and m are switched as Matlab indexing is column-wise, while subplot indexing is row-wise :(
[subplot_col,subplot_row]=ind2sub([n,m],p);  

% note subplot suppors vector p inputs- so a merged subplot of higher dimentions will be created
subplot_cols=1+max(subplot_col)-min(subplot_col); % number of column elements in merged subplot 
subplot_rows=1+max(subplot_row)-min(subplot_row); % number of row elements in merged subplot   

% single subplot dimensions:
%height=(1-(m+1)*gap_vert)/m;
%axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
height=(1-(marg_lower+marg_upper)-(m-1)*gap_vert)/m;
%width =(1-(n+1)*gap_horz)/n;
%axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
width =(1-(marg_left+marg_right)-(n-1)*gap_horz)/n;

% merged subplot dimensions:
merged_height=subplot_rows*( height+gap_vert )- gap_vert;
merged_width= subplot_cols*( width +gap_horz )- gap_horz;

% merged subplot position:
merged_bottom=(m-max(subplot_row))*(height+gap_vert) +marg_lower;
merged_left=(min(subplot_col)-1)*(width+gap_horz) +marg_left;
pos_vec=[merged_left merged_bottom merged_width merged_height];

% h_subplot=subplot(m,n,p,varargin{:},'Position',pos_vec);
% Above line doesn't work as subplot tends to ignore 'position' when same mnp is utilized
h=subplot('Position',pos_vec,varargin{:});

if (nargout < 1),  clear h;  end

end
function plot_rainbow_matrix(x,y,linemarker)

if nargin<3
    linemarker = '-';
if nargin == 2&&ischar(y)
    linemarker = y;
    y = x;
    x = meshgrid(1:size(y,1),ones(1,size(y,2)))';    
end
if nargin == 1
    y = x;
    x = meshgrid(1:size(y,1),ones(1,size(y,2)))';
end
end
if size(x,2) == 1
    x = repmat(x,1,size(y,2));
end
    figure(gcf)
    hold on
    L = size(x,2);
    co = jet(L);
    for i = 1:L
        plot(x(:,i),y(:,i),linemarker,'Color',co(i,:))
    end
end







        