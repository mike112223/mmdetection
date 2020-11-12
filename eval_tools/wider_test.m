% WIDER FACE Evaluation
% Conduct the evaluation on the WIDER FACE validation set. 
%
% Shuo Yang Dec 2015
%
clear;
close all;
addpath(genpath('./plot'));

%evaluate on different settings
setting_name_list = {'easy';'medium';'hard'};
setting_class = 'setting_int';

fprintf('Plot pr curve under overall setting.\n');
dateset_class = 'Test';

% scenario-Int:
seting_class = 'int';
dir_int = sprintf('./plot/baselines/%s/setting_%s',dateset_class, seting_class);
wider_plot(setting_name_list,dir_int,seting_class,dateset_class);
