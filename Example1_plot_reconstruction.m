clear all
close all
clc

cos_num = 6;
sin_num = 6;
load(strcat('reconstructionV1backtrack_',num2str(cos_num),'_',num2str(sin_num),'delta0pz.mat'))
snapshots = [size(coeff_history,2)];
cos_coeffs_ex = [10.0, 1.0, -.5, 2.0, 1.0, -.5];
sin_coeffs_ex = [.2, 1.0, -.5, 2.0, 1.0, -.5];



snapshots = [1, 21, 51, size(coeff_history,2)];

[xx, vals_ex] = create_fun_for_plot(cos_coeffs_ex, sin_coeffs_ex);
fig = figure();
set(gcf, 'Color', 'w')
fsize = 28;
ylevel = -.7;
for kk = snapshots
    plot(xx,vals_ex,'--b','LineWidth',4);
    hold on
    plot(xx,vals_history(:,kk),'-k','LineWidth',4);
    hold off
    ax = gca;
    ax.XLim = [0,4];
    ax.YLim = [0,10];
    ax.FontSize = fsize;
    ax.XTick = 1:1:4;
    ax.GridAlpha = .9;
    ax.GridLineStyle = '--';
    if kk == 1
        lgd = legend('Exact $\tilde{a}$', strcat("Initial guess"),'FontSize',fsize, 'Interpreter','Latex','LineWidth',2);
        lgd.ItemTokenSize = [50, 18];  % [length, height] in points
    else
        lgd = legend('Exact $\tilde{a}$', strcat("Iteration ",num2str(kk-1)),'FontSize',fsize, 'Interpreter','Latex','LineWidth',2);
        lgd.ItemTokenSize = [50, 18];  % [length, height] in points
    end
    
    grid on

    drawnow
    saveas(gcf,strcat('plots/','ex_1_',num2str(kk)),'epsc')
    pause
end


function [xx, vals] = create_fun_for_plot(cos_coeffs, sin_coeffs)
    nn = 1000;
    xx = linspace(0,4,nn);
    vals = zeros(1,nn);
    
    for nn = 1:length(cos_coeffs)
        vals = vals + 1/2*cos_coeffs(nn) * cos((nn-1)*pi/2*xx);
    end
    
    for nn = 1:length(sin_coeffs)
        vals = vals + 1/2*sin_coeffs(nn) * sin(nn*pi/2*xx);
    end
end